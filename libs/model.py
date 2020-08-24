import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.utils import *
from libs.autoencoder import encoder3, decoder3, encoder_res18
from libs.deeplab_org import DeepLab_org
from collections import OrderedDict

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
       You probably saved the model using nn.DataParallel, which stores the model in module, and now you are trying to load it
       without DataParallel. You can either add a nn.DataParallel temporarily in your network for loading purposes, or you can
       load the weights file, create a new ordered dict without the module prefix, and load it back
    """
    state_dict_new = OrderedDict()
    #print(type(state_dict))
    for k, v in state_dict.items():
        #print(k)
        name = k[7:] # remove the prefix module.
        # My heart is broken, the pytorch have no ability to do with the problem.
        state_dict_new[name] = v
    return state_dict_new

class NLM_woSoft(nn.Module):
	"""
	Non-local mean layer w/o softmax on affinity
	"""
	def __init__(self):
		super(NLM_woSoft, self).__init__()

	def forward(self, in1, in2):
		n,c,h,w = in1.size()
		in1 = in1.view(n,c,-1)
		in2 = in2.view(n,c,-1)

		affinity = torch.bmm(in1.permute(0,2,1), in2)
		return affinity

def transform(aff, frame1):
	"""
	Given aff, copy from frame1 to construct frame2.
	INPUTS:
	 - aff: (h*w)*(h*w) affinity matrix
	 - frame1: n*c*h*w feature map
	"""
	b,c,h,w = frame1.size()
	frame1 = frame1.view(b,c,-1)
	frame2 = torch.bmm(frame1, aff)
	return frame2.view(b,c,h,w)

class normalize(nn.Module):
	"""Given mean: (R, G, B) and std: (R, G, B),
	will normalize each channel of the torch.*Tensor, i.e.
	channel = (channel - mean) / std
	"""

	def __init__(self, mean, std = (1.0,1.0,1.0)):
		super(normalize, self).__init__()
		self.mean = nn.Parameter(torch.FloatTensor(mean).cuda(), requires_grad=False)
		self.std = nn.Parameter(torch.FloatTensor(std).cuda(), requires_grad=False)

	def forward(self, frames):
		b,c,h,w = frames.size()
		frames = (frames - self.mean.view(1,3,1,1).repeat(b,1,h,w))/self.std.view(1,3,1,1).repeat(b,1,h,w)
		return frames

def create_flat_grid(F_size, GPU=True):
	"""
	INPUTS:
	 - F_size: feature size
	OUTPUT:
	 - return a standard grid coordinate
	"""
	b, c, h, w = F_size
	theta = torch.tensor([[1,0,0],[0,1,0]])
	theta = theta.unsqueeze(0).repeat(b,1,1)
	theta = theta.float()

	# grid is a uniform grid with left top (-1,1) and right bottom (1,1)
	# b * (h*w) * 2
	grid = torch.nn.functional.affine_grid(theta, F_size)
	grid[:,:,:,0] = (grid[:,:,:,0]+1)/2 * w
	grid[:,:,:,1] = (grid[:,:,:,1]+1)/2 * h
	grid_flat = grid.view(b,-1,2)
	if(GPU):
		grid_flat = grid_flat.cuda()
	return grid_flat


class GC(nn.Module):
    def __init__(self, inplanes, planes, kh=7, kw=7):
        super(GC, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, 256, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))
        self.conv_l2 = nn.Conv2d(256, planes, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r1 = nn.Conv2d(inplanes, 256, kernel_size=(1, kw),
                                 padding=(0, int(kw/2)))
        self.conv_r2 = nn.Conv2d(256, planes, kernel_size=(kh, 1),
                                 padding=(int(kh/2), 0))

    def forward(self, x):
        x_l = self.conv_l2(self.conv_l1(x))
        x_r = self.conv_r2(self.conv_r1(x))
        x = x_l + x_r
        return x

class Model_switchGTfixdot_swCC_Res_ZVOS(nn.Module):
	def __init__(self, encoder_dir = None, decoder_dir = None,
					   temp = None, pretrainRes = False, uselayer=4):
		'''
		For switchable concenration loss
		Using Resnet18
		'''
		super(Model_switchGTfixdot_swCC_Res_ZVOS, self).__init__()
		self.gray_encoder = encoder_res18(pretrained = pretrainRes, uselayer=uselayer)
		self.rgb_encoder = encoder3(reduce = True)
		self.nlm = NLM_woSoft()
		self.decoder = decoder3(reduce = True)
		self.temp = temp
		self.softmax = nn.Softmax(dim=1)
		self.cos_window = torch.Tensor(np.outer(np.hanning(40), np.hanning(40))).cuda()
		self.normalize = normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.value_num = 256
		self.key_num = 64
		self.channel = 512
		self.CAM = DeepLab_org(base='densenet169', c_output=1)
		#self.CAM = nn.DataParallel(self.model_org).cuda()
		self.CAM.load_state_dict(convert_state_dict(torch.load('./libs/sal.pth')))
		for param in self.CAM.parameters():
			param.requires_grad = False
		self.pool = nn.MaxPool2d(kernel_size=2)
		self.weak_conv1 = GC(512 * 2, self.value_num)
		self.weak_bn = nn.BatchNorm2d(self.value_num)
		self.weak_conv2_1 = nn.Conv2d(self.value_num, 1, kernel_size=3, stride=1, dilation=6, padding=6)
		self.weak_conv2_2 = nn.Conv2d(self.value_num, 1, kernel_size=3, stride=1, dilation=12, padding=12)
		self.weak_conv2_3 = nn.Conv2d(self.value_num, 1, kernel_size=3, stride=1, dilation=18, padding=18)
		self.prelu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()

		if(not encoder_dir is None):
			print("Using pretrained encoders: %s."%encoder_dir)
			self.rgb_encoder.load_state_dict(torch.load(encoder_dir))
		if(not decoder_dir is None):
			print("Using pretrained decoders: %s."%decoder_dir)
			self.decoder.load_state_dict(torch.load(decoder_dir))

		for param in self.decoder.parameters():
			param.requires_grad = False
		for param in self.rgb_encoder.parameters():
			param.requires_grad = False


	def _make_encoder(self, all_channel, key_num, value_num):
		memory_output = nn.Sequential(
			# nn.Conv2d(all_channel, 256, kernel_size=1, stride=1, bias=False),
			# nn.BatchNorm2d(256, affine=affine_par),
			# nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(all_channel, key_num + value_num,
					  kernel_size=1, stride=2, bias=False),
			nn.BatchNorm2d(key_num + value_num),
			nn.ReLU(inplace=True))

		return memory_output

	def forward(self, gray1, gray1_org, gray1_sal, gray2, color1=None, color2=None, segments=None):
		gray1 = (gray1 + 1) / 2
		gray2 = (gray2 + 1) / 2
		gray1_org = (gray1_org + 1) / 2

		gray1 = self.normalize(gray1)
		gray2 = self.normalize(gray2)
		gray1_org = self.normalize(gray1_org)

		Fgray1 = self.gray_encoder(gray1)
		Fgray2 = self.gray_encoder(gray2)
		Fgray1_org = self.gray_encoder(gray1_org)
		Fgray2_c = Fgray2.clone()
		B, N, C, H, W = segments.size()

		pool_1 = self.pool(Fgray1_org)
		feat_sz = pool_1.size()
		pool_2 = self.pool(Fgray2_c)  # reduce computation
		saleincy_matrix = self.nlm(pool_1, pool_2)
		saleincy_matrix = F.softmax(saleincy_matrix * self.temp, dim=1)
		saleincy_matrix_T = F.softmax(torch.transpose(saleincy_matrix, 1, 2), dim=1)
		Attention_fea = torch.bmm(pool_2.view(-1, feat_sz[1], feat_sz[2] * feat_sz[3]), saleincy_matrix_T).contiguous()
		Attention_fea = Attention_fea.view(-1, feat_sz[1], feat_sz[2], feat_sz[3])
		# print('fea size:', pool_1.size(), Attention_fea.size())
		input1_att = torch.cat([pool_1, Attention_fea], dim=1)
		saliency_map1 = self.weak_conv1(input1_att)
		saliency_map1 = self.weak_bn(saliency_map1)
		saliency_map1 = self.prelu(saliency_map1)
		saliency_map1_1 = self.weak_conv2_1(saliency_map1)  # self.weak_conv2(saliency_map1)
		saliency_map1_2 = self.weak_conv2_2(saliency_map1)
		saliency_map1_3 = self.weak_conv2_3(saliency_map1)  # saliency_map2 = self.weak_conv2(saliency_map2)
		saliency_map1s = sum([saliency_map1_1, saliency_map1_2, saliency_map1_3])
		saliency_map1s = self.sigmoid(saliency_map1s)
		#print('fea size:', Fgray1.size()) 512, 60, 112
		aff = self.nlm(Fgray1, Fgray2)
		aff_norm = self.softmax(aff*self.temp)

		if(color1 is None):
			# for testing
			return aff_norm, Fgray1, Fgray2

		Fcolor1 = self.rgb_encoder(color1)
		Fcolor2 = self.rgb_encoder(color2)
		Fcolor2_est = transform(aff_norm, Fcolor1)
		pred2 = self.decoder(Fcolor2_est)

		Fcolor1_est = transform(aff_norm.transpose(1,2), Fcolor2)
		pred1 = self.decoder(Fcolor1_est)

		return pred1, pred2, aff_norm, aff, Fgray1, Fgray2, saliency_map1s


class Model_switchGTfixdot_swCC_Res(nn.Module):
	def __init__(self, encoder_dir = None, decoder_dir = None,
					   temp = None, pretrainRes = False, uselayer=4):
		'''
		For switchable concenration loss
		Using Resnet18
		'''
		super(Model_switchGTfixdot_swCC_Res, self).__init__()
		self.gray_encoder = encoder_res18(pretrained = pretrainRes, uselayer=uselayer)
		self.rgb_encoder = encoder3(reduce = True)
		self.nlm = NLM_woSoft()
		self.decoder = decoder3(reduce = True)
		self.temp = temp
		self.softmax = nn.Softmax(dim=1)
		self.cos_window = torch.Tensor(np.outer(np.hanning(40), np.hanning(40))).cuda()
		self.normalize = normalize(mean=[0.485, 0.456, 0.406],
								   std=[0.229, 0.224, 0.225])


		if(not encoder_dir is None):
			print("Using pretrained encoders: %s."%encoder_dir)
			self.rgb_encoder.load_state_dict(torch.load(encoder_dir))
		if(not decoder_dir is None):
			print("Using pretrained decoders: %s."%decoder_dir)
			self.decoder.load_state_dict(torch.load(decoder_dir))

		for param in self.decoder.parameters():
			param.requires_grad = False
		for param in self.rgb_encoder.parameters():
			param.requires_grad = False

	def forward(self, gray1, gray2, color1=None, color2=None):
		gray1 = (gray1 + 1) / 2
		gray2 = (gray2 + 1) / 2

		gray1 = self.normalize(gray1)
		gray2 = self.normalize(gray2)


		Fgray1 = self.gray_encoder(gray1)
		Fgray2 = self.gray_encoder(gray2)
		#print('fea size:', Fgray1.size()) 512, 60, 112
		aff = self.nlm(Fgray1, Fgray2)
		aff_norm = self.softmax(aff*self.temp)

		if(color1 is None):
			# for testing
			return aff_norm, Fgray1, Fgray2

		Fcolor1 = self.rgb_encoder(color1)
		Fcolor2 = self.rgb_encoder(color2)
		Fcolor2_est = transform(aff_norm, Fcolor1)
		pred2 = self.decoder(Fcolor2_est)

		Fcolor1_est = transform(aff_norm.transpose(1,2), Fcolor2)
		pred1 = self.decoder(Fcolor1_est)

		return pred1, pred2, aff_norm, aff, Fgray1, Fgray2