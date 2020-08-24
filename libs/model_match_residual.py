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


class Residual_block(nn.Module):
	def __init__(self, inplanes, planes,stride=1):
		super(Residual_block, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,stride=1,padding=1, bias=False )
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=1,padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.stride = stride
		self.inplanes = inplanes
		self.planes = planes
		self.downsample = nn.Sequential(
			nn.Conv2d(inplanes, planes,kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes))

	def forward(self, x):
		identity = x
		#identity = self.downsample(x)
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		if not self.inplanes == self.planes:
			identity = self.downsample(x)
		out += identity

		out = self.relu(out)

		#U_layer = F.upsample(U_layer, scale_factor=2, mode='bilinear')
		#out = out+U_layer
		return out

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
		self.value_num = 128
		self.key_num = 32
		self.channel = 512
		self.CAM = DeepLab_org(base='densenet169', c_output=1)
		# self.CAM = nn.DataParallel(self.model_org).cuda()
		self.CAM.load_state_dict(convert_state_dict(torch.load('./libs/sal.pth')))
		for param in self.CAM.parameters():
			param.requires_grad = False
		self.pool = nn.MaxPool2d(kernel_size=2)
		self.fusion3_1 = Residual_block(self.value_num * 2, self.value_num * 2)
		self.fusion3_2 = Residual_block(self.value_num * 2, self.value_num)
		self.fusion2_1 = Residual_block(self.value_num, self.value_num)
		self.fusion2_2 = Residual_block(self.value_num, self.value_num)
		self.weak_conv1 = GC(self.value_num, self.value_num // 2)
		self.weak_bn = nn.BatchNorm2d(self.value_num // 2)

		# self.weak_conv2 = nn.Conv2d(self.value_num,1,1, bias=True)
		self.weak_conv2_1 = nn.Conv2d(self.value_num // 2, 1, kernel_size=3, stride=1, dilation=6, padding=6)
		self.weak_conv2_2 = nn.Conv2d(self.value_num // 2, 1, kernel_size=3, stride=1, dilation=12, padding=12)
		self.weak_conv2_3 = nn.Conv2d(self.value_num // 2, 1, kernel_size=3, stride=1, dilation=18, padding=18)
		self.memory_encoder = self._make_encoder(self.channel, key_num=self.key_num, value_num=self.value_num)
		self.linear_e = nn.Linear(self.key_num, self.key_num, bias=False)
		self.prelu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()

	def _make_encoder(self, all_channel, key_num, value_num):
		memory_output = nn.Sequential(
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(all_channel, key_num + value_num, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(key_num + value_num),
			nn.ReLU(inplace=True))

		return memory_output

	def forward(self, img_ref, img_org, frame1_sal, img_tar, color1=None, color2=None, segments=None):
		#forward(self, gray1, gray1_org, gray1_sal, gray2, color1=None, color2=None, segments=None):
		n, c, h_ref, w_ref = img_ref.size()
		n, c, h_tar, w_tar = img_tar.size()
		# np.save('im1.npy',img_org.detach().cpu().numpy())

		gray_ref = copy.deepcopy(img_ref[:, 0].view(n, 1, h_ref, w_ref).repeat(1, 3, 1, 1))
		gray_tar = copy.deepcopy(img_tar[:, 0].view(n, 1, h_tar, w_tar).repeat(1, 3, 1, 1))
		gray_org = copy.deepcopy(img_org[:, 0].view(n, 1, h_tar, w_tar).repeat(1, 3, 1, 1))
		# np.save('im1_1.npy', img_org.detach().cpu().numpy())
		# np.save('frame1_sal.npy', frame1_sal.detach().cpu().numpy())
		gray_ref = (gray_ref + 1) / 2
		gray_tar = (gray_tar + 1) / 2
		gray_org = (gray_org + 1) / 2

		gray_ref = self.normalize(gray_ref)
		gray_tar = self.normalize(gray_tar)
		gray_org = self.normalize(gray_org)

		Fgray1 = self.gray_encoder(gray_ref)  # 512*32*32
		Fgray2 = self.gray_encoder(gray_tar)
		Fgray1_org = self.gray_encoder(gray_org)
		layer_feat1s = self.gray_encoder.get_features(gray_org)  # 3,4 layer
		Fgray2_c = Fgray2.clone()
		# np.save('feat2.npy', Fgray2.detach().cpu().numpy())
		Fcolor1 = self.rgb_encoder(img_ref)

		output = []

		input1_att = self.memory_compute(Fgray1_org, segments)  # 4 layer 256 channel
		# print('h_v size, h_v_org size:', torch.max(input1_att), torch.max(exemplar), torch.min(input1_att), torch.max(exemplar))
		input1_att_up = F.upsample(input1_att, scale_factor=2, mode='bilinear')
		layer_fusion1 = input1_att_up + self.fusion3_1(layer_feat1s[-1])  # 256
		layer_fusion1 = self.fusion3_2(layer_fusion1)
		input2_att_up = layer_fusion1  # F.upsample(layer_fusion1, scale_factor=2, mode='bilinear')
		# print('fea size:', input2_att_up.size(), layer_feat1s[-1].size(), layer_feat1s[-2].size())
		layer_fusion2 = input2_att_up + self.fusion2_1(layer_feat1s[-2])  # 128
		layer_fusion2 = self.fusion2_2(layer_fusion2)  # 128
		saliency_map1 = self.weak_conv1(layer_fusion2)  # 128
		saliency_map1 = self.weak_bn(saliency_map1)
		saliency_map1 = self.prelu(saliency_map1)
		# saliency_map1s = self.weak_conv2(saliency_map1)
		saliency_map1_1 = self.weak_conv2_1(saliency_map1)  #
		saliency_map1_2 = self.weak_conv2_2(saliency_map1)
		saliency_map1_3 = self.weak_conv2_3(saliency_map1)  # saliency_map2 = self.weak_conv2(saliency_map2)
		saliency_map1s = sum([saliency_map1_1, saliency_map1_2, saliency_map1_3])
		saliency_map1s = self.sigmoid(saliency_map1s)
		aff = self.nlm(Fgray1, Fgray2)
		aff_norm = self.softmax(aff * self.temp)

		Fcolor1 = self.rgb_encoder(color1)
		Fcolor2 = self.rgb_encoder(color2)
		Fcolor2_est = transform(aff_norm, Fcolor1)
		pred2 = self.decoder(Fcolor2_est)

		Fcolor1_est = transform(aff_norm.transpose(1, 2), Fcolor2)
		pred1 = self.decoder(Fcolor1_est)

		return pred1, pred2, aff_norm, aff, Fgray1, Fgray2, saliency_map1s

	def memory_compute(self, Fgray1_org, segments):
		B, N, C, H, W = segments.size()
		video_frames = [elem.view(B, C, H, W) for elem in segments.split(1, dim=1)]
		video_frames = [copy.deepcopy(video_frames[frame][:, 0].view(B, 1, H, W).repeat(1, 3, 1, 1)) \
						for frame in range(0, 8)]
		video_frames = [(video_frames[frame] + 1) / 2 for frame in range(0, 8)]  # gray_ref = (gray_ref + 1) / 2
		video_frames = [self.normalize(video_frames[frame]) for frame in range(0, 8)]
		querys = [self.gray_encoder(video_frames[frame]) for frame in range(0, 8)]

		querys_en = torch.stack([self.memory_encoder(querys[frame]) for frame in range(0, 8)], dim=2)

		exemplar_en1 = self.memory_encoder(Fgray1_org)
		# np.save('feat.npy', Fgray1_org.detach().cpu().numpy())
		fea_size = exemplar_en1.size()[2:]
		exemplar_key1 = exemplar_en1[:, :self.key_num, :, :].view(-1, self.key_num,
																  fea_size[0] * fea_size[1])  # N,C,H*W
		exemplar_value1 = exemplar_en1[:, self.key_num:, :,
						  :]  # .view(-1, self.value_num, fea_size[0] * fea_size[1])
		# N,C,H*W
		query_flat_key = querys_en[:, :self.key_num, :, :, :].view(-1, self.key_num,
																   8 * fea_size[0] * fea_size[1])
		query_flat_value = querys_en[:, self.key_num:, :, :, :].view(-1, self.value_num,
																	 8 * fea_size[0] * fea_size[1])
		exemplar_t1 = torch.transpose(exemplar_key1, 1, 2).contiguous()  # batch size x dim x num
		# print('fea size:', exemplar_t1.size())
		exemplar_corr1 = self.linear_e(exemplar_t1)
		A = torch.bmm(exemplar_corr1, query_flat_key.clone())
		A = F.softmax(A, dim=1)  #
		# query_flat_value_t = torch.transpose(query_flat_value,1,2).contiguous()
		B = F.softmax(torch.transpose(A, 1, 2), dim=1)
		# query_att = torch.bmm(exemplar_flat, A).contiguous() 
		exemplar_att1 = torch.bmm(query_flat_value.clone(), B).contiguous()
		input1_att = exemplar_att1.view(-1, self.value_num, fea_size[0], fea_size[1])
		# print('h_v size, h_v_org size:', torch.max(input1_att), torch.max(exemplar), torch.min(input1_att), torch.max(exemplar))
		input1_att = torch.cat([input1_att, exemplar_value1.clone()], 1)
		return input1_att
