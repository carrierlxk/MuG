# a combination of track and match
# 1. load fullres images, resize to 640**2
# 2. warmup: set random location for crop
# 3. loc-match: add attention
import os
import cv2
import sys
import time
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from libs.GOT_all_global_new_loader import VidListv1, VidListv2
import torch.backends.cudnn as cudnn
import libs.transforms_multi as transforms
from my_model_new_residual import track_match_comb as Model
from libs.loss import L1_loss, my_BCE_loss, my_L1_loss
from libs.concentration_loss import ConcentrationSwitchLoss as ConcentrationLoss
from libs.train_utils import save_vis, AverageMeter, save_checkpoint, log_current
from libs.utils import diff_crop
#from libs.deeplab_org import DeepLab_org
import torch.nn.functional as F

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def parse_args():
	parser = argparse.ArgumentParser(description='')

	# file/folder pathes
	parser.add_argument("--videoRoot", type=str, default="/raid/tracking_dataset/GOT-10k/train/", help='train video path')
	parser.add_argument("--videoList", type=str, default="/Data2/Kinetices/compress/train.txt", help='train video list (after "train_256")')
	parser.add_argument("--encoder_dir",type=str, default='weights/encoder_single_gpu.pth', help="pretrained encoder")
	parser.add_argument("--decoder_dir",type=str, default='weights/decoder_single_gpu.pth', help="pretrained decoder")
	parser.add_argument('--resume', type=str, default='', metavar='PATH', help='path to latest checkpoint (default: none)')
	parser.add_argument("-c","--savedir",type=str,default="match_track_comb/",help='checkpoints path')
	parser.add_argument("--Resnet", type=str, default="r18", help="choose from r18 or r50")

	# main parameters
	parser.add_argument("--pretrainRes",action="store_true")
	parser.add_argument("--batchsize",type=int, default=1, help="batchsize")
	parser.add_argument('--workers', type=int, default=16)
	parser.add_argument("--patch_size", type=int, default=256, help="crop size for localization.")
	parser.add_argument("--full_size", type=int, default=640, help="full size for one frame.")
	parser.add_argument("--rotate",type=int,default=10,help='degree to rotate training images')
	parser.add_argument("--scale",type=float,default=1.2,help='random scale')
	parser.add_argument("--lr",type=float,default=0.0001,help='learning rate')
	parser.add_argument('--lr-mode', type=str, default='poly')
	parser.add_argument("--window_len",type=int,default=2,help='number of images (2 for pair and 3 for triple)')
	parser.add_argument("--log_interval",type=int,default=10,help='')
	parser.add_argument("--save_interval",type=int,default=100,help='save every x epoch')
	parser.add_argument("--momentum",type=float,default=0.9,help='momentum')
	parser.add_argument("--weight_decay",type=float,default=0.005,help='weight decay')
	parser.add_argument("--device", type=int, default=4, help="0~device_count-1 for single GPU, device_count for dataparallel.")
	parser.add_argument("--temp", type=int, default=1, help="temprature for softmax.")

	# set epoches
	parser.add_argument("--wepoch",type=int,default=10,help='warmup epoch')
	parser.add_argument("--nepoch",type=int,default=20,help='max epoch')

	# concenration regularization
	parser.add_argument("--lc",type=float,default=1e4, help='weight of concentration loss')
	parser.add_argument("--lc_win",type=int,default=8, help='win_len for concentration loss')

	# orthorganal regularization
	parser.add_argument("--color_switch",type=float,default=0.1, help='weight of color switch loss')
	parser.add_argument("--coord_switch",type=float,default=0.1, help='weight of color switch loss')

	print("Begin parser arguments.")
	args = parser.parse_args()
	assert args.videoRoot is not None
	assert args.videoList is not None
	if not os.path.exists(args.savedir):
		os.mkdir(args.savedir)
	args.savepatch = os.path.join(args.savedir,'savepatch')
	args.logfile = open(os.path.join(args.savedir,"logargs.txt"),"w") 
	args.multiGPU = args.device == torch.cuda.device_count()

	if not args.multiGPU:
		torch.cuda.set_device(args.device)
	if not os.path.exists(args.savepatch):
		os.mkdir(args.savepatch)

	args.vis = True
	if args.color_switch > 0:
		args.color_switch_flag = True
	else:
		args.color_switch_flag = False
	if args.coord_switch > 0:
		args.coord_switch_flag = True
	else:
		args.coord_switch_flag = False

	try:
		from tensorboardX import SummaryWriter
		global writer
		writer = SummaryWriter()
	except ImportError:
		args.vis = False
	print(' '.join(sys.argv))
	print('\n')
	args.logfile.write(' '.join(sys.argv))
	args.logfile.write('\n')
	
	for k, v in args.__dict__.items():
		#print(k, ':', v)
		args.logfile.write('{}:{}\n'.format(k,v))
	args.logfile.close()
	return args
	

def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []
    #print('prams:', type(model.parameters()))
    if torch.cuda.device_count() == 1:
        b.append(model.encoder.layer5)
    else:
        #b.append(model.module.encoder.feature)
        b.append(model.module.gray_encoder)
        #b.append(model.module.encoder.feature.)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b_module=[]
    if torch.cuda.device_count() == 1:
        b.append(model.linear_e.parameters())
        b.append(model.main_classifier.parameters())
    else:
        b_module.append(model.module.weak_conv1)
        b_module.append(model.module.memory_encoder)
        b_module.append(model.module.fusion3_1)
        b_module.append(model.module.fusion3_2)
        b_module.append(model.module.fusion2_1)
        b_module.append(model.module.fusion2_2)
        b.append(model.module.weak_bn.parameters())
        b.append(model.module.weak_conv2_1.parameters())
        b.append(model.module.weak_conv2_2.parameters())
        b.append(model.module.weak_conv2_3.parameters())
        b.append(model.module.linear_e.parameters())
        #b.append(model.module.corr_layer.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i
    for i in range(len(b_module)):
        for j in b_module[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def adjust_learning_rate(args, optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	if args.lr_mode == 'step':
		lr = args.lr * (0.1 ** (epoch/100 // args.step))
	elif args.lr_mode == 'poly':
		lr = args.lr * (1 - epoch/100 / args.nepoch) ** 0.9
	else:
		raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr
	

def create_loader(args):
	print('ok!')
	dataset_train_warm = VidListv1(args.videoRoot, args.videoList, args.patch_size, args.rotate, args.scale)
	dataset_train = VidListv2(args.videoRoot, args.videoList, args.patch_size, args.window_len, args.rotate, args.scale, args.full_size)

	if args.multiGPU:
		train_loader_warm = torch.utils.data.DataLoader(
			dataset_train_warm, batch_size=args.batchsize, shuffle = True, num_workers=args.workers, pin_memory=True, drop_last=True)
		train_loader = torch.utils.data.DataLoader(
			dataset_train, batch_size=args.batchsize, shuffle = True, num_workers=args.workers, pin_memory=True, drop_last=True)
	else:
		train_loader_warm = torch.utils.data.DataLoader(
			dataset_train_warm, batch_size=args.batchsize, shuffle = True, num_workers=0, drop_last=True)
		train_loader = torch.utils.data.DataLoader(
			dataset_train, batch_size=args.batchsize, shuffle = True, num_workers=0, drop_last=True)
	return train_loader_warm, train_loader

my_creteria = nn.BCEWithLogitsLoss()

def train(args):

	loader_warm, loader = create_loader(args)
	print('length:', len(loader))
	cudnn.benchmark = True
	best_loss = 1e10
	start_epoch = 0
	#			  color_switch=args.color_switch_flag, coord_switch=args.coord_switch_flag)
	model = Model(args.pretrainRes, args.encoder_dir, args.decoder_dir, temp = args.temp, Resnet = args.Resnet, color_switch = args.color_switch_flag, coord_switch = args.coord_switch_flag)
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			start_epoch = checkpoint['epoch']
			best_loss = checkpoint['best_loss']
			new_params = model.state_dict().copy()
			for i in checkpoint['state_dict']:
				# Scale.layer5.conv2d_list.3.weight
				i_parts = i.split('.')  # 针对多GPU的情况
				#print('i_parts:  ', '.'.join(i_parts))
				# if  not i_parts[1]=='main_classifier': #and not '.'.join(i_parts[1:-1]) == 'layer5.bottleneck' and not '.'.join(i_parts[1:-1]) == 'layer5.bn':  #init model pretrained on COCO, class name=21, layer5 is ASPP
				new_params['.'.join(i_parts[1:])] = checkpoint['state_dict'][i]
			model.load_state_dict(new_params)
			#model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{} ({})' (epoch {})".format(args.resume, best_loss, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	if args.multiGPU:
		model = torch.nn.DataParallel(model).cuda()

		closs = ConcentrationLoss(win_len=args.lc_win, stride=args.lc_win,
								   F_size=torch.Size((args.batchsize//torch.cuda.device_count(),2, args.patch_size//8, args.patch_size//8)), temp = args.temp)
		closs = nn.DataParallel(closs).cuda()
		optimizer = torch.optim.Adam([{'params': get_1x_lr_params(model), 'lr': 1.0*args.lr },  #针对特定层进行学习，有些层不学习
                {'params': get_10x_lr_params(model), 'lr': 10*args.lr}], args.lr, weight_decay=args.weight_decay) # momentum=args.momentum,
			#torch.optim.Adam(filter(lambda p: p.requires_grad, model._modules['module'].parameters()),args.lr)
	else:
		closs = ConcentrationLoss(win_len=args.lc_win, stride=args.lc_win,
								   F_size=torch.Size((args.batchsize,2,
													  args.patch_size//8,
													  args.patch_size//8)), temp = args.temp)
		model.cuda()
		closs.cuda()
		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),args.lr)

	for epoch in range(start_epoch, args.nepoch):
		if epoch < args.wepoch:
			lr = adjust_learning_rate(args, optimizer, epoch)
			print("Base lr for epoch {}: {}.".format(epoch, optimizer.param_groups[0]['lr']))
			best_loss = train_iter(args, loader_warm, model, closs, optimizer, epoch, best_loss)
		else:
			lr = adjust_learning_rate(args, optimizer, epoch-args.wepoch)
			print("Base lr for epoch {}: {}.".format(epoch, optimizer.param_groups[0]['lr']))
			best_loss = train_iter(args, loader, model, closs, optimizer, epoch, best_loss)


def forward(frame1, frame1_org, frame1_sal, frame2, model, warm_up, patch_size=None, segments=None):
	n, c, h, w = frame1.size()
	if warm_up:
		output = model(frame1, frame2)
	else:
		#print('second stage size:', frame1.size(), frame2.size())
		output = model(frame1, frame1_org, frame1_sal, frame2, warm_up=False, patch_size=[patch_size//8, patch_size//8],segments=segments)
		new_c = output[2] #new location
		# gt patch
		# print("HERE2: ", frame2.size(), new_c, patch_size)
		color2_gt = diff_crop(frame2, new_c[:,0], new_c[:,2], new_c[:,1], new_c[:,3], patch_size, patch_size)
		output.append(color2_gt)
	return output


def train_iter(args, loader, model, closs, optimizer, epoch, best_loss):
	losses = AverageMeter()
	batch_time = AverageMeter()
	losses = AverageMeter()
	c_losses = AverageMeter()
	model.train()
	end = time.time()
	if args.coord_switch_flag:
		coord_switch_loss = nn.L1Loss()
		sc_losses = AverageMeter()

	if epoch < 1 or (epoch>=args.wepoch and epoch< args.wepoch+2):
		thr = None
	else:
		thr = 2.5
	train_len = len(loader)
	for i,item in enumerate(loader):
		#print('iteration:', i, len(item))
		frames = item[0]
		frames_pair = item[1]
		segments = item[2]
		segments = torch.stack(segments, dim=1)
		#print('segment:',segments.size())
		org_pair = item[3]
		frame1_var = frames[0].cuda()
		frame2_var = frames[1].cuda()
		frame1_org = frames_pair[0].cuda()
		frame1_sal = org_pair[0].cuda()
		if epoch < args.wepoch:
			output = forward(frame1_var, frame2_var, model, warm_up=True)
			color2_est = output[0]
			aff = output[1]
			b,x,_ = aff.size()
			color1_est = None
			if args.color_switch_flag:
				color1_est = output[2]
			loss_ = L1_loss(color2_est, frame2_var, 10, 10, thr=thr, pred1=color1_est, frame1_var = frame1_var)

			if epoch >=1 and args.lc > 0:
				constraint_loss = torch.sum(closs(aff.view(b,1,x,x))) * args.lc
				c_losses.update(constraint_loss.item(), frame1_var.size(0))
				loss = loss_ + constraint_loss
			else:
				loss = loss_
			#if(i % args.log_interval == 0):
			#	save_vis(color2_est, frame2_var, frame1_var, frame2_var, args.savepatch)
		else:
			# print("input: ", frame1_var.size(), frame2_var.size())
			output = forward(frame1_var, frame1_org, frame1_sal, frame2_var, model, warm_up=False, patch_size = args.patch_size, segments = segments)
			color2_est = output[0]
			aff = output[1]
			new_c = output[2]
			coords = output[3]
			pred_1 = output[4]
			gt_mask1 = output[5]
			Fcolor2_crop = output[-1]

			b,x,x = aff.size()
			color1_est = None
			count = 5

			constraint_loss = torch.sum(closs(aff.view(b,1,x,x))) * args.lc
			c_losses.update(constraint_loss.item(), frame1_var.size(0))

			if args.color_switch_flag:
				count += 1
				color1_est = output[count]
			pred_1 = F.upsample(pred_1, gt_mask1.size()[2:], mode='bilinear')
			my_l1_loss = my_L1_loss(pred_1, gt_mask1)
			print('output range:', torch.max(pred_1), torch.min(pred_1), torch.max(gt_mask1), torch.min(gt_mask1))
			gt_mask1 = F.sigmoid(gt_mask1).detach()
			gt_mask1[gt_mask1 > 0.2] = 1
			gt_mask1[gt_mask1 < 0.2] = 0
			frame_loss = my_l1_loss+my_creteria(pred_1, gt_mask1)#* (1 - 0.05)#my_BCE_loss(pred_1, gt_mask1) #
			#gt_self = F.sigmoid(pred_1).detach()
			#gt_self[gt_self > 0.2] = 1
			#gt_self[gt_self <= 0.2] = 0
			#frame_loss = frame_loss + my_creteria(pred_1, gt_self) * 0.05
			#loss_color = L1_loss(color2_est, Fcolor2_crop, 10, 10, thr=thr, pred1=color1_est, frame1_var = frame1_var)
			#print('loss:', frame_loss, loss_color, constraint_loss)
			loss_ =  frame_loss #0.01*loss_color + 0.01*constraint_loss
			
			if args.coord_switch_flag:
				count += 1
				grids = output[count]
				C11 = output[count+1]
				loss_coord = args.coord_switch * coord_switch_loss(C11, grids)
				loss = loss_ + loss_coord
				sc_losses.update(loss_coord.item(), frame1_var.size(0))				
			else:
				loss = loss_
				
			#if(i % args.log_interval == 0):
			#	save_vis(color2_est, Fcolor2_crop, frame1_var, frame2_var, args.savepatch, new_c)

		losses.update(loss.item(), frame1_var.size(0))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		batch_time.update(time.time() - end)
		end = time.time()			

		if((i + epoch*train_len) % args.save_interval == 0):
			is_best = losses.avg < best_loss
			best_loss = min(losses.avg, best_loss)
			checkpoint_path = os.path.join(args.savedir, str(epoch+1)+'checkpoint_latest.pth.tar')
			save_checkpoint({
					'epoch': epoch + 1,
					'state_dict': model.state_dict(),
					'best_loss': best_loss,
				}, is_best, filename=checkpoint_path, savedir = args.savedir)
			log_current(epoch, losses.avg, best_loss, filename = "log_current.txt", savedir=args.savedir)

	return best_loss


if __name__ == '__main__':
	args = parse_args()
	train(args)
	writer.close()
