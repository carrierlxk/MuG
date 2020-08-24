# OS libraries
import os
import copy
import queue
import argparse
import scipy.misc
import numpy as np
from tqdm import tqdm
from numpy.random import randint
# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# Customized libraries
from libs.test_utils import *
from libs.model import transform
from libs.vis_utils import norm_mask
from libs.model_match_residual import Model_switchGTfixdot_swCC_Res_ZVOS as Model
#from my_model import track_match_comb as Model
class VideoRecord(object):
    pass
############################## helper functions ##############################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type = int, default = 1,
                        help = "batch size")
    parser.add_argument("-o", "--out_dir",type = str,default = "results_ZVOS_residual/",
                        help = "output saving path")
    parser.add_argument("--pretrainRes", action="store_true")
    parser.add_argument("--encoder_dir", type=str, default='weights/encoder_single_gpu.pth', help="pretrained encoder")
    parser.add_argument("--decoder_dir", type=str, default='weights/decoder_single_gpu.pth', help="pretrained decoder")
    parser.add_argument("--device", type = int, default = 5,
                        help = "0~4 for single GPU, 5 for dataparallel.")
    parser.add_argument("-c","--checkpoint_dir",type = str,
                        default = "weights/checkpoint_latest.pth.tar",
                        help = "checkpoints path")
    parser.add_argument("-s", "--scale_size", type = int, nargs = "+",
                        help = "scale size, a single number for shorter edge, or a pair for height and width")
    parser.add_argument("--pre_num", type = int, default = 1,
                        help = "preceding frame numbers")
    parser.add_argument("--temp", type = float,default = 1,
                        help = "softmax temperature")
    parser.add_argument("--topk", type = int, default = 5,
                        help = "accumulate label from top k neighbors")
    parser.add_argument("-d", "--davis_dir", type = str,
                        default = "/workspace/DAVIS/",
                        help = "davis dataset path")


    args = parser.parse_args()
    args.is_train = False

    args.multiGPU = args.device == 5
    if not args.multiGPU:
        torch.cuda.set_device(args.device)

    args.val_txt = os.path.join(args.davis_dir, "val_seqs.txt")
    args.davis_dir = os.path.join(args.davis_dir, "JPEGImages/480p/")

    return args

############################## testing functions ##############################

def forward(frame1, frame1_org, frame1_sal, frame2, model, seg, segments):
    """
    propagate seg of frame1 to frame2
    """
    n, c, h, w = frame1.size()
    frame1_gray = frame1[:,0].view(n,1,h,w)
    frame1_org_gray = frame1_org[:, 0].view(n, 1, h, w)
    frame2_gray = frame2[:,0].view(n,1,h,w)
    frame1_gray = frame1_gray.repeat(1,3,1,1)
    frame2_gray = frame2_gray.repeat(1,3,1,1)
    frame1_org_gray = frame1_org_gray.repeat(1, 3, 1, 1)
    for ii in range(0,8):
        temp = segments[:,ii]
        #print('temp size:', temp.size())
        temp_gray = temp[:, 0].view(n, 1, h, w)
        temp_gray = temp_gray.repeat(1, 3, 1, 1)
        segments[:,ii] = temp_gray
    #print('input size:', frame1_gray.size(), segments.size())
    output = model(frame1_gray, frame1_org_gray, frame1_sal, frame2_gray, frame1, frame2, segments)
    #print('output size:',len(output))
    aff = output[2]
    mask = output[-1]
    frame2_seg = transform_topk(aff, seg.cuda(), k=args.topk)

    return frame2_seg, mask

def _sample_indices(record):
    """
    :param record: VideoRecord
    :return: list
    """
    average_duration = (record.num_frames - 1 + 1) // 8
    if average_duration > 0:
        offsets = np.multiply(list(range(8)), average_duration) + randint(average_duration,
                                                                                          size=8)
    elif record.num_frames > 8:
        offsets = np.sort(randint(record.num_frames - 1 + 1, size=8))
    else:
        offsets = np.zeros((8,))
    return offsets + 1

def test(model, frame_list, video_dir, first_seg, seg_ori):
    """
    test on a video given first frame & segmentation
    """
    video_dir = os.path.join(video_dir)
    video_nm = video_dir.split('/')[-1]
    video_folder = os.path.join(args.out_dir, video_nm)
    os.makedirs(video_folder, exist_ok = True)

    transforms = create_transforms()

    # The queue stores args.pre_num preceding frames
    que = queue.Queue(args.pre_num)
    record = VideoRecord()
    record.num_frames = len(frame_list)-1
    # first frame
    video_samples = _sample_indices(record)
    frame1, ori_h, ori_w, frame1_sal = my_read_frame(frame_list[0], transforms, args.scale_size)
    n, c, h, w = frame1.size()
    segments = [my_read_frame(frame_list[ii], transforms, args.scale_size) for ii in video_samples]
    segments = torch.stack(([segments[ii][0] for ii in range(0,8)]), dim=1)
    print('segments size:', len(segments), frame1.size(), segments.size())
    # saving first segmentation
    out_path = os.path.join(video_folder,"00000.png")
    imwrite_indexed(out_path, seg_ori)

    for cnt in tqdm(range(0,len(frame_list))):
        frame_tar, ori_h, ori_w, _ = my_read_frame(frame_list[cnt], transforms, args.scale_size)

        with torch.no_grad():
            # frame 1 -> frame cnt
            frame_tar_acc, mask_org = forward(frame1,frame_tar, frame1_sal, segments[:,-2], model, first_seg, segments)
            tmp_queue = list(que.queue)
            for pair in tmp_queue:
                framei = pair[0]
                segi = pair[1]
                frame_tar_est_i, mask = forward(framei, frame_tar, frame1_sal, frame_tar, model, segi,segments)
                frame_tar_acc += frame_tar_est_i
            frame_tar_avg = frame_tar_acc / (1 + len(tmp_queue))

            #output, = forward(frame_tar, frame_tar, model, seg, segments)
            # frame cnt - i -> frame cnt, (i = 1, ..., pre_num)
            predict_mask = mask_org
            predict_mask = F.interpolate(predict_mask,[ori_h, ori_w])
            predict_mask = predict_mask.squeeze().cpu().numpy()
            print('output size:', np.max(predict_mask), np.min(predict_mask))

        frame_nm = frame_list[cnt].split('/')[-1].replace(".jpg",".png")
        out_path = os.path.join(video_folder,frame_nm)
        if (que.qsize() == args.pre_num):
            que.get()
        seg = copy.deepcopy(frame_tar_avg)
        frame, ori_h, ori_w, _ = my_read_frame(frame_list[cnt], transforms, args.scale_size)
        que.put([frame, seg])

        # upsampling & argmax
        frame_tar_avg = torch.nn.functional.interpolate(frame_tar_avg, scale_factor=8, mode='bilinear')
        frame_tar_avg = frame_tar_avg.squeeze()
        frame_tar_avg = norm_mask(frame_tar_avg.squeeze())
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

        # saving to disk
        frame_tar_seg = frame_tar_seg.squeeze().cpu().numpy()
        frame_tar_seg = np.array(frame_tar_seg, dtype=np.uint8)
        frame_tar_seg = scipy.misc.imresize(frame_tar_seg, (ori_h, ori_w), "nearest")

        output_path = os.path.join(video_folder, frame_nm.split('.')[0] + '_seg.png')
        #print('output path:',output_path)
        #imwrite_indexed(output_path, frame_tar_seg)

        output_path = os.path.join(video_folder, frame_nm.split('.')[0]+'.png')
        #predict_mask = scipy.misc.imresize(predict_mask, (ori_h, ori_w), "bilinear")
        seg_ori = (predict_mask * 255).astype(np.uint8)
        # print('max value:', np.max(seg_ori))
        seg_ori = Image.fromarray(seg_ori)
        seg_ori.save(output_path, format='PNG')

############################## main function ##############################

if(__name__ == '__main__'):
    args = parse_args()
    with open(args.val_txt) as f:
        lines = f.readlines()
    f.close()

    # loading pretrained model
    model = Model(pretrainRes=False, temp = args.temp, uselayer=4)
    #model = Model(args.pretrainRes, args.encoder_dir, args.decoder_dir, temp=args.temp, Resnet=args.Resnet,
    #              color_switch=args.color_switch_flag, coord_switch=args.coord_switch_flag)
    #model = Model(pretrainRes=False,  temp=args.temp, color_switch=True, coord_switch=0)
    if(args.multiGPU):
        model = nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoint_dir)
    best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{} ({})' (epoch {})"
          .format(args.checkpoint_dir, best_loss, checkpoint['epoch']))
    model.cuda()
    model.eval()


    # start testing
    for cnt,line in enumerate(lines):
        video_nm = line.strip()
        print('[{:n}/{:n}] Begin to segmentate video {}.'.format(cnt,len(lines),video_nm))

        video_dir = os.path.join(args.davis_dir, video_nm)
        frame_list = read_frame_list(video_dir)
        seg_dir = frame_list[0].replace("JPEGImages","Annotations")
        seg_dir = seg_dir.replace("jpg","png")
        _, first_seg, seg_ori = my_read_seg(seg_dir, args.scale_size)
        test(model, frame_list, video_dir, first_seg, seg_ori)
