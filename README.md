# MuG
code for CVPR 2020 paper:
Learning Video Object Segmentation from Unlabeled Videos
##

![](../master/overview.png)

## Pre-compute results

The segmentation results of object--level zero-shot VOS (DAVIS16-val dataset), instance-level zero-shot VOS (DAVIS2017-test-dev dataset) and  one-shot VOS (DAVIS2016-val and DAVIS 2017-val datasets) under both unsupervised and weakly supervised conditionscan be download from [GoogleDrive](https://drive.google.com/file/d/1Gn3XmqPhaw7Z2CMEoyTy_w1n7xjdjbMh/view?usp=sharing).

## Code runing

1. Setup environment: Pytorch 1.1.0, tqdm, scipy 1.2.1.
2. Prepare training data. Download training datasets from Got10k tracking [dataset](http://got-10k.aitestunion.com/) or Youtube-VOS [dataset](https://youtube-vos.org/challenge/2019/). Generate a csv file in a format of 'GOT-10k_Train_000001,	120'. The first term is video name, the second term is video length.
3. Download the weakly supervised saliency generation model and inference code from [here](https://github.com/zengxianyu/mws) and unsupervised saliency detection from [here] (https://github.com/ruanxiang/mr_saliency)
4. Change all the paths in MuG_GOT_global_new_residual.py, my_model_new_residual.py and libs/model_match_residual.py.
Run run_train_all_GOT_global_new_residual.sh for network training.
5. Run  run_ZVOS.sh for network inference.


### Other related projects/papers:
[Zero-shot Video Object Segmentation via Attentive Graph Neural Networks](https://github.com/carrierlxk/AGNN)

[See More, Know More: Unsupervised Video Object Segmentation with Co-Attention Siamese Networks(CVPR19)](https://github.com/carrierlxk/COSNet)

[Saliency-Aware Geodesic Video Object Segmentation (CVPR15)](https://github.com/wenguanwang/saliencysegment)

[Learning Unsupervised Video Primary Object Segmentation through Visual Attention (CVPR19)](https://github.com/wenguanwang/AGS)

[Joint-task Self-supervised Learning for Temporal Correspondence](https://github.com/Liusifei/UVC)

Any comments, please email: carrierlxk@gmail.com
