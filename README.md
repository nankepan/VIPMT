# VIPMT
This is the implementation of our paper: Multi-grained Temporal Prototype Learning for Few-shot Video Object Segmentation that has been accepted to  IEEE International Conference on Computer Vision (ICCV) 2023.

## Abstract
Few-Shot Video Object Segmentation (FSVOS) aims to segment objects in a query video with the same category defined by a few annotated support images. However, this task was seldom explored. In this work, based on IPMT, a state-of-the-art few-shot image segmentation method that combines external support guidance information with adaptive query guidance cues, we propose to leverage multi-grained temporal guidance information for handling the temporal correlation nature of video data. We decompose the query video information into a clip prototype and a memory prototype for capturing local and long-term internal temporal guidance, respectively. Frame prototypes are further used for each frame independently to handle fine-grained adaptive guidance and enable bidirectional clip-frame prototype communication. To reduce the influence of noisy memory, we propose to leverage the structural similarity relation among different predicted regions and the support for selecting reliable memory frames. Furthermore, a new segmentation loss is also proposed to enhance the category discriminability of the learned prototypes. Experimental results demonstrate that our proposed video IPMT model significantly outperforms previous FSVOS models on two benchmark datasets.

# Environment

```
conda create -n VIPMT python=3.6
conda activate VIPMT
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
conda install opencv cython
pip install easydict imgaug
```


# Usage
## Preparation

1. Download the 2019 version of [Youtube-VIS](https://youtube-vos.org/dataset/vis/) dataset.
1. Download [VSPW 480P](https://github.com/sssdddwww2/vspw_dataset_download) dataset.
2. Put the dataset in the `./data` folder.
```
data
└─ Youtube-VOS
    └─ train
        └─ Annotations
        └─ JPEGImages
        └─ train.json
└─ VSPW_480p
    └─ data
```
3. Install [cocoapi](https://github.com/youtubevos/cocoapi) for Youtube-VIS.
4. Download the ImageNet pretrained [backbone](https://drive.google.com/file/d/1PIMA7uG_fcvXUvjDUL7UIVp6KmGdSFKi/view?usp=sharing) and put it into the `pretrain_model` folder.
```
pretrain_model
└─ resnet50_v2.pth
```
5. Update `config/config.py`.

## Training

```
python train.py --group 1 --batch_size 4
```

## Inference

```
python test.py --group 1
```

# References
Part of the code is based upon:
[IPMT](https://github.com/LIUYUANWEI98/IPMT)
[DANet](https://github.com/scutpaul/DANet)
