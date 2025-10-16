
# SRNet

This repository contains the source code for our paper:

SRNet: Self-supervised Structure Regularization for Stereo Matching

<!-- <img src="IGEV-Stereo/IGEV-Stereo.png"> -->

## 📢 News
2024-12-30: We add bfloat16 training to prevent potential NAN issues during the training process.<br>

## Demos
Pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1SsMHRyN7808jDViMN1sKz1Nx-71JxUuz?usp=share_link)

We assume the downloaded pretrained weights are located under the pretrained_models directory.

You can demo a trained model on pairs of images. To predict stereo for Middlebury, run
```Shell
python demo_imgs.py \
--restore_ckpt pretrained_models/sceneflow/sceneflow.pth \
-l=path/to/your/left_imgs \
-r=path/to/your/right_imgs
```
or you can demo a trained model pairs of images for a video, run:
```Shell
python demo_video.py \
--restore_ckpt pretrained_models/sceneflow/sceneflow.pth \
-l=path/to/your/left_imgs \
-r=path/to/your/right_imgs
```

To save the disparity values as .npy files, run any of the demos with the ```--save_numpy``` flag.

<img src="IGEV-Stereo/demo-imgs.png" width="90%">

## Comparison with RAFT-Stereo

| Method | KITTI 2012 <br> (3-noc) | KITTI 2015 <br> (D1-all) | Memory (G) | Runtime (s) |
|:-:|:-:|:-:|:-:|:-:|
| RAFT-Stereo | 1.30 % | 1.82 % | 1.02 | 0.38 |
| IGEV-Stereo | 1.12 % | 1.59 % | 0.66 | 0.18 |


## Environment
* NVIDIA RTX 3090
* Python 3.8

### Create a virtual environment and activate it.

```Shell
conda create -n IGEV python=3.8
conda activate IGEV
```
### Dependencies

```Shell
bash env.sh
```

Alternatively, you can install a higher version of PyTorch that supports bfloat16 training.

```Shell
bash env_bfloat16.sh
```

## Required Data
To evaluate/train IGEV-Stereo, you will need to download the required datasets. 
* [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (Includes FlyingThings3D, Driving & Monkaa)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)
* [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-test-data)

By default `core/stereo_datasets.py` will search for the datasets in these locations. 

```
├── /data
    ├── sceneflow
        ├── frames_finalpass
            ├── TRAIN
                ├── A
                ├── ...
                ├── 15mm_focallength
                ├── ...
                ├── funnyworld_augmented0_x2
                ├── ...
            ├── TEST
        ├── disparity
    ├── KITTI
        ├── KITTI_2012
            ├── training
            ├── testing
            ├── vkitti
        ├── KITTI_2015
            ├── training
            ├── testing
            ├── vkitti
    ├── Middlebury
        ├── trainingH
        ├── trainingH_GT
    ├── ETH3D
        ├── two_view_training
        ├── two_view_training_gt
    ├── DTU_data
        ├── dtu_train
        ├── dtu_test
```
You should replace the default path with your own.

### DTU
* Download pre-processed [DTU's training set](https://polybox.ethz.ch/index.php/s/ugDdJQIuZTk4S35) (provided by PatchmatchNet). The dataset is already organized as follows:
```
root_directory
├──Cameras_1
├──Rectified
└──Depths_raw
```
* Download our processed camera parameters from [here](https://drive.google.com/file/d/1DAAFXV6bZx0NNWFQMwoSeWMt5mr64myD/view?usp=sharing). Unzip all the camera folders into `root_directory/Cameras_1`.

## Evaluation

To evaluate on Scene Flow or Middlebury or ETH3D, run

```Shell
python evaluate_stereo.py --restore_ckpt ./pretrained_models/sceneflow/sceneflow.pth --dataset sceneflow
```
or
```Shell
python evaluate_stereo.py --restore_ckpt ./pretrained_models/sceneflow/sceneflow.pth --dataset middlebury_H
```
or
```Shell
python evaluate_stereo.py --restore_ckpt ./pretrained_models/sceneflow/sceneflow.pth --dataset eth3d
```

## Training

To train on Scene Flow, run

```Shell
python train_stereo.py --logdir ./checkpoints/sceneflow
```

To train on KITTI, run
```Shell
python train_stereo.py --logdir ./checkpoints/kitti --restore_ckpt ./pretrained_models/sceneflow/sceneflow.pth --train_datasets kitti
```

## Bfloat16 Training
NaN values during training: If you encounter NaN values in your training, this is likely due to overflow when using float16. This can happen when large gradients or high activation values exceed the range represented by float16. To fix this: 

-Try switching to `bfloat16` by using `--precision_dtype bfloat16`.

-Alternatively, you can use `float32` precision by setting `--precision_dtype float32`.

### Training with bfloat16
1. Before you start training, make sure you have hardware that supports bfloat16 and the right environment set up for mixed precision training.
```Shell
bash env_bfloat16.sh
```

2. Then you can train the model with bfloat16 precision:
```Shell
python train_stereo.py --mixed_precision --precision_dtype bfloat16
```

## Submission

For submission to the KITTI benchmark, run
```Shell
python save_disp.py
```

## MVS training and evaluation

To train on DTU, run

```Shell
python train_mvs.py
```

To evaluate on DTU, run

```Shell
python evaluate_mvs.py
```

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex


```


# Acknowledgements

This project is based on [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), and [CoEx](https://github.com/antabangun/coex). We thank the original authors for their excellent works.

