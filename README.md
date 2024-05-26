# Learning Inclusion Matching for Animation Paint Bucket Colorization

[Project Page](https://ykdai.github.io/projects/InclusionMatching) | [Video](https://www.youtube.com/watch?v=nNnPUItGvSo)

<img src="assets/teaser.png" width="800px"/>

This repository provides the official implementation for the following paper:

<p>
<div><strong>Learning Inclusion Matching for Animation Paint Bucket Colorization</strong></div>
<div><a href="https://ykdai.github.io/">Yuekun Dai</a>, 
     <a href="https://shangchenzhou.com/">Shangchen Zhou</a>,
     <a href="https://github.com/dienachtderwelt">Qinyue Li</a>, 
     <a href="https://li-chongyi.github.io/">Chongyi Li</a>,
     <a href="https://www.mmlab-ntu.com/person/ccloy/">Chen Change Loy</a></div>
<div>Accepted to <strong>CVPR 2024</strong></div><div><a href=https://arxiv.org/abs/2403.18342> arXiv </a>
</p>

### BasicPBC
Colorizing line art is a pivotal task in the production of hand-drawn cel animation. 
In this work, we introduce a new learning-based inclusion matching pipeline, which directs the network to comprehend the inclusion relationships between segments. To facilitate the training of our network, we also propose a unique dataset **PaintBucket-Character**. This dataset includes rendered line arts alongside their colorized counterparts, featuring various 3D characters.

### Update
- **2024.05.26**: Add Colab demo of BasicPBC. <a href="https://colab.research.google.com/drive/1kJKLdFXY0O6jRxQCYS4xjaEcOTu7sLPu?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>
- **2024.04.25**: Light-weighted model released.
- **2024.04.12**: Support multiple ground-truth inference.
- **2024.04.08**: Model inference updated. Support all resolutions and unclosed line art images.
- **2024.03.30**: Checkpoint and training code of our BasicPBC are released.
- **2024.03.29**: This repo is created.

### TODO
- [X] Add google colab inference script.
- [X] Add trapped-ball segmentation module for unclosed line arts inference.
- [X] Add a lightweight BasicPBC which can process images in 2K resolution without encountering Out-Of-Memory (OOM) error on 16GB RAM GPU.

### Installation

1. Clone the repo

    ```bash
    git clone https://github.com/ykdai/BasicPBC.git
    ```

1. Install dependent packages

    ```bash
    cd BasicPBC
    pip install -r requirements.txt
    ```

1. Install BasicPBC  
    Please run the following commands in the **BasicPBC root path** to install BasicPBC:

    ```bash
    python setup.py develop
    ```

### Data Download

The details of our dataset can be found at [this page](https://github.com/ykdai/BasicPBC/tree/main/dataset). Dataset can be downloaded using the following links. 

|     | Google Drive | Baidu Netdisk | Number | Description|
| :--- | :--: | :----: | :---- | ---- |
| PaintBucket-Character Train/Test | [link](https://drive.google.com/file/d/1gIJVKyeIu4PQshZnHG6TWj5kSZjnMb2_/view?usp=sharing) | [link](https://pan.baidu.com/s/12AMfqwlPF-7R30RWRdUBfg?pwd=cvpr) | 11,345/3,000 | 3D rendered frames for training and testing. Our dataset is a mere 2GB in size, so feel free to download it and enjoy exploring. 😆😆 |
| PaintBucket-Real Test            |                              /                               |                              /                               | 200          | Hand-drawn frames for testing.                               |

Due to copyright issues, we do not provide download links for the real hand-drawn dataset. Please contact us through the e-mail if you want to use it or wish to get project files of our dataset. These hand-drawn frames are only for evaluation and not for any commercial activities. 

### Pretrained Model

You can download the pretrained checkpoints from the following links. Please place it under the `ckpt` folder and unzip it, then you can run the `basicsr/test.py` for inference. 

|     |  Google Drive   |  Baidu Netdisk  |
| :--- | :-------------: | :-------------: |
| BasicPBC | [link](https://drive.google.com/file/d/1QOUaulX9-mZo5wyKpWw_9QUKkRwBwMU2/view?usp=sharing)  | [link](https://pan.baidu.com/s/1CCbtoQmzYs8Q2rvSHL9-5w?pwd=cvpr) |
| BasicPBC-Light | [link](https://drive.google.com/file/d/1Oq_4Kk8AbjmIKOwufYNmRpu9WdasTxiR/view?usp=drive_link)  | [link](https://pan.baidu.com/s/1EwiaORp7I7lWRud1iVlM-w?pwd=cvpr ) |


### Model Inference
To estimate the colorized frames with our checkpoint trained on PaintBucket-Character, you can run the `basicsr/test.py` by using:
```bash
python basicsr/test.py -opt options/test/basicpbc_pbch_test_option.yml
```
Or you can test the lightweight model by:
```bash
python basicsr/test.py -opt options/test/basicpbc_light_test_option.yml
```
The colorized results will be saved at `results/`.

To play with your own data, put your anime clip(s) under `dataset/test/`. The clip folder should contain at least one colorized `gt` frame and `line` of all frames.  
We also provide two simple examples: `laughing_girl` and `smoke_explosion`.
To play with your own data, put your anime clip(s) under `dataset/test/`. The clip folder should contain at least one colorized `gt` frame and `line` of all frames.  
We also provide two simple examples: `laughing_girl` and `smoke_explosion`.
```
├── dataset 
    ├── test
        ├── laughing_girl
            ├── gt
                ├── 0000.png
            ├── line
                ├── 0000.png
                ├── 0001.png
                ├── ...
        ├── smoke_explosion
            ├── gt
            ├── line
```
To inference on `laughing_girl`, run `inference_line_frames.py` by using:
```bash
python inference_line_frames.py --path dataset/test/laughing_girl
```
Or run this to try with `smoke_explosion`:
```bash
python inference_line_frames.py --path dataset/test/smoke_explosion/  --mode nearest
```
Find results under `results/`.

`inference_line_frames.py` provides several arguments for different use cases.  

- `--mode` can be either `forward` or `nearest`. By default, `forward` processes your frames sequentially. If set `nearest`, frames will be predicted from the nearest ***gt***. e.g. Given ***gt*** *0000.png* and *0005.png*, ***line*** *0003.png* will be colored according to *0004.png* and *0004.png* is colored according to *0005.png*.  
    ```bash
    python inference_line_frames.py --path dataset/test/smoke_explosion/  --mode nearest
    ```
- `--seg_type` is `default` if not specified. It's fast and simple, but not work if your ***line*** contains unclosed region. `trappedball` is robust to this case(acknowledge <a href="https://github.com/hepesu/LineFiller">@hepesu/LineFiller</a>). To decide which one to use, you can first set `default` together with `--save_color_seg`. It will produce colorized segmentation results. If you find out that some segments are not seperated properly, switch to `trappedball`:
    ```bash
    python inference_line_frames.py --path dataset/test/smoke_explosion/  --seg_type trappedball
    ```
- `--skip_seg` can help your skip the segmentation part. You can use it when `seg` already exists.
- `--keep_line` will generate another folder named `[your_clip]_keepline` which merges the original line in the `line` folder with the colorized output. This config is mainly for the line drawn not by the binary pen.
- `--raft_res` can change the resolution for the optical flow estimation (default is 320). We notice that sometimes the performance is bad due to he wrong optical flow estimation. Thus, if the performance is not satisfied on your case, you can change this to 640 to have a try by using `--raft_res 640`.
- `--use_light_model` will use the light-weighted model for inference. Add this if working on low memory GPU. Notice that this argument may produce poorer results than the base model.
- `--multi_clip` is used if you would like to inference on many clips at the same time. Put all clips within a single folder under `dataset/test/`, e.g.:
    ```
    ├── dataset 
        ├── test
            ├── your_clip_folder
                ├── clip01
                ├── clip02
                ├── ...
    ```
    In this case, run:
    ```bash
    python inference_line_frames.py --path dataset/test/your_clip_folder/  --multi_clip
    ```

### Model Training

**Training with single GPU**

To train a model with your own data/model, you can edit the `options/train/basicpbc_pbch_train_option.yml` and run the following command. 
To train a model with your own data/model, you can edit the `options/train/basicpbc_pbch_train_option.yml` and run the following command. 

```bash
python basicsr/train.py -opt options/train/basicpbc_pbch_train_option.yml
```

**Training with multiple GPU**

You can run the following command for multiple GPU training:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/dist_train.sh 2 options/train/basicpbc_pbch_train_option.yml
```

### BasicPBC structure

```
├── BasicPBC
    ├── assets
    ├── basicsr
        ├── archs
        ├── data
        ├── losses
        ├── metrics
        ├── models
        ├── ops
        ├── utils
    ├── dataset
    	├── train
	    	├── PaintBucket_Char
        ├── test
        	├── PaintBucket_Char
        	├── PaintBucket_Real
    ├── experiments
    ├── options
        ├── test
        ├── train
    ├── paint
    ├── raft
    ├── results
    ├── scripts
```

### License

This project is licensed under <a rel="license" href="https://github.com/ykdai/BasicPBC/blob/main/LICENSE">S-Lab License 1.0</a>. Redistribution and use of the dataset and code for non-commercial purposes should follow this license.

### Citation

If you find this work useful, please cite:

```
@article{InclusionMatching2024,
  title     = {Learning Inclusion Matching for Animation Paint Bucket Colorization},
  author    = {Dai, Yuekun and Zhou, Shangchen and Li, Qinyue and Li, Chongyi and Loy, Chen Change},
  journal   = {CVPR},
  year      = {2024},
}
```

### Contact
If you have any question, please feel free to reach me out at `ydai005@e.ntu.edu.sg`.
