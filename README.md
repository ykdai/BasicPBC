# Segment matching based Paint Bucket Colorization

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
In this work, we introduce a new learning-based inclusion matching pipeline, which directs the network to comprehend the inclusion relationships between segments.
To facilitate the training of our network, we also develope a unique dataset. This dataset includes rendered line arts alongside their colorized counterparts, featuring various 3D characters.

### Update
- **2024.03.29**: This repo is created.

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

Dataset can be downloaded using the following links.

Note that due to copyright issues, we do not provide download links for the Real dataset. Please contact us if you want to use the dataset for testing purposes only and not for any commercial activities.

|     | Baidu Netdisk | Google Drive | Number | Description|
| :--- | :--: | :----: | :---- | ---- |
| PaintBucket Character Train | [link](TODO) | [link](TODO) | 11,345 | 3D rendered frames for training |
| PaintBucket Character Test | [link](TODO) | [link](TODO) | 3,000 | 3D rendered frames for testing |
| PaintBucket Character Real | - | - | 200 | hand-drawn frames for real scenario testing |


### Pretrained Model

You can download the pretrained checkpoints from the following links. Please place it under the `experiments` folder and unzip it, then you can run the `basicsr/test.py` for inference. 

|                        Baidu Netdisk                         |                         Google Drive                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| [link](https://pan.baidu.com/s/1EJSYIbbQe5SZYiNIcvrmNQ?pwd=xui4) | [link](https://drive.google.com/file/d/1uFzIBNxfq-82GTBQZ_5EE9jgDh79HVLy/view?usp=sharing) |

### Model Inference
To estimate the colorized frames with our checkpoint trained on PaintBucket-Character, you can run the `basicsr/test.py` by using:
```bash
python basicsr/test.py -opt options/test/antunetclip6ch_raftdcn_pbcls_adjmerge_sep012_shuffle_option.yml
```
Find visualize results under `results/`.

To inference on your own data, put frame clip folder under `test_data/user_clips/`. The clip folder should contain gt of the 1st frame and line arts of all frames.
```
├── test_data
    ├── user_clips
        ├── put_your_clip_here
            ├── gt
                ├── 0000.png
            ├── line
                ├── 0000.png
                ├── 0001.png
                ├── ...
```
Run the `inference_line_frames.py` by using:
```bash
python inference_line_frames.py --folder_path test_data/user_clips
```
Find results under `results/user_clips/`.

### Model Training

**Training with single GPU**

To train a model with your own data/model, you can edit the `options/train/antunetclip6ch_raftdcn_pbcls_adjmerge_sep012_shuffle_option.yml` and run the following codes. You can also add `--debug` argument to start the debug mode:

```bash
python basicsr/train.py -opt options/train/antunetclip6ch_raftdcn_pbcls_adjmerge_sep012_shuffle_option.yml
```

**Training with multiple GPU**

You can run the following command for multiple GPU tranining:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/dist_train.sh 2 options/train/antunetclip6ch_raftdcn_pbcls_adjmerge_sep012_shuffle_option.yml
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
    ├── data
    │   ├── PaintBucket_Char
    ├── experiments
    ├── options
        ├── test
        ├── train
    ├── paint
    ├── raft
    ├── results
    ├── scripts
    ├── test_data
    │   ├── PaintBucket_Char
    │   ├── PaintBucket_Real
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
