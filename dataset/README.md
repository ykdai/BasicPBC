# PaintBucket Character Dataset

<img src="assets/fig3.jpg" width="400px"/>

We developed a unique dataset, referred to as
PaintBucket-Character. This dataset includes rendered line
arts alongside their colorized counterparts, featuring various 3D characters including both Japanese and Western cartoon styles.

### Data Download

Dataset can be downloaded using the following links.

Note that due to copyright issues, we do not provide download links for the Real dataset. Please contact us if you want to use the dataset for testing purposes only and not for any commercial activities.

|     | Baidu Netdisk | Google Drive | Number | Description|
| :--- | :--: | :----: | :---- | ---- |
| PaintBucket Character Train | [link](TODO) | [link](TODO) | 11,345 | 3D rendered frames for training |
| PaintBucket Character Test | [link](TODO) | [link](TODO) | 3,000 | 3D rendered frames for testing |
| PaintBucket Character Real | - | - | 200 | hand drawn frames for real scenario testing |

It is recommended to symlink the dataset root to `BasicPBC/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
BasicPBC
├── assets
├── basicsr
├── data
│   ├── PaintBucket_Char
├── experiments
├── options
├── paint
├── raft
├── results
├── scripts
├── test_data
│   ├── PaintBucket_Char
```

### PaintBucket Character Train

This dataset comprises 11,345 3D rendered frames for training. It includes 12 characters each in a seperate folder. Within a character's folder, there are 6 sub folders: **gt**, **json_color**, **json_index**, **label**, **line** and **seg**.

```
BasicPBC
├── data
│   ├── PaintBucket_Char
│   │   ├── Abe
│   │   │   ├── gt
│   │   │   ├── json_color
│   │   │   ├── json_index
│   │   │   ├── label
│   │   │   ├── line
│   │   │   ├── seg
│   │   ├── Aj
│   │   ├── BigVegas
│   │   ├── Clarie
│   │   ├── Doozy
│   │   ├── Jackie
│   │   ├── Kaya
│   │   ├── Kita
│   │   ├── Ortiz
│   │   ├── Remy
│   │   ├── Ryo
│   │   ├── TheBoss
```

- **gt**: ground truth colorized frames. Use *paint.utils.read_img_2_np* to read.

<img src="assets/gt0242.png" width="400px"/>

- **json_color**: for each segment (line-enclosed region), gives ground truth color in RGBA values. Use *paint.utils.load_json* to read. 
```json
{
    "1": [0, 0, 0, 0],
    "2": [243, 229, 218, 255],
    "3": [210, 93, 83, 255],
    ... ...
}
```

- **json_index**: for each segment, gives number of pixels and corresponding **label**'s index.
<br> e.g. "2": [16, 27] means for segment 2 its corresponding **label** is 27 and it has 16 pixels. (segment 1 is always the background and it is not included in the **label**.) <br>

```json
{
    "1": [2829395, -1],
    "2": [152325, 61],
    "3": [428, 25],
    ... ...
}
```

- **label**: TODO

<img src="assets/label0242.png" width="400px"/>

- **line**: line-art frames. They are binarized 3-channel images. Use *paint.utils.read_img_2_np* to read.

<img src="assets/line0242.png" width="400px"/>

- **seg**: give each segment an index using RGB value. e.g. segment 42 has color (0,0,42). Use *paint.utils.read_seg_2_np* to read as a 2D numpy array.

<img src="assets/seg0242.png" width="400px"/>

### PaintBucket Character Test

This dataset contains 3,000 3D rendered frames of 10 characters for testing. Folder structure is almost the same as training set except that:

- **seg**: combines contents of **seg** and **json_color** in training set. png files give each segment an index. json file contains ground truth color for each segment.

```
BasicPBC
├── test_data
│   ├── PaintBucket_Char
│   │   ├── amy
│   │   │   ├── gt
│   │   │   ├── line
│   │   │   ├── seg
│   │   │   |   ├── 0000.json
│   │   │   |   ├── 0000.png
│   │   │   |   ├── ... ...
│   │   ├── Bocchi
│   │   ├── Ichika
│   │   ├── michelle
│   │   ├── mousey
│   │   ├── Mremireh_O_Desbiens
│   │   ├── Racer
│   │   ├── Sporty_Granny
│   │   ├── timmy
│   │   ├── Ty
```

### PaintBucket Character Real

This dataset has 200 hand-drawn frames from 20 short clips.

```
BasicPBC
├── test_data
│   ├── PaintBucket_Real
│   │   ├── dog
│   │   │   ├── gt
│   │   │   ├── line
│   │   │   ├── seg
│   │   ├── hairflycoolboy
│   │   ├── hairflyinggirl
│   │   ├── idol
│   │   ├── jumpinggirl
│   │   ├── kyogirl
│   │   ├── laughing_girl
│   │   ├── open_eye
│   │   ├── robothand
│   │   ├── runninggirl
│   │   ├── selfprotectboy
│   │   ├── shockinggirl
│   │   ├── sittinggirl
│   │   ├── smiling_girl
│   │   ├── standupboy
│   │   ├── strongman
│   │   ├── turning_around_boy
│   │   ├── turning_around_girl
│   │   ├── typing
│   │   ├── witcher
```