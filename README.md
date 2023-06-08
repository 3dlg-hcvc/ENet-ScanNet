# ENet on ScanNet
This repo contains helper tools for extracting [ENet](https://vitalab.github.io/article/2019/05/06/ENet.html) features for [ScanNet](http://www.scan-net.org/) video frames, which can be used to generate projected multi-view per-point features for 3D spoint clouds as model input (e.g. [ScanRefer](https://github.com/daveredrum/ScanRefer), [3DVG-Transformer](https://github.com/zlccccc/3DVG-Transformer), [3D-SPS](https://github.com/fjhzhixi/3D-SPS), [3DJCG](https://github.com/zlccccc/3DVL_Codebase), [D3Net](https://github.com/daveredrum/D3Net), [M3DRef-CLIP](https://github.com/3dlg-hcvc/M3DRef-CLIP), etc.).

## Setup
### Conda (recommended)
We recommend the use of [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage system dependencies.

```shell
# create and activate the conda environment
conda create -n enet python=3.10
conda activate enet

# install PyTorch 2.0.1
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia

# install packages
pip install -r requirements.txt
```

### Pip
```shell
# create and activate the virtual environment
virtualenv env
source env/bin/activate

# install PyTorch 2.0.1
pip install torch torchvision

# install packages
pip install -r requirements.txt
```

## Data Preparation
### ScanNet v2 dataset
1. Download the [ScanNet v2 dataset (train/val/test)](http://www.scan-net.org/), the raw dataset files should be organized as follows:
    ```shell
    enet-scannet # project root
    ├── dataset
    │   ├── scannetv2
    │   │   ├── scans
    │   │   │   ├── [scene_id]
    │   │   │   │   ├── [scene_id].sens
    │   │   │   │   ├── [scene_id]_vh_clean_2.ply
    ```
2. Pre-process the data, it extracts video frames from `.sens` files:
   ```shell
   python dataset/scannetv2/preprocess_data.py +workers={cpu_count}
   ```
   The output files should have the following format:
   ```shell
   enet-scannet # project root
    ├── dataset
    │   ├── scannetv2
    │   │   ├── video_frames
    │   │   │   ├── [scene_id]
    │   │   │   │   ├── color
    │   │   │   │   │   ├── *.jpg
    │   │   │   │   ├── depth
    │   │   │   │   │   ├── *.png
    │   │   │   │   ├── pose
    │   │   │   │   │   ├── *.txt
   ```

### Pre-trained ENet weights
1. Download [pre-trained ENet weights](http://kaldir.vc.in.tum.de/ScanRefer/scannetv2_enet.pth), the file should be organized as follows:
   ```shell
    enet-scannet # project root
    ├── checkpoints
    │   ├── scannetv2_enet.pth
    ```

## Run
1. Extract enet features for video frames:
```shell
python extract_enet_features.py
```
Then, it outputs the `output/video_frame_features.h5` file with the following format:
```shell
 video_frame_features.h5 # the output file
 ├── [scene_id] # dataset
 │   ├── (frames, 128, 32, 41) # (#frames, #feature_channel, image_height, image_width)
 ...
```

2. Project enet features to point cloud:
```shell
python project_features_to_points.py
```
Then, it outputs `output/multiview_features_{train/val/test}.h5` files with the following format:
```shell
output/multiview_features_{train/val/test}.h5 # the output file
 ├── [scene_id] # dataset
 │   ├── (points, 128) # (#points, #feature_channel)
 ...
```

## Acknowledgement
This repo is built upon [D3Net](https://github.com/daveredrum/D3Net) and [ScanNet](https://github.com/ScanNet/ScanNet).