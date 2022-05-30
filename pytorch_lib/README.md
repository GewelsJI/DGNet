# DGNet-Pytorch Implementation

## Introduction

The repo provides inference code of **DGNet** with [Pytorch deep-learning framework](https://github.com/pytorch/pytorch).

## Usage

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with a single
GeForce RTX TITAN GPU.

1. Prerequisites:

   Note that DGNet is only tested on Ubuntu OS with the following environments. It may work on other operating systems (
   i.e., Windows) as well but we do not guarantee that it will.

    + Creating a virtual environment in terminal: `conda create -n DGNet python=3.6`.

    + Installing necessary packages: `pip install -r ./pytorch_lib/requirements.txt
      `

2. Prepare the data:

    + downloading testing dataset and move it into `./dataset/TestDataset/`, which can be found in [OneDrive](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EXcBqW3Ses5HlYFeTAPlmiwBtPwXisbr53uIDGoM4h0UOg?e=d5tK9C).
    + downloading training dataset and move it into `./dataset/TrainDataset/`, which can be found in [OneDrive](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EUgtKNJSBYpElpgQzrIZLDEBmu9Stp5UL3P5HHkrHGXIyQ?e=5OgCok).
    + downloading pretrained weights of DGNet and DGNet-S and move it into `./pytorch_lib/snapshot/`, which can be found in [OneDrive](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EdjQje05VJZPoEFfFRLWT0sBsevoeyFE8O3PyCRCusUK1A?e=P0Fi9M).
    + preparing the EfficientNet-B1/B4 weights on ImageNet (refer to [here](https://github.com/GewelsJI/DGNet/blob/00e4d2b54667eb71f734f60d46fffe47fbf2725e/lib/utils.py#L556)).

3. Training Configuration:

    + Assigning your costumed path, like `--save_path `, `--train_root` and `--val_root` in `MyTrain.py`.
    + Just enjoy it via running `python ./pytorch_lib/MyTrain.py` in your terminal.

4. Testing Configuration:

    + After you download all the pre-trained models and testing datasets, just run `./pytorch_lib/MyTest.py` to generate the final
      prediction map: replace your trained model directory (`--snap_path`).
    + Just enjoy it via running `python ./pytorch_lib/MyTest.py` in your terminal.

5. Evaluation Configuration

    + Assigning your costumed path, like `--CamMapPath `, `--DataPath`,`--ResDir`, `--Models`, `--Datasets` in `./eval/main.m`.
    + Just enjoy it via running `main.m` in MatLab.


## Citation

If you find our work useful in your research, please consider citing:
    
    
    @article{ji2022gradient,
       title={Deep Gradient Learning for Efficient Camouflaged Object Detection},
       author={Ji, Ge-Peng and Fan, Deng-Ping and Chou, Yu-Cheng and Dai, Dengxin and Liniger, Alexander and Van Gool, Luc},
       journal={arXiv},
       year={2022}
    }