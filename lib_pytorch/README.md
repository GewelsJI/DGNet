# DGNet-Pytorch Implementation

## Introduction

The repo provides inference code of **DGNet** with [Pytorch deep-learning framework](https://github.com/pytorch/pytorch).

## Usage

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with a single GeForce RTX TITAN GPU.

1. Prerequisites:

   Note that DGNet is only tested on Ubuntu OS with the following environments. It may work on other operating systems (
   i.e., Windows) as well but we do not guarantee that it will.

    + Creating a virtual environment in terminal: `conda create -n DGNet python=3.6`.

    + Installing necessary packages: `pip install -r ./lib_pytorch/requirements.txt`
      `

2. Prepare the data:

    + downloading testing dataset and move it into `./dataset/TestDataset/`, which can be found in [OneDrive](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EXcBqW3Ses5HlYFeTAPlmiwBtPwXisbr53uIDGoM4h0UOg?e=d5tK9C).
    + downloading training dataset and move it into `./dataset/TrainDataset/`, which can be found in [OneDrive](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EUgtKNJSBYpElpgQzrIZLDEBmu9Stp5UL3P5HHkrHGXIyQ?e=5OgCok).
    + downloading pretrained weights of DGNet and DGNet-S and move it into `./lib_pytorch/snapshot/`, which can be found in [GitHub](https://github.com/GewelsJI/DGNet/releases/tag/Checkpoints).
    + preparing the EfficientNet-B1/B4 weights on ImageNet (refer to [here](https://github.com/GewelsJI/DGNet/blob/00e4d2b54667eb71f734f60d46fffe47fbf2725e/lib/utils.py#L556)).

3. Training Configuration:
    + move into `cd lib_pytorch`
    + Assigning your costumed path `MyTrain.py`.
        + train DGNet (EfficientNet-B4): `python MyTrain.py --gpu_id 0 --model DGNet --save_path ./snapshot/Exp-DGNet/`
        + train DGNet-S (EfficientNet-B0): `python MyTrain.py --gpu_id 0 --model DGNet-S  --save_path ./snapshot/Exp-DGNet-S/`
        + train DGNet-PVTv2-B0: `python MyTrain.py --gpu_id 1 --model DGNet-PVTv2-B0  --save_path ./snapshot/Exp-DGNet-PVTv2-B0/`
        + train DGNet-PVTv2-B1: `python MyTrain.py --gpu_id 1 --model DGNet-PVTv2-B1  --save_path ./snapshot/Exp-DGNet-PVTv2-B1/`
        + train DGNet-PVTv2-B2: `python MyTrain.py --gpu_id 0 --model DGNet-PVTv2-B2  --save_path ./snapshot/Exp-DGNet-PVTv2-B2/`
        + train DGNet-PVTv2-B3: `python MyTrain.py --gpu_id 1 --model DGNet-PVTv2-B3  --save_path ./snapshot/Exp-DGNet-PVTv2-B3/`
    + We also have plan to release more DGNet variants with different backbones. Please stay tuned.

4. Testing Configuration:
    + move into `cd lib_pytorch`
    + After you download all the pre-trained models and testing datasets, just run `./lib_pytorch/MyTest.py` to generate the final
      prediction map: replace your trained model directory (`--snap_path`).
        + test DGNet (EfficientNet-B4): `python MyTest.py --gpu_id 1 --model DGNet --snap_path ./snapshot/DGNet.pth`
        + test DGNet-S (EfficientNet-B0): `python MyTest.py --gpu_id 1 --model DGNet-S --snap_path ./snapshot/DGNet-S.pth`
        + test DGNet-PVTv2-B0: `python MyTest.py --gpu_id 1 --model DGNet-PVTv2-B0 --snap_path ./snapshot/DGNet-PVTv2-B0.pth`
        + test DGNet-PVTv2-B1: `python MyTest.py --gpu_id 1 --model DGNet-PVTv2-B1 --snap_path ./snapshot/DGNet-PVTv2-B1.pth`
        + test DGNet-PVTv2-B2: `python MyTest.py --gpu_id 0 --model DGNet-PVTv2-B2 --snap_path ./snapshot/DGNet-PVTv2-B2.pth`
        + test DGNet-PVTv2-B3: `python MyTest.py --gpu_id 0 --model DGNet-PVTv2-B3 --snap_path ./snapshot/DGNet-PVTv2-B3.pth`
    + All the segmentation predictions of above DGNet variants could be found in [OneDrive]().

5. Evaluation Configuration
    + Matlab-style eval (reported in our paper)
        + Assigning your costumed path, like `--CamMapPath `, `--DataPath`,`--ResDir`, `--Models`, `--Datasets` in `./eval/main.m`.
        + Just enjoy it via running `main.m` in MatLab.
    + Python-style eval (revised from [PySODMetrics](https://github.com/lartpang/PySODMetrics) and [SOCToolbox](https://github.com/mczhuge/SOCToolbox) projects)
        + Assigning your costumed competitors (`--model_lst`) in `./lib_pytorch/MyEval.py`
        + Just evaluate them via running  `python MyEval.py`
        + Please note that there maybe has a slight difference between matlab-style and python-style codes.
        + The below table presents the performance comparison among different model variants on COD10K dataset. Completed evaluation results are available at [here](https://github.com/GewelsJI/DGNet/tree/main/lib_pytorch/eval_txt/20221103_DGNet_benchmark).

      | Method         | Input Size | S-measure | Max E-measure | #Params (M) | Download                                                                                            |
      |----------------|:----:|:---------:|:-------------:|:-----------:|-----------------------------------------------------------------------------------------------------|
      | DGNet-S (EfficientNet-B0)        |  352 |   0.810   |     0.904     |     7.02    | 32M  [[GitHub]](https://github.com/GewelsJI/DGNet/releases/download/Checkpoints/DGNet-S.pth)        |
      | DGNet (EfficientNet-B4)         |  352 |   0.822   |     0.911     |    19.22    | 80M  [[GitHub]](https://github.com/GewelsJI/DGNet/releases/download/Checkpoints/DGNet.pth)          |
      | DGNet-PVTv2-B0 |  352 |   0.801   |     0.890     |    15.34    | 16M  [[GitHub]](https://github.com/GewelsJI/DGNet/releases/download/Checkpoints/DGNet-PVTv2-B0.pth) |
      | DGNet-PVTv2-B1 |  352 |   0.826   |     0.908     |    26.70    | 60M  [[GitHub]](https://github.com/GewelsJI/DGNet/releases/download/Checkpoints/DGNet-PVTv2-B1.pth) |
      | DGNet-PVTv2-B2 |  352 |   0.844   |     0.926     |    46.57    | 104M [[GitHub]](https://github.com/GewelsJI/DGNet/releases/download/Checkpoints/DGNet-PVTv2-B2.pth) |
      | DGNet-PVTv2-B3 |  352 |   0.851   |     0.931     |    63.89    | 180M [[GitHub]](https://github.com/GewelsJI/DGNet/releases/download/Checkpoints/DGNet-PVTv2-B3.pth) |



## Citation

If you find our work useful in your research, please consider citing:
    
    
    @article{ji2022gradient,
      title={Deep Gradient Learning for Efficient Camouflaged Object Detection},
      author={Ji, Ge-Peng and Fan, Deng-Ping and Chou, Yu-Cheng and Dai, Dengxin and Liniger, Alexander and Van Gool, Luc},
      journal={Machine Intelligence Research},
      year={2023}
    } 