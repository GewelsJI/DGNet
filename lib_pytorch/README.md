# DGNet-Pytorch Implementation

## Introduction

The repo provides the inference code of **DGNet** with [Pytorch deep-learning framework](https://github.com/pytorch/pytorch).

## Usage

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with a single GeForce RTX TITAN GPU.

1. Prerequisites:

   Note that DGNet is only tested on Ubuntu OS with the following environments. It may work on other operating systems (
   i.e., Windows) as well but we do not guarantee that it will.

    + Creating a virtual environment in terminal: `conda create -n DGNet python=3.6`.

    + Installing necessary packages: `pip install -r ./lib_pytorch/requirements.txt`
      `

2. Prepare the data:

    + downloading the testing dataset and moving it into `./dataset/TestDataset/`, which can be found in [Google Drive](https://drive.google.com/file/d/1L4zo8Mml08Q2sDPnqT01Nqxx4wv6FMDa/view?usp=sharing).
    + downloading the training dataset and moving it into `./dataset/TrainDataset/`, which can be found in [Google Drive](https://drive.google.com/file/d/11-5bBnfVal03D74dtRlJUpuWfmVLc8x9/view?usp=sharing).
    + downloading the pretrained weights of DGNet and DGNet-S and moving it into `./lib_pytorch/snapshot/`, which can be found in [Google Drive](https://drive.google.com/file/d/1ynUqt6DIHWQv7dT0vIbSOANtZsVvFdHs/view?usp=sharing).
    + preparing the EfficientNet-B1/B4 weights on ImageNet (refer to [here](https://github.com/GewelsJI/DGNet/blob/00e4d2b54667eb71f734f60d46fffe47fbf2725e/lib/utils.py#L556)).

3. Training Configuration:
    + move into `cd lib_pytorch`
    + Assigning your costumed path `MyTrain.py`.
        + train DGNet (EfficientNet-B4): `python MyTrain.py --gpu_id 0 --model DGNet --save_path ./snapshot/Exp-DGNet/`
        + train DGNet-S (EfficientNet-B0): `python MyTrain.py --gpu_id 0 --model DGNet-S  --save_path ./snapshot/Exp-DGNet-S/`
        + train DGNet-PVTv2-B0: `python MyTrain.py --gpu_id 0 --model DGNet-PVTv2-B0  --save_path ./snapshot/Exp-DGNet-PVTv2-B0/`
        + train DGNet-PVTv2-B1: `python MyTrain.py --gpu_id 0 --model DGNet-PVTv2-B1  --save_path ./snapshot/Exp-DGNet-PVTv2-B1/`
        + train DGNet-PVTv2-B2: `python MyTrain.py --gpu_id 0 --model DGNet-PVTv2-B2  --save_path ./snapshot/Exp-DGNet-PVTv2-B2/`
        + train DGNet-PVTv2-B3: `python MyTrain.py --gpu_id 0 --model DGNet-PVTv2-B3  --save_path ./snapshot/Exp-DGNet-PVTv2-B3/`
        + train DGNet-PVTv2-B4: `python MyTrain.py --gpu_id 0 --model DGNet-PVTv2-B4  --save_path ./snapshot/Exp-DGNet-PVTv2-B4/`
    + We also have plan to release more DGNet variants with different backbones. Please stay tuned.

4. Testing Configuration:
    + move into `cd lib_pytorch`
    + After you download all the pre-trained models and testing datasets, just run `./lib_pytorch/MyTest.py` to generate the final
      prediction map: replace your trained model directory (`--snap_path`).
        + test DGNet (EfficientNet-B4): `python MyTest.py --gpu_id 0 --model DGNet --snap_path ./snapshot/DGNet.pth`
        + test DGNet-S (EfficientNet-B1): `python MyTest.py --gpu_id 0 --model DGNet-S --snap_path ./snapshot/DGNet-S.pth`
        + test DGNet-PVTv2-B0: `python MyTest.py --gpu_id 0 --model DGNet-PVTv2-B0 --snap_path ./snapshot/DGNet-PVTv2-B0.pth`
        + test DGNet-PVTv2-B1: `python MyTest.py --gpu_id 0 --model DGNet-PVTv2-B1 --snap_path ./snapshot/DGNet-PVTv2-B1.pth`
        + test DGNet-PVTv2-B2: `python MyTest.py --gpu_id 0 --model DGNet-PVTv2-B2 --snap_path ./snapshot/DGNet-PVTv2-B2.pth`
        + test DGNet-PVTv2-B3: `python MyTest.py --gpu_id 0 --model DGNet-PVTv2-B3 --snap_path ./snapshot/DGNet-PVTv2-B3.pth`
        + test DGNet-PVTv2-B4: `python MyTest.py --gpu_id 0 --model DGNet-PVTv2-B4 --snap_path ./snapshot/DGNet-PVTv2-B4.pth`
    + All the segmentation predictions of the above DGNet variants could be found in [Google Drive](https://drive.google.com/file/d/1LNqfZ3mYkNjtM3xG4tzZSee1WQMBPmDj/view?usp=sharing).

5. Evaluation Configuration
    + Matlab-style eval (reported in our paper)
        + Assigning your costumed path, like `--CamMapPath `, `--DataPath`,`--ResDir`, `--Models`, `--Datasets` in `./eval/main.m`.
        + Just enjoy it via running `main.m` in MatLab.
    + Python-style eval (revised from [PySODMetrics](https://github.com/lartpang/PySODMetrics) and [SOCToolbox](https://github.com/mczhuge/SOCToolbox) projects)
        + Assigning your costumed competitors (`--model_lst`) in `./lib_pytorch/MyEval.py`
        + Just evaluate them via running  `python MyEval.py`
        + Please note that there maybe has a slight difference between matlab-style and python-style codes.
        + The below table presents the performance comparison among different model variants on COD10K dataset. Completed evaluation results (*.txt) are available at [here](https://github.com/GewelsJI/DGNet/tree/main/lib_pytorch/eval_txt/20221103_DGNet_benchmark).

      | Method         | Input Size | S-measure | Max E-measure | #Params (M)  | #MACs (G) | Results Download | Checkpoint Download |
      |---|---|---|---|---|---|---|---|
      | DGNet-S (EfficientNet-B1)        |  352 |   0.810   |     0.904     |     7.02    |     1.20    | [GoogleDrive](https://drive.google.com/file/d/1so0wsQmJIfi4IAtWYcdfk_3Z3TWHyOuh/view?usp=sharing) | [GoogleDrive-32M](https://drive.google.com/file/d/1dEVmkTtrnuKm57tbwDGwoaopmBmhJwbb/view?usp=sharing) |
      | DGNet (EfficientNet-B4)         |  352 |   0.822   |     0.911     |    19.22    |     2.77    | [GoogleDrive](https://drive.google.com/file/d/1VFM3tWsHrhzIeujodR6ql18DlxVKpI3e/view?usp=sharing) | [GoogleDrive-80M](https://drive.google.com/file/d/136-nbdodomrfO0TcHah84W9gA-PHJDfZ/view?usp=drive_link) |
      | DGNet-PVTv2-B0 |  352 |   0.801   |     0.890     |    15.34    |     7.86    | [GoogleDrive](https://drive.google.com/file/d/1OVq-mEO2bTtB6YGuoo5Zdh9PvWuqJl8v/view?usp=sharing) | [GoogleDrive-16M](https://drive.google.com/file/d/1972kvkDyyVJl4IPwvAMYKGmgOq0tjf-y/view?usp=drive_link) |
      | DGNet-PVTv2-B1 |  352 |   0.826   |     0.908     |    26.70   |     12.45    | [GoogleDrive](https://drive.google.com/file/d/1J-eSTjjEX8nVIS8lHaM7Jd1zFHquESRR/view?usp=sharing) | [GoogleDrive-60M](https://drive.google.com/file/d/1qQYVfITH6AeBQiNWcckqHlcepKurJejz/view?usp=drive_link) |
      | DGNet-PVTv2-B2 |  352 |   0.844   |     0.926     |    46.57    |     19.38    | [GoogleDrive](https://drive.google.com/file/d/10FTWjj3_Gyq-7V_2c9BSgieG2a9lTqMQ/view?usp=sharing) | [GoogleDrive-104M](https://drive.google.com/file/d/1yXX4-qj6RUVvvokQDZvQM6CQ9MhF15dg/view?usp=drive_link) |
      | DGNet-PVTv2-B3 |  352 |   0.851   |     0.931     |    63.89    |     27.08    | [GoogleDrive](https://drive.google.com/file/d/1H2nAnmlv7eD_Ef4zF8t2SUYL6-jO44mX/view?usp=sharing) | [GoogleDrive-180M](https://drive.google.com/file/d/1TntFFGF3eWfEc77208KO4D949otuhoJ_/view?usp=drive_link) |



## Citation

If you find our work useful in your research, please consider citing:
    
    
      @article{ji2023gradient,
        title={Deep Gradient Learning for Efficient Camouflaged Object Detection},
        author={Ji, Ge-Peng and Fan, Deng-Ping and Chou, Yu-Cheng and Dai, Dengxin and Liniger, Alexander and Van Gool, Luc},
        journal={Machine Intelligence Research},
        pages={92-108},
        volume={20},
        issue={1},
        year={2023}
      } 
