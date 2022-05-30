# DGNet-Jittor Implementation

## Introduction

The repo provides inference code of **DGNet** with [Jittor deep-learning framework](https://github.com/Jittor/jittor).

> **Jittor** is a high-performance deep learning framework based on JIT compiling and meta-operators. The whole framework and meta-operators are compiled just-in-time. A powerful op compiler and tuner are integrated into Jittor. It allowed us to generate high-performance code with specialized for your model. Jittor also contains a wealth of high-performance model libraries, including: image recognition, detection, segmentation, generation, differentiable rendering, geometric learning, reinforcement learning, etc. The front-end language is Python. Module Design and Dynamic Graph Execution is used in the front-end, which is the most popular design for deeplearning framework interface. The back-end is implemented by high performance language, such as CUDA, C++.

## Usage

The training and testing experiments are conducted using [Jiitor](https://github.com/Jittor/jittor) with a single
GeForce RTX TITAN GPU.

1. Prerequisites:

   + Create environment by `python3.7 -m pip install jittor` on Linux. 
   As for MacOS or Windows users, using Docker `docker run --name jittor -v $PATH_TO_PROJECT:/home/DGNet -it jittor/jittor /bin/bash` is easier and necessary. 
   A simple way to debug and run the script is running a new command in the container through `docker exec -it jittor /bin/bash` and start the experiments. (More details refer to this [installation tutorial](https://github.com/Jittor/jittor#install))

2. Prepare the data:

    + downloading testing dataset and move it into `./dataset/TestDataset/`, which can be found in [OneDrive](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EXcBqW3Ses5HlYFeTAPlmiwBtPwXisbr53uIDGoM4h0UOg?e=d5tK9C).
    + downloading training dataset and move it into `./dataset/TrainDataset/`, which can be found in [OneDrive](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EUgtKNJSBYpElpgQzrIZLDEBmu9Stp5UL3P5HHkrHGXIyQ?e=5OgCok).
    + downloading pretrained weights of Jittor implementation DGNet and DGNet-S and move it into `./jittor_lib/snapshot/`, which can be found in [OneDrive](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EezZ9PWXpGZOkEIieYBA5esBsITit1pKkUjdntvUGut_Dw?e=Y20PrV).
    + preparing the EfficientNet-B1/B4 weights on ImageNet (refer to [here](https://github.com/GewelsJI/DGNet/blob/00e4d2b54667eb71f734f60d46fffe47fbf2725e/lib/utils.py#L556)).

3. Training Configuration:

    + Assigning your costumed path, like `--save_path `, `--train_root` and `--val_root` in `MyTrain.py`.
    + Just enjoy it via running `python ./jittor_lib/MyTrain.py` in your terminal.

4. Testing Configuration:

    + After you download all the pre-trained models and testing datasets, just run `./jittor_lib/MyTest.py` to generate the final
      prediction map: replace your trained model directory (`--snap_path`).
    + Just enjoy it via running `python ./jittor_lib/MyTest.py` in your terminal.

5. Evaluation Configuration

    + Assigning your costumed path, like `--CamMapPath `, `--DataPath`,`--ResDir`, `--Models`, `--Datasets` in `./eval/main.m`.
    + Just enjoy it via running `main.m` in MatLab.

> Note that the Jittor model is just converted from the original PyTorch model via toolbox, and thus, the trained weights of PyTorch model can be used to the inference of Jittor model.


## Performance Comparison

The performance has slight difference due to the different operator implemented between two frameworks.  The download link ([Pytorch](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EcwgyI1KDnBDjoFMZCLNJkAB7GjBYGgvDPlBAruSAVCOxw?e=RrBvHd) / [Jitror](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EbRmYVvdBIhEtRKWBUhzsNMBQ8F7Pnw7sUBAPDeN_Po_6A?e=crMmfP)) of prediction results on four testing dataset, including CAMO, COD10K, NC4K.


|  CAMO-Test dataset   	| $S_\alpha$  	 | $E_\phi$  	 | $F_\beta^w$  	 | M     	 |
|----------------------	|---------------|-------------|----------------|---------|
|  PyTorch             	| 0.839       	 | 0.901     	 | 0.769        	 | 0.057 	 |
|  Jittor              	| 0.841       	 | 0.891     	 | 0.774        	 | 0.057 	 |

|  COD10K-Test dataset 	| $S_\alpha$  	 | $E_\phi$  	 | $F_\beta^w$  	 | M     	 |
|----------------------	|---------------|-------------|----------------|---------|
|  PyTorch             	| 0.822       	 | 0.877     	 | 0.693        	 | 0.033 	 |
|  Jittor              	| 0.826       	 | 0.879     	 | 0.700        	 | 0.033 	 |

| NC4K dataset    	      | $S_\alpha$  	 | $E_\phi$  	   | $F_\beta^w$  	 | M     	    |
|------------------------|---------------|---------------|----------------|------------|
| PyTorch              	 | 0.857       	 | 0.907     	   | 0.784        	 | 0.042 	    |
| Jittor               	 | 0.859      	  | 0.909       	 | 0.789        	 | 0.042    	 |

## Citation

If you find our work useful in your research, please consider citing:
    
    
    @article{ji2022gradient,
       title={Deep Gradient Learning for Efficient Camouflaged Object Detection},
       author={Ji, Ge-Peng and Fan, Deng-Ping and Chou, Yu-Cheng and Dai, Dengxin and Liniger, Alexander and Van Gool, Luc},
       journal={arXiv},
       year={2022}
    }
    
    @article{hu2020jittor,
      title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
      author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
      journal={Information Sciences},
      volume={63},
      number={222103},
      pages={1--21},
      year={2020}
    }