# DGNet-Jittor Implementation

## Introduction

The repo provides inference code of **DGNet** with [Jittor deep-learning framework](https://github.com/Jittor/jittor).

> **Jittor** is a high-performance deep learning framework based on JIT compiling and meta-operators. The whole framework and meta-operators are compiled just-in-time. A powerful op compiler and tuner are integrated into Jittor. It allowed us to generate high-performance code with specialized for your model. Jittor also contains a wealth of high-performance model libraries, including: image recognition, detection, segmentation, generation, differentiable rendering, geometric learning, reinforcement learning, etc. The front-end language is Python. Module Design and Dynamic Graph Execution is used in the front-end, which is the most popular design for deeplearning framework interface. The back-end is implemented by high performance language, such as CUDA, C++.

## Usage

The training and testing experiments are conducted using [Jiitor](https://github.com/Jittor/jittor) with a single GeForce RTX TITAN GPU.

1. Prerequisites:

   + Create environment by `python3.7 -m pip install jittor` on Linux. 
   As for MacOS or Windows users, using Docker `docker run --name jittor -v $PATH_TO_PROJECT:/home/DGNet -it jittor/jittor /bin/bash` is easier and necessary. 
   A simple way to debug and run the script is running a new command in the container through `docker exec -it jittor /bin/bash` and start the experiments. (More details refer to this [installation tutorial](https://github.com/Jittor/jittor#install))

2. Prepare the data:

    + downloading the testing dataset and moving it into `./dataset/TestDataset/`, which can be found in [Google Drive](https://drive.google.com/file/d/1L4zo8Mml08Q2sDPnqT01Nqxx4wv6FMDa/view?usp=sharing).
    + downloading the training dataset and moving it into `./dataset/TrainDataset/`, which can be found in [Google Drive](https://drive.google.com/file/d/11-5bBnfVal03D74dtRlJUpuWfmVLc8x9/view?usp=sharing).
    + downloading the pretrained weights of DGNet and DGNet-S and moving it into `./lib_pytorch/snapshot/`, which can be found in [Google Drive](https://drive.google.com/file/d/1Sy6cGDYGQVFnTxTGvmJu5QP7spAai6cT/view?usp=sharing).
    + preparing the EfficientNet-B1/B4 weights on ImageNet (refer to [here](https://github.com/GewelsJI/DGNet/blob/00e4d2b54667eb71f734f60d46fffe47fbf2725e/lib/utils.py#L556)).

3. Training Configuration:

    + Assigning your costumed path, like `--save_path `, `--train_root` and `--val_root` in `MyTrain.py`.
    + Just enjoy it via running `python ./lib_jittor/MyTrain.py` in your terminal.

4. Testing Configuration:

    + After you download all the pre-trained models and testing datasets, just run `./lib_jittor/MyTest.py` to generate the final
      prediction map: replace your trained model directory (`--snap_path`).
    + Just enjoy it via running `python ./lib_jittor/MyTest.py` in your terminal.

5. Evaluation Configuration

    + Assigning your costumed path, like `--CamMapPath `, `--DataPath`,`--ResDir`, `--Models`, `--Datasets` in `./eval/main.m`.
    + Just enjoy it via running `main.m` in MatLab.

> Note that the Jittor model is just converted from the original PyTorch model via toolbox, and thus, the trained weights of PyTorch model can be used to the inference of Jittor model.


## Performance Comparison

We submit the results of the Pytorch implemented DGNet in our manuscript, and we also re-train the DGNet that is rewritted by jittor framework. We observe that the performance has slight fluctuation between the re-trained Jittor-based DGNet and the Pytorch-based DGNet.

The download link of prediction results ([jittor results, google drive](https://drive.google.com/file/d/1-aYfI8oQp5c8gbiRTC57xEMOFCpIPgmS/view?usp=sharing)) on three testing dataset, including CAMO, COD10K, and NC4K.


|  CAMO-Test dataset   	| $S_\alpha$  	 | $E_\phi^{ad}$  	 | $F_\beta^w$  	 | $M$     	 |
|----------------------	|---------------|----------------|----------------|----------|
|  PyTorch             	| 0.839       	 | 0.901     	    | 0.769        	 | 0.057 	  |
|  Jittor              	| 0.841       	 | 0.891     	    | 0.774        	 | 0.057 	  |

|  COD10K-Test dataset 	| $S_\alpha$  	 | $E_\phi^{ad}$  	 | $F_\beta^w$  	 | $M$     	 |
|----------------------	|---------------|-------------|----------------|---------|
|  PyTorch             	| 0.822       	 | 0.877     	 | 0.693        	 | 0.033 	 |
|  Jittor              	| 0.826       	 | 0.879     	 | 0.700        	 | 0.033 	 |

| NC4K dataset    	      | $S_\alpha$  	 | $E_\phi^{ad}$  	   | $F_\beta^w$  	 | $M$     	    |
|------------------------|---------------|---------------|----------------|------------|
| PyTorch              	 | 0.857       	 | 0.907     	   | 0.784        	 | 0.042 	    |
| Jittor               	 | 0.859      	  | 0.909       	 | 0.789        	 | 0.042    	 |

## Speedup

Jittor-based framework could speed up the training duration, significantly decreasing the experimental efforts.
 
  | 	             | PyTorch    	    | Jittor     	    | Speedup    	  |
  |---------------|-----------------|-----------------|---------------|
  | DGNet         | 8.8 hours     	 | 6.4 hours     	 | 1.37x       	 |
  | DGNet-S     	 | 7.9 hours    	  | 4.8 hours    	  | 1.65x       	 |



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

    @article{hu2020jittor,
      title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
      author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
      journal={Information Sciences},
      volume={63},
      number={222103},
      pages={1--21},
      year={2020}
    }
