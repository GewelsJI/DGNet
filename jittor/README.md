# Deep Gradient Learning for Efficient Camouflaged Object Detection (DGNet-Jittor Implementation)

## Introduction

The repo provides inference code of **DGNet** with [Jittor deep-learning framework](https://github.com/Jittor/jittor).

> **Jittor** is a high-performance deep learning framework based on JIT compiling and meta-operators. The whole framework and meta-operators are compiled just-in-time. A powerful op compiler and tuner are integrated into Jittor. It allowed us to generate high-performance code with specialized for your model. Jittor also contains a wealth of high-performance model libraries, including: image recognition, detection, segmentation, generation, differentiable rendering, geometric learning, reinforcement learning, etc. The front-end language is Python. Module Design and Dynamic Graph Execution is used in the front-end, which is the most popular design for deeplearning framework interface. The back-end is implemented by high performance language, such as CUDA, C++.

## Usage

DGNet is also implemented in the Jittor toolbox which can be found in `./jttor`.
+ Create environment by `python3.7 -m pip install jittor` on Linux. 
As for MacOS or Windows users, using Docker `docker run --name jittor -v $PATH_TO_PROJECT:/home/DGNet -it jittor/jittor /bin/bash` 
is easier and necessary. 
A simple way to debug and run the script is running a new command in the container through `docker exec -it jittor /bin/bash` and start the experiments. (More details refer to this [installation tutorial](https://github.com/Jittor/jittor#install))

+ First, run `sudo sysctl vm.overcommit_memory=1` to set the memory allocation policy.

+ Second, switch to the project root by `cd /home/DGNet`

+ For testing, run `python3.7 jittor/MyTest.py`. 

> Note that the Jittor model is just converted from the original PyTorch model via toolbox, and thus, the trained weights of PyTorch model can be used to the inference of Jittor model.


## Citation

If you find our work useful in your research, please consider citing:
    
    
    @article{ji2022gradient,
          title={Deep Gradient Learning for Efficient Camouflaged Object Detection},
          author={Ge-Peng Ji and Deng-Ping Fan and Yu-Cheng Chou and Dengxin Dai and Alexander Liniger and Luc Van Gool},
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
