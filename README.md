# <p align=center>`Deep Gradient Learning for Efficient Camouflaged Object Detection`</p>

> **Authors:**
> [Ge-Peng Ji](https://github.com/GewelsJI),
> [Deng-Ping Fan](https://dengpingfan.github.io/),
> [Yu-Cheng Chou](https://github.com/johnson111788),
> [Dengxin Dai](https://vas.mpi-inf.mpg.de/dengxin/),
> [Alexander Liniger](https://people.ee.ethz.ch/~aliniger/) &
> [Luc Van Gool](https://ee.ethz.ch/the-department/faculty/professors/person-detail.OTAyMzM=.TGlzdC80MTEsMTA1ODA0MjU5.html).

This repository contains the source code, prediction results, and evaluation toolbox of our Deep Gradient Network, also called DGNet. Technical report could be found at [arXiv]().

> If you have any questions about our paper, feel free to contact me via e-mail (gepengai.ji@gmail.com & johnson111788@gmail.com &dengpfan@gmail.com). And if you are using our code and evaluation toolbox for your research, please cite this paper ([BibTeX](#4-citation)).


## 1. Features

<p align="center">
    <img src="assest/BubbleBarFig.png"/> <br />
    <em> 
    Figure 1: We present the scatter relationship between the performance weighted F-measure and parameters of all competitors on CAMO-Test. These scatters are in various colors for better visual recognition and are also corresponding to the histogram (Right).
    The larger size of the coloured scatter point, the heavier the model parameter. (Right) We also report the parallel histogram comparison of model's parameters, MACs, and performance.
    </em>
</p>

- **Novel supervision.** We propose to excavate the texture information via learning the objectlevel gradient rather than using boundary-aware or uncertainty-aware modelling.

- **Simple but efficient.** We decouple all the heavy designs as much as we can, yeilding a simple but efficient framework. We hope this framework could be served as a baseline learning paradigm for the COD field.

- **Best trade-off.** We achieve new SOTA with the best performance-efficiency trade-off on existing cutting-edge COD benchmarks.

## 2. :fire: NEWS :fire:

- [2022/05/25] Releasing the project and whole benmarking results.
- [2022/05/23] Creating repository.


## 3. Proposed Framework

### 3.1. Overview

<p align="center">
    <img src="assest/DGNetFramework.png"/> <br />
    <em> 
    Figure 2: Overall pipeline of the proposed DGNet, It consists of two connected learning branches, i.e., context encoder and texture encoder. 
    Then, we introduce a gradient-induced transition (GIT) to collaboratively aggregate the feature that is derived from the above two encoders. Finally, a neighbor connected decoder (NCD [1]) is adopted to generate the prediction.
    </em>
</p>


<p align="center">
    <img src="assest/GIT.png"/> <br />
    <em> 
    Figure 3: Illustration of the proposed gradient-induced transition (GIT). 
    It use a soft grouping strategy to provide parallel nonlinear projections at multiple fine-grained sub-spaces, which enables the network to probe multi-source representations jointly.
    </em>
</p>


> References of neighbor connected decoder (NCD) benchmark works<br>
> [1] Concealed Object Detection. TPAMI, 2022. ([Code Page](https://github.com/GewelsJI/SINet-V2))<br>

### 3.2. Usage

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with a single
GeForce RTX TITAN GPU.

1. Prerequisites:

   Note that DGNet is only tested on Ubuntu OS with the following environments. It may work on other operating systems (
   i.e., Windows) as well but we do not guarantee that it will.

    + Creating a virtual environment in terminal: `conda create -n DGNet python=3.6`.

    + Installing necessary packages: `pip install -r requirements.txt
      `

2. Prepare the data:

    + downloading testing dataset and move it into `./dataset/TestDataset/`, which can be found in [OneDrive](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EQixoFPEPnBHoH6tnG69Ip4BDu8H0-lZAsRkd_lk0hmvMA?e=rsc2eH).
    + downloading training dataset and move it into `./dataset/TrainDataset/`, which can be found in [OneDrive](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/ES4rY6EjIrxEp6wsArncLywBxGOQgIXSTWGe2YPCMzHeqQ?e=Qx2hMV).
    + downloading pretrained weights of DGNet and DGNet-S and move it into `./snapshot/`, which can be found in [OneDrive](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EdjQje05VJZPoEFfFRLWT0sBsevoeyFE8O3PyCRCusUK1A?e=P0Fi9M).
    + preparing the EfficientNet-B1/B4 weights on ImageNet (refer to [here](https://github.com/GewelsJI/DGNet/blob/00e4d2b54667eb71f734f60d46fffe47fbf2725e/lib/utils.py#L556)).

3. Training Configuration:

    + Assigning your costumed path, like `--save_path `, `--train_root` and `--val_root` in `MyTrain.py`.
    + Just enjoy it via running `python MyTrain.py` in your terminal.

4. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `MyTest.py` to generate the final
      prediction map: replace your trained model directory (`--snap_path`).

    + Just enjoy it!

5. Evaluation Configuration

    + Assigning your costumed path, like `--gt_root `, `--pred_root`,`--data_lst` and `--model_lst` in `MyEval.py`.
    + You can choose to evaluate the model by default setting or evaluate only the super-/subclass by configure
      the `--eval_type` in `MyEval.py`.
    + Just enjoy it via running `python MyEval.py` in your terminal.

### 3.3 Evaluation

One-key evaluation is written in MATLAB code `./eval/`, please follow this the instructions in `./eval/main.m` and just
run it to generate the evaluation results in `./eval-result/`.

### 3.4 COD Benchmark Results:

The prediction of our DGNet and DGNet-S can be found in [OneDrive](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EfYhCVo-L4ZAmrQoq0oVD9kBdL7LO1wQEwmeJjS92u9nLA?e=IB3Onb). The whole benchmark results can be found at [OneDrive](https://anu365-my.sharepoint.com/:u:/g/personal/u7248002_anu_edu_au/EUU-S6qsNWZJj7FEsvLPuv0Bu3CXAUaZQY7fKbwRdAqRGw?e=OYTyXQ).


<p align="center">
    <img src="assest/QualitativeResult_new_elite_v8.png"/> <br />
    <em> 
    Figure 4: Visualization of popular COD baselines and the proposed DGNet. Interestingly, these competitors fail to provide complete segmentation results for the camouflaged objects that touch the image boundary. By contrast, our approach can precisely locate the target region and provide exact predictions due to the gradient learning strategy.
    </em>

</p>

## 4. Citation

Please cite our paper if you find the work useful:

    @article{ji2022gradient,
          title={Deep Gradient Learning for Efficient Camouflaged Object Detection},
          author={Ji, Ge-Peng and Fan, Deng-Ping and Chou, Yu-Cheng and Dai, Dengxin and Liniger, Alexander and Van Gool, Luc},
          journal={arXiv},
          year={2022}
    } 

---

**[⬆ back to top](#0-preface)**
