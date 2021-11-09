***************************************
 Under Review Please Do Not Transfer
***************************************
# Deep Gradient Learning for Efficient Camouflaged Object Detection

PyTorch implementation of our Deep Gradient Network (DGNet).

## 1. Preface

- **Introduction.** This repository contains the source code our Deep Gradient Network, also called DGNet ([arXiv](), [SuppMaterial]()).

## 2. Overview

<p align="center">
    <img src="./assest/FeatureVis.pdf"/> <br />
    <em> 
    Figure 1: Visualizations of learned texture features. We observe that the feature under the (a) boundary-based supervision contains diffused noises in the background. 
    In contrast, (b) gradient-based supervision enforces the network focus on the regions where the intensity change dramatically..
    </em>
</p>

<p align="center">
    <img src="./assest/DGNet-Framework.pdf"/> <br />
    <em> 
    Figure 2: Overall pipeline of the proposed DGNet. From the perspective in Figure 1, we present a deep gradient network via explicitly object-level gradient map supervision. The underlying hypothesis is that there is some intensity change inside the camouflaged objects. It consists of two connected learning branches, i.e., context encoder and texture encoder. Then, we introduce the gradient-induced transition (GIT) to collaboratively aggregate the feature that is derived from the above two encoders. Finally, a neighbor connected decoder (NCD) is adopted to generate the prediction.
    </em>
</p>

<p align="center">
    <img src="./assest/BubbleBarFig.pdf"/> <br />
    <em> 
    Figure 3: We present the scatter relationship between the performance weighted F-measure and parameters of all competitors on CAMO-Test.
    These scatters are in various colors for better visual recognition and are also corresponding to the histogram (Right).
    The larger size of the coloured scatter point, the heavier the model parameter.
    (Right) We also report the parallel histogram comparison of model's parameters, MACs, and performance.
    </em>
</p>

## 3. Inference

1. Prepare the data:

    + downloading testing dataset and move it into `./dataset/TestDataset/`, 
    which can be found in [Baidu Drive](https://pan.baidu.com/s/1Gg9zco1rt8314cuemqMFBg) (Password: 3wih), [Google Drive](https://drive.google.com/file/d/1LraHmnmgqibzqpqTi4E4l1O2MTusJjrZ/view?usp=sharing).

1. Inference Configuration:

    + After you download the testing dataset, just run `bash inference.sh` to create a virtual environment and installing necessary packages to generate the final prediction map: 
    
    + Just enjoy it!

---

**[⬆ back to top](#0-preface)**
