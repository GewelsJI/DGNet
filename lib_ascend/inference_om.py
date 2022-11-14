#!/usr/bin/env python
#-*-coding:utf-8-*-

# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Paper Title: Deep Gradient Learning for Camouflaged Object Detection
Project Page: https://github.com/GewelsJI/DGNet
Author: Ge-Peng Ji
Paper Citation:
@article{ji2022gradient,
  title={Deep Gradient Learning for Efficient Camouflaged Object Detection},
  author={Ji, Ge-Peng and Fan, Deng-Ping and Chou, Yu-Cheng and Dai, Dengxin and Liniger, Alexander and Van Gool, Luc},
  journal={Machine Intelligence Research},
  year={2023}
} 
"""

import os
import numpy as np
from mindx.sdk.base import Tensor, Model
import torch.nn.functional as F
import torch
from utils.dataset import test_dataset as EvalDataset
import imageio


def infer(filepath, SAVE_PATH, device_id):
    model = Model(filepath, device_id)
    print(model)

    val_loader = EvalDataset(image_root='./data/NC4K/Imgs/',
                             gt_root='./data/NC4K/GT/',
                             testsize=352)
    os.makedirs(SAVE_PATH, exist_ok=True)
    for i in range(val_loader.size):
        images, gt, name, _ = val_loader.load_data()
        gt = np.asarray(gt, np.float32)
        images = images.numpy()
        imageTensor = Tensor(images)
        imageTensor.to_device(device_id)
        out = model.infer(imageTensor)
        out = out[0]
        out.to_host()

        res = torch.from_numpy(np.array(out))
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('--> save results: {}'.format(SAVE_PATH+name))
        imageio.imwrite(SAVE_PATH+name, res)


if __name__ == "__main__":
    infer(
        filepath='./snapshots/DGNet-PVTv2-B3/DGNet-PVTv2-B3.om', 
        SAVE_PATH='./seg_results_om/Exp-DGNet-PVTv2-B3-OM/NC4K-Test/',
        device_id=2)