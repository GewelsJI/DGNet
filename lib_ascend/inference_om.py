# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

import os
import argparse

import cv2
import imageio
import mindspore
import numpy as np
from mindx.sdk.base import Tensor, Model


def get_image(image_path, mean, std):
    """
    get image by its path.
    :param 
        image_path: the path of image
        mean: the mean value of samples (from ImageNet)
        std: the std value of samples (from ImageNet)
    :return: a numpy array of image
    """
    image_bgr = cv2.imread(image_path)
    imge_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_size = imge_rgb.shape
    imge_rgb = cv2.resize(imge_rgb, (352, 352))
    imge_rgb = np.array([imge_rgb])
    image = imge_rgb.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    image = (image - np.asarray(mean)[None, :, None, None]) / np.asarray(std)[None, :, None, None]
    image = np.ascontiguousarray(image, dtype=np.float32)
    return image, image_size[0], image_size[1]


def infer(om_path, save_path, device_id, data_path='./data/NC4K/Imgs'):
    """
    Paper Title: Deep Gradient Learning for Camouflaged Object Detection
    Original Project Page: https://github.com/GewelsJI/DGNet
    Author: Ge-Peng Ji
    Paper Citation Bibtex:
    @article{ji2022gradient,
      title={Deep Gradient Learning for Efficient Camouflaged Object Detection},
      author={Ji, Ge-Peng and Fan, Deng-Ping and Chou, Yu-Cheng and Dai, 
              Dengxin and Liniger, Alexander and Van Gool, Luc},
      journal={Machine Intelligence Research},
      year={2023}
    } 
    """
    model = Model(om_path, device_id)
    print(model)
    
    os.makedirs(save_path, exist_ok=True)
    for img_name in os.listdir(data_path):
        image, h, w = get_image(
            os.path.join(data_path, img_name), 
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])

        # put image array into ascend ai processor
        image_tensor = Tensor(image)
        image_tensor.to_device(device_id)

        # infer
        out = model.infer(image_tensor)
        out = out[0]
        out.to_host()
        res = np.array(out)

        # save results
        res = mindspore.Tensor(res)
        
        res = mindspore.ops.Sigmoid()(res)
        res = mindspore.nn.ResizeBilinear()(res, (h, w))
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = res.asnumpy().squeeze()
        imageio.imwrite(save_path+img_name.replace('.jpg', '.png'), res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--om_path', type=str, 
        default='./snapshots/DGNet-PVTv2-B3/DGNet.om',
        help='the test rgb images root')
    parser.add_argument(
        '--save_path', type=str, 
        default='./seg_results_om/Exp-DGNet-OM/NC4K/',
        help='the test rgb images root')
    args = parser.parse_args()
    
    infer(
        om_path=args.om_path, 
        save_path=args.save_path,
        device_id=0)