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

import cv2
import torch
import numpy as np
import os
import onnxruntime
import torch.nn.functional as F


def inference_onnx(ROOT_PATH, OUT_PATH, SNAP_PATH):
    for img_name in os.listdir(ROOT_PATH):
        # 加载图片
        image_src = cv2.imread(ROOT_PATH+img_name)

        # 对图片进行预处理
        resized = cv2.resize(image_src, (352, 352), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0

        # 加载ONNX模型
        session = onnxruntime.InferenceSession(SNAP_PATH)
        input_name = session.get_inputs()[0].name

        # 执行推理流程
        onnx_out = session.run(None, {input_name: img_in})
        onnx_out_tensor = torch.from_numpy(onnx_out[0])

        onnx_out_tensor = F.upsample(onnx_out_tensor, size=image_src.shape[:2], mode='bilinear', align_corners=False)
        onnx_out = onnx_out_tensor.sigmoid().data.cpu().numpy().squeeze()

        os.makedirs(OUT_PATH, exist_ok=True)

        res = 255.0 * onnx_out
        res = res.astype(np.uint8)
        # 保存结果图到对应文件夹
        cv2.imwrite(OUT_PATH+img_name, res)
        print('--> predict: ', OUT_PATH+img_name)


if __name__ == '__main__':
    inference_onnx(
        ROOT_PATH = './data/NC4K/Imgs/',
        OUT_PATH = './snapshots/DGNet/seg_results_onnx/NC4K/',
        SNAP_PATH = './snapshots/DGNet/DGNet.onnx')