# common libraries
import os
import torch
import argparse
# torch libraries
import torch.backends.cudnn as cudnn
# customized libraries
from lib.DGNet import DGNet as Network


def evaluator(trainsize=352):

    if opt.gpu:
        # set the device for training
        if opt.gpu_id == '0':
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print('USE GPU 0')
        elif opt.gpu_id == '1':
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            print('USE GPU 1')
        cudnn.benchmark = True

    print('--- Code Info ---\n'
          'Deep Gradient Learning for Efficient Concealed Object Detection (Supplementary Material)\n'
          'Anonymous CVPR 2022 submission (Paper ID 2241)\n'
          '\n--- Note ---\n'
          'tensor format: (BatchSize, Channel, Weight, Height)')

    # 1. define input tensor
    image = torch.randn(1, 3, trainsize, trainsize)
    image = image.cuda() if opt.gpu else image

    # 2. define our proposed FSNet
    model = Network(channel=64, arc='B4', M=[8, 8, 8], N=[4, 8, 16])
    model = model.cuda() if opt.gpu else model

    # 3. forward
    output = model(image)

    # 4. finish
    print('\n--- Finish Inference ---\n'
          'Output Size:\n'
          '\tP^C ({})\n'
          '\tP^G ({})'.format(output[0].shape, output[1].shape))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--gpu_id', type=str, default='0')
    opt = parser.parse_args()

    evaluator(trainsize=352)