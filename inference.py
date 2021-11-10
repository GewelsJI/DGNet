# common libraries
import os
import torch
import argparse
import numpy as np
from scipy import misc
# torch libraries
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# customized libraries
from lib.DGNet import DGNet as Network
from utils.dataset import test_dataset as EvalDataset


def evaluator(val_root, trainsize=352):

    if opt.gpu:
        # set the device for training
        if opt.gpu_id == '0':
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print('USE GPU 0')
        elif opt.gpu_id == '1':
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            print('USE GPU 1')
        cudnn.benchmark = True

    # 1. define input tensor
    val_loader = EvalDataset(image_root=val_root + 'Imgs/',
                             gt_root=val_root + 'GT/',
                             testsize=trainsize)

    # 2. define our proposed FSNet
    model = Network(channel=64, arc='B4', M=[8, 8, 8], N=[4, 8, 16])
    model = model.cuda() if opt.gpu else model
    model.load_state_dict(torch.load(opt.snap_path))

    for i in range(val_loader.size):
        image, gt, name, _ = val_loader.load_data()
        gt = np.asarray(gt, np.float32)

        image = image.cuda() if opt.gpu else image

        # 3. forward
        output = model(image)
        output = F.upsample(output[0], size=gt.shape, mode='bilinear', align_corners=False)
        output = output.sigmoid().data.cpu().numpy().squeeze()
        output = (output - output.min()) / (output.max() - output.min() + 1e-8)

        misc.imsave(opt.save_path + name, output)
        print('>>> prediction save at: {}'.format(opt.save_path + name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--snap_path', type=str, default='./snapshot/DGNet/Net_epoch_best.pth')
    parser.add_argument('--save_path', type=str, default='./result/')
    opt = parser.parse_args()

    print('--- Code Info ---\n'
          'Deep Gradient Learning for Efficient Concealed Object Detection (Supplementary Material)\n'
          'Anonymous CVPR 2022 submission (Paper ID 2241)\n'
          '\n--- Note ---\n'
          'tensor format: (BatchSize, Channel, Weight, Height)')

    os.makedirs(opt.save_path, exist_ok=True)
    evaluator(val_root='./dataset/TestDataset/',
              trainsize=352)