import os
import torch
import argparse
import numpy as np
from scipy import misc  # NOTES: pip install scipy == 1.2.2 (prerequisite!)

import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from utils.data_val import test_dataset as EvalDataset
from lib.DGNet_V4 import DGNet as Network


def evaluator(val_root, trainsize=352):

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')
    cudnn.benchmark = True

    model = Network(channel=64, arc='B4', group_list=[8, 8, 8], group_list_N=[4, 8, 16]).cuda()

    val_loader = EvalDataset(image_root=val_root + 'Imgs/',
                             gt_root=val_root + 'GT/',
                             testsize=trainsize)

    model.eval()
    with torch.no_grad():
        for i in range(val_loader.size):
            image, gt, name, _ = val_loader.load_data()
            gt = np.asarray(gt, np.float32)
            # gt /= (gt.max() + 1e-8)

            image = image.cuda()

            output = model(image)
            output = F.upsample(output[0], size=gt.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)

            misc.imsave(map_save_path + name, output)
            print('>>> prediction save at: {}'.format(map_save_path + name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='1',
                        help='train use gpu')
    opt = parser.parse_args()

    txt_save_path = './result/{}/'.format(opt.snap_path.split('/')[-2])
    os.makedirs(txt_save_path, exist_ok=True)

    for data_name in ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']:

        map_save_path = txt_save_path + "res_map/{}/".format(data_name)
        os.makedirs(map_save_path, exist_ok=True)

        evaluator(
            val_root='./dataset/TestDataset/' + data_name + '/',
            trainsize=352
        )