import os
import argparse
import imageio

import torch
import jittor as jt
from jittor import nn

from utils.dataset import test_dataset as EvalDataset
from lib.DGNet import DGNet as Network

jt.flags.use_cuda = 0


def evaluator(model, val_root, map_save_path, trainsize=352):
    val_loader = EvalDataset(image_root=val_root + 'Imgs/',
                             gt_root=val_root + 'GT/',
                             testsize=trainsize).set_attrs(batch_size=1, shuffle=False)

    model.eval()
    with jt.no_grad():
        for image, gt, name, _ in val_loader:
            c, h, w = gt.shape

            output = model(image)
            output = nn.upsample(output[0], size=(h, w), mode='bilinear', align_corners=False)
            output = output.sigmoid().data.squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)
            print(name)
            imageio.imwrite(map_save_path + name[0], output)
            print('>>> prediction save at: {}'.format(map_save_path + name[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snap_path', type=str, default='./snapshot/DGNet-S/Net_epoch_best.pth',
                        help='train use gpu')
    parser.add_argument('--gpu_id', type=str, default='1',
                        help='train use gpu')
    opt = parser.parse_args()

    txt_save_path = './result/{}/'.format(opt.snap_path.split('/')[-2])
    os.makedirs(txt_save_path, exist_ok=True)

    print('>>> configs:', opt)

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

    model = Network(channel=64, arc='B4', M=[8, 8, 8], N=[4, 8, 16])
    model.load(opt.snap_path)
    model.eval()

    for data_name in ['CAMO', 'COD10K', 'NC4K']:
        map_save_path = txt_save_path + "res_map/{}/".format(data_name)
        os.makedirs(map_save_path, exist_ok=True)
        evaluator(
            model=model,
            val_root='./dataset/TestDataset/' + data_name + '/',
            map_save_path=map_save_path,
            trainsize=352)
