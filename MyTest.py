import os
import sys
import torch
import argparse
import numpy as np
import prettytable as pt
from scipy import misc  # NOTES: pip install scipy == 1.2.2 (prerequisite!)

import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from utils.dataset import test_dataset as EvalDataset
from lib.DGNet import DGNet as Network
import eval.metrics as Measure


class Logger(object):
    def __init__(self, fileN='Default.log'):
        self.terminal = sys.stdout
        self.log = open(fileN, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def evaluator(snap_path, val_root, gpu_id, trainsize=352, if_simply_metric=False, if_save_map=False):
    # define measures
    FM = Measure.Fmeasure()
    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()

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

    # model = Network(channel=32, arc='B1', group_list=[8, 8, 8], group_list_N=[4,8,16]).cuda()
    model = Network(channel=64, arc='B4', group_list=[8, 8, 8], group_list_N=[4,8,16]).cuda()

    # metric_fn = Metrics(metrics, w_metrics)
    val_loader = EvalDataset(image_root=val_root + 'Imgs/',
                            gt_root=val_root + 'GT/',
                            testsize=trainsize)

    model.load_state_dict(torch.load(snap_path))

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
            if if_save_map:
                misc.imsave(map_save_path + name, output)
                print('>>> prediction save at: {}'.format(map_save_path + name))

            FM.step(pred=output, gt=gt)
            WFM.step(pred=output, gt=gt)
            SM.step(pred=output, gt=gt)
            EM.step(pred=output, gt=gt)
            MAE.step(pred=output, gt=gt)

    fm = FM.get_results()['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']

    tb = pt.PrettyTable()
    tb.field_names = ["Method", "Dataset", "Smeasure", "wFmeasure", "MAE", "adpEm", "meanEm", "maxEm", "adpFm", "meanFm", "maxFm"]
    tb.add_row([snap_path.split('/')[-2], val_root.split('/')[-2], sm.round(4), wfm.round(4), mae.round(4), em['adp'].round(4), em['curve'].mean().round(4), em['curve'].max().round(4), fm['adp'].round(4), fm['curve'].mean().round(4), fm['curve'].max().round(4)])
    print(tb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snap_path', type=str, default='./snapshot/ExpV4_Small_Polyp_32BS/Net_epoch_best.pth', 
                        help='train use gpu')
    parser.add_argument('--gpu_id', type=str, default='1', 
                        help='train use gpu')
    parser.add_argument('--if_simply_metric', type=bool, default=False)
    parser.add_argument('--if_save_map', type=bool, default=True, help='train use gpu')
    opt = parser.parse_args()

    txt_save_path = './result/{}/'.format(opt.snap_path.split('/')[-2])
    os.makedirs(txt_save_path, exist_ok=True)
    
    sys.stdout = Logger(txt_save_path + 'evaluation_results.log')
    print('>>> configs:', opt)

    for data_name in ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']:
        if opt.if_save_map:
            map_save_path = txt_save_path + "res_map/{}/".format(data_name)
            os.makedirs(map_save_path, exist_ok=True)
        evaluator(
            snap_path=opt.snap_path,
            val_root='./dataset/TestDataset/'+data_name+'/',
            gpu_id=opt.gpu_id,
            trainsize=352,
            if_simply_metric=opt.if_simply_metric,
            if_save_map=opt.if_save_map)
