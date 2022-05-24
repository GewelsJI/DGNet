# -*- coding: utf-8 -*-
# @Time     : 2021/09/14
# @Author   : Johnson-Chou
# @Email    : johnson111788@gmail.com
# @FileName : builder.py

import numpy as np
import torch.nn as nn

from eval.metrics import Fmeasure, MAE, Smeasure, Emeasure, WeightedFmeasure, DICE, IoU

_TYPE = np.float64


class Metrics(object):

    def __init__(self, metrics, w_metrics):
        super(Metrics, self).__init__()

        if len(metrics) != len(w_metrics):
            raise Exception("Invalid Metrics Length: {}, {}".format(metrics, w_metrics))

        if 'mae' in metrics and len(metrics) > 1:
            raise Exception("Invalid Metrics Symbol: {}".format(metrics))
        else:
            self.metrics = metrics

        self.score_dict = dict()
        self.w_metrics = w_metrics

        self.Fmeasure = Fmeasure()
        self.MAE = MAE()
        self.Smeasure = Smeasure()
        self.Emeasure = Emeasure()
        self.WeightedFmeasure = WeightedFmeasure()
        self.DICE = DICE()
        self.IoU = IoU()

    def step(self, x, gt):

        for metric in self.metrics:
            if 'fm' == metric:
                fm = self.Fmeasure.step(x, gt).get_results()['fm']
                self.score_dict.update(adpFm=fm)
                self.score_dict.update(meanFm=fm['curve'].mean())
                self.score_dict.update(maxFm=fm['curve'].max())

            if 'mae' == metric:
                mae = self.MAE.step(x, gt).get_results()['mae']
                self.score_dict.update(MAE=mae)

            if 'sm' == metric:
                sm = self.Smeasure.step(x, gt).get_results()['sm']
                self.score_dict.update(Sm=sm)

            if 'em' == metric:
                em = self.Emeasure.step(x, gt).get_results()['em']
                self.score_dict.update(adpEm=em)
                self.score_dict.update(meanEm=em['curve'].mean())
                self.score_dict.update(maxEm=em['curve'].max())

            if 'wfm' == metric:
                wfm = self.WeightedFmeasure.step(x, gt).get_results()['wfm']
                self.score_dict.update(wFm=wfm)

            # if 'iou' == metric:
            #     iou = self.IoU.step(x, gt)
            #     if 'iou' in self.score_dict.keys():
            #         self.score_dict['iou'].append(iou)
            #     else:
            #         self.score_dict['iou'] = [iou]

            # if 'dice' == metric:
            #     dice = self.DICE.step(x, gt)
            #     if 'dice' in self.score_dict.keys():
            #         self.score_dict['dice'].append(dice)
            #     else:
            #         self.score_dict['dice'] = [dice]

    def get_results(self):
        metrics_dict = {}
        w_metrics_dict = {}
        for key, value in self.score_dict.items():
            metrics_dict[key] = np.mean(np.array(value, _TYPE))
            w_metrics_dict[key] = self.w_metrics[self.metrics.index(key)]

        metrics_score, metrics_print = None, None
        for metric_key, metric_value in metrics_dict.items():

            metrics_score = metric_value * w_metrics_dict[metric_key] if metrics_score is None \
                else metrics_score + metric_value * w_metrics_dict[metric_key]
            metrics_print = '{}: {:.04f}'.format(metric_key, metric_value) if metrics_print is None \
                else metrics_print + ', {}: {:.04f}'.format(metric_key, metric_value)

        metrics_print += ', Weighted Score: {:.04f}'.format(metrics_score)
        metrics_dict['Weighted_Total'] = metrics_score

        return metrics_dict, metrics_print, metrics_score
        




