import os
import logging
from datetime import datetime
from tensorboardX import SummaryWriter

import jittor as jt
from jittor import nn

import eval.python.metrics as Measure
from jittor_lib.lib.DGNet import DGNet as Network
from jittor_lib.utils.dataset import get_loader, test_dataset

jt.flags.use_cuda = 1


def structure_loss(pred, mask):
    weit = (1 + (5 * jt.abs((nn.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask))))
    wbce = nn.binary_cross_entropy_with_logits(pred, mask)
    wbce = (((weit * wbce).sum(dim=2).sum(dim=2)) / weit.sum(dim=2).sum(dim=2))
    pred = jt.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=2).sum(dim=2)
    union = ((pred + mask) * weit).sum(dim=2).sum(dim=2)
    wiou = (1 - ((inter + 1) / ((union - inter) + 1)))
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, save_path, writer):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for (i, (images, gts, grads)) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            preds = model(images)
            loss_pred = structure_loss(preds[0], gts)
            loss_grad = nn.binary_cross_entropy_with_logits(preds[1], grads)

            loss = (loss_pred + loss_grad)

            optimizer.clip_grad_norm(opt.clip)
            optimizer.step(loss)

            step += 1
            epoch_step += 1
            
            loss_all += loss.data
            if (((i % 20) == 0) or (i == total_step) or (i == 1)):
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} loss_pred: {:.4f} loss_grad: {:0.4f}'.format(
                        datetime.now(), epoch, opt.epoch, i, total_step, loss[0], loss_pred[0], loss_grad[0]))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} loss_pred: {:.4f} loss_grad: {:0.4f}'.format(
                        epoch, opt.epoch, i, total_step, loss[0], loss_pred[0], loss_grad[0]))
                writer.add_scalars('Loss_Statistics',
                                   {'loss_pred': loss_pred[0].numpy(), 'loss_grad': loss_grad[0].numpy(), 'Loss_total': loss[0].numpy()},
                                   global_step=step)
                res = preds[0][0].clone()
                res = res.sigmoid().data.squeeze()
                res = ((res - res.min()) / ((res.max() - res.min()) + 1e-08))
                writer.add_image('Pred_final', res, step, dataformats='HW')
                res = preds[1][0].clone()
                res = res.sigmoid().data.squeeze()
                res = ((res - res.min()) / ((res.max() - res.min()) + 1e-08))
                writer.add_image('Pred_grad', res, step, dataformats='HW')
        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all[0]))
        writer.add_scalar('Loss-epoch', loss_all[0], global_step=epoch)
        if ((epoch % 50) == 0):
            model.save(save_path + 'Net_epoch_{}.pkl'.format(epoch))
    except KeyboardInterrupt:
        print('>>> Keyboard Interrupt: save model and exit.')
        if (not os.path.exists(save_path)):
            os.makedirs(save_path)
        model.save(save_path + 'Net_epoch_{}.pkl'.format(epoch))
        print('>>> Save checkpoints successfully!')
        raise


def val(val_loader, model, epoch, save_path, writer):
    global best_metric_dict, best_score, best_epoch
    FM = Measure.Fmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    metrics_dict = dict()
    model.eval()
    with jt.no_grad():
        for image, gt, name, _ in val_loader:
            c, h, w = gt.shape
            res = model(image)
            res = nn.upsample(res[0], size=(h, w), mode='bilinear')
            res = res.sigmoid().data.squeeze()
            res = ((res - res.min()) / ((res.max() - res.min()) + 1e-08))
            gt = gt.numpy().squeeze()
            FM.step(pred=res, gt=gt)
            SM.step(pred=res, gt=gt)
            EM.step(pred=res, gt=gt)
        metrics_dict.update(Sm=SM.get_results()['sm'])
        metrics_dict.update(mxFm=FM.get_results()['fm']['curve'].max().round(3))
        metrics_dict.update(mxEm=EM.get_results()['em']['curve'].max().round(3))
        cur_score = ((metrics_dict['Sm'] + metrics_dict['mxFm']) + metrics_dict['mxEm'])
        if (epoch == 1):
            best_score = cur_score
            print('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(epoch, metrics_dict['mxFm'],
                                                                             metrics_dict['Sm'], metrics_dict['mxEm']))
            logging.info('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(epoch, metrics_dict['mxFm'],
                                                                                    metrics_dict['Sm'],
                                                                                    metrics_dict['mxEm']))
        elif (cur_score > best_score):
            best_metric_dict = metrics_dict
            best_score = cur_score
            best_epoch = epoch
            model.save(save_path + 'Net_epoch_best.pkl')
            print('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
        else:
            print(
                '[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})\n[Best Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                    epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm'],
                    best_epoch, best_metric_dict['mxFm'], best_metric_dict['Sm'], best_metric_dict['mxEm']))
            logging.info(
                '[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})\n[Best Epoch:{}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                    epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm'],
                    best_epoch, best_metric_dict['mxFm'], best_metric_dict['Sm'], best_metric_dict['mxEm']))

            print('>>> not find the best epoch -> continue training ...')


if (__name__ == '__main__'):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=12, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--train_root', type=str, default='./dataset/TrainDataset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./dataset/TestDataset/CAMO/',
                        help='the test rgb images root')
    parser.add_argument('--gpu_id', type=str, default='1', help='train use gpu')
    parser.add_argument('--save_path', type=str, default='./jittor_lib/snapshot/DGNet_Jittor/',
                        help='the path to save model and log')
    opt = parser.parse_args()
    if (opt.gpu_id == '0'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        print('USE GPU 0')
    elif (opt.gpu_id == '1'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        print('USE GPU 1')
    elif (opt.gpu_id == '2'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        print('USE GPU 2')
    elif (opt.gpu_id == '3'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        print('USE GPU 3')

    model = Network(channel=64, arc='B4', M=[8, 8, 8], N=[4, 8, 16])
    # model = Network(channel=32, arc='B1', M=[8, 8, 8], N=[8, 16, 32])

    if (opt.load is not None):
        model.load_parameters(jt.load(opt.load))
        print('load model from ', opt.load)
    optimizer = jt.optim.Adam(model.parameters(), opt.lr)
    save_path = opt.save_path
    if (not os.path.exists(save_path)):
        os.makedirs(save_path)
    print('load data...')
    val_loader = test_dataset(image_root=(opt.val_root + 'Imgs/'), gt_root=(opt.val_root + 'GT/'),
                              testsize=opt.trainsize).set_attrs(batch_size=1, shuffle=False)
    train_loader = get_loader(image_root=(opt.train_root + 'Imgs/'), gt_root=(opt.train_root + 'GT/'),
                              grad_root=(opt.train_root + 'Gradient-Foreground/'), batchsize=opt.batchsize,
                              trainsize=opt.trainsize, num_workers=4).set_attrs(batch_size=opt.batchsize, shuffle=True,
                                                                                num_workers=4)

    total_step = len(train_loader)
    logging.basicConfig(filename=(save_path + 'log.log'), format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('>>> current mode: network-train/val')
    logging.info('>>> config: {}'.format(opt))
    print('>>> config: : {}'.format(opt))
    step = 0
    writer = SummaryWriter((save_path + 'summary'))
    best_score = 0
    best_epoch = 0
    cosine_schedule = jt.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-05)
    print('>>> start train...')
    for epoch in range(1, opt.epoch):
        cosine_schedule.step()
        
        train(train_loader, model, optimizer, epoch, save_path, writer)
        if (epoch > (opt.epoch // 2)):
            val(val_loader, model, epoch, save_path, writer)
