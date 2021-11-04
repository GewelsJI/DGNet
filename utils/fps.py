import numpy as np
import torch
import time

def computeTime(model, device='cuda'):
    inputs = torch.randn(1, 3, 352, 352)
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    time_spent = []
    for idx in range(100):
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if idx > 10:
            time_spent.append(time.time() - start_time)
    print('Avg execution time (ms): %.4f, FPS:%d'%(np.mean(time_spent),1*1//np.mean(time_spent)))
    return 1*1//np.mean(time_spent)

if __name__=="__main__":

    torch.backends.cudnn.benchmark = True
    from lib.DGNet_V4 import DGNet as Network
    from lib.DGNet_V4_Res2Net import DGNet as NetworkRes2Net
    from lib.DGNet_V4_ResNet import DGNet as NetworkResNet

    # model = NetworkResNet(channel=32, arc='34', group_list=[8, 8, 8], group_list_N=[8, 16, 32]).cuda()
    model = NetworkResNet(channel=64, arc='34', group_list=[8, 8, 8], group_list_N=[4, 8, 16]).cuda()

    # model = NetworkResNet(channel=32, arc='50', group_list=[8, 8, 8], group_list_N=[8, 16, 32]).cuda()
    # model = NetworkResNet(channel=64, arc='50', group_list=[8, 8, 8], group_list_N=[4, 8, 16]).cuda()

    # model = NetworkRes2Net(channel=32, arc='B1', group_list=[8, 8, 8], group_list_N=[8, 16, 32]).cuda()
    # model = NetworkRes2Net(channel=64, arc='B1', group_list=[8, 8, 8], group_list_N=[4, 8, 16]).cuda()

    # model = Network(channel=64, arc='B4', group_list=[8, 8, 8], group_list_N=[4,8,16]).cuda()

    computeTime(model)
