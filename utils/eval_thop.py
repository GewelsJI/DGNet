import torch
import sys
print(sys.path)
sys.path.insert(0, '')

from lib.DGNet_V4_Res2Net import DGNet as NetworkRes2Net
from lib.DGNet_V4_ResNet import DGNet as NetworkResNet

model = NetworkResNet(channel=32, arc='34', group_list=[8, 8, 8], group_list_N=[8, 16, 32]).cuda()
# model = NetworkResNet(channel=64, arc='34', group_list=[8, 8, 8], group_list_N=[4, 8, 16]).cuda()

# model = NetworkResNet(channel=32, arc='50', group_list=[8, 8, 8], group_list_N=[8, 16, 32]).cuda()
# model = NetworkResNet(channel=64, arc='50', group_list=[8, 8, 8], group_list_N=[4, 8, 16]).cuda()

# model = NetworkRes2Net(channel=32, arc='B1', group_list=[8, 8, 8], group_list_N=[8, 16, 32]).cuda()
# model = NetworkRes2Net(channel=64, arc='B1', group_list=[8, 8, 8], group_list_N=[4, 8, 16]).cuda()

input = torch.randn(1, 3, 352, 352).cuda()


# macs, params = profile(net, inputs=(input, ))
#
# print(str(params/1024/1024)+'\t'+str(macs/1024/1024/1024))

from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  macs, params = get_model_complexity_info(model, (3, 352, 352), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)

print(params, macs)
