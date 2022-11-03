import torch
import time
import numpy as np
import jittor as jt
from ptflops import get_model_complexity_info

from lib_jittor.lib.DGNet import DGNet as jt_Network
from lib_pytorch.lib.DGNet import DGNet as py_Network

jt.flags.use_cuda = 1

bs = 16
test_img = np.random.random((bs,3,352,352)).astype('float32')

pytorch_test_img = torch.Tensor(test_img).cuda()
jittor_test_img = jt.array(test_img)

turns = 100

pytorch_model = py_Network(channel=32, arc='EfficientNet-B1', M=[8, 8, 8], N=[8, 16, 32]).cuda()
# pytorch_model = py_Network(channel=64, arc='EfficientNet-B4', M=[8, 8, 8], N=[4, 8, 16]).cuda()
# pytorch_model = py_Network(channel=32, arc='PVTv2-B0', M=[8, 8, 8], N=[8, 16, 32]).cuda()
# pytorch_model = py_Network(channel=64, arc='PVTv2-B1', M=[8, 8, 8], N=[4, 8, 16]).cuda()   
# pytorch_model = py_Network(channel=64, arc='PVTv2-B2', M=[8, 8, 8], N=[4, 8, 16]).cuda()
# pytorch_model = py_Network(channel=64, arc='PVTv2-B3', M=[8, 8, 8], N=[4, 8, 16]).cuda()
# pytorch_model = py_Network(channel=64, arc='PVTv2-B4', M=[8, 8, 8], N=[4, 8, 16]).cuda()
jittor_model = jt_Network(channel=64, arc='B4', M=[8, 8, 8], N=[4, 8, 16])

pytorch_model.eval()
jittor_model.eval()

jittor_model.load_parameters(pytorch_model.state_dict())

macs, params = get_model_complexity_info(pytorch_model, (3, 352, 352), as_strings=True,
                                            print_per_layer_stat=False, verbose=True)
print(f"- Pytorch model parameters: {params}, MACs: {macs}")

for i in range(10):
    pytorch_result = pytorch_model(pytorch_test_img)
torch.cuda.synchronize()
sta = time.time()
for i in range(turns):
    pytorch_result = pytorch_model(pytorch_test_img)
torch.cuda.synchronize()
end = time.time()
tc_time = round((end - sta) / turns, 5)
tc_fps = round(bs * turns / (end - sta),0)
print(f"- Pytorch forward average time cost: {tc_time}, Batch Size: {bs}, FPS: {tc_fps}")


for i in range(10):
    jittor_result = jittor_model(jittor_test_img)
    jittor_result[0][0].sync()
jt.sync_all(True)
sta = time.time()
for i in range(turns):
    jittor_result = jittor_model(jittor_test_img)
    jittor_result[0][0].sync()
jt.sync_all(True)
end = time.time()
jt_time = round((time.time() - sta) / turns, 5)
jt_fps = round(bs * turns / (end - sta),0)
print(f"- Jittor forward average time cost: {jt_time}, Batch Size: {bs}, FPS: {jt_fps}")