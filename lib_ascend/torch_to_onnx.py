import sys
import torch
import torch.onnx
# import torchvision.models as models
from lib.DGNet import DGNet


def pth2onnx(model_type, input_file, output_file):
    # 对于efficient模型的初始化
    if model_type == 'DGNet':
        model = DGNet(channel=64, arc='EfficientNet-B4', M=[8, 8, 8], N=[4, 8, 16])
        model.context_encoder.set_swish(False)
    elif model_type == 'DGNet-S':
        model = DGNet(channel=32, arc='EfficientNet-B1', M=[8, 8, 8], N=[8, 16, 32])
        model.context_encoder.set_swish(False)
    elif model_type == 'DGNet-PVTv2-B0':
        model = DGNet(channel=32, arc='PVTv2-B0', M=[8, 8, 8], N=[8, 16, 32])
    elif model_type == 'DGNet-PVTv2-B1':
        model = DGNet(channel=64, arc='PVTv2-B1', M=[8, 8, 8], N=[4, 8, 16])   
    elif model_type == 'DGNet-PVTv2-B2':
        model = DGNet(channel=64, arc='PVTv2-B2', M=[8, 8, 8], N=[4, 8, 16])
    elif model_type == 'DGNet-PVTv2-B3':
        model = DGNet(channel=64, arc='PVTv2-B3', M=[8, 8, 8], N=[4, 8, 16])   
    else:
        raise Exception("Invalid Model Symbol: {}".format(model_type))

    # 对于 pvtv2 模型的初始化
    # model = DGNet(channel=64, arc='PVTv2-B1', M=[8, 8, 8], N=[4, 8, 16])
    model.load_state_dict(torch.load(input_file, map_location=torch.device('cpu')))
    
    # 调整模型为eval mode
    model.eval()  
    # 输入节点名
    input_names = ["image"]  
    # 输出节点名
    output_names = ["pred"]  
    dynamic_axes = {'image': {0: '-1'}, 'pred': {0: '-1'}} 
    dummy_input = torch.randn(1, 3, 352, 352)
    
    print('--> start transformation (from *.pth to *.onnx)')
    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, opset_version=11, verbose=False) 


if __name__ == "__main__":
    pth2onnx(
        model_type='DGNet-PVTv2-B3', 
        input_file='./snapshots/DGNet-PVTv2-B3/DGNet-PVTv2-B3.pth', 
        output_file='./snapshots/DGNet-PVTv2-B3/DGNet-PVTv2-B3.onnx')


"""
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --framework=5 --model=DGNet-PVTv2-B3.onnx --output=DGNet-PVTv2-B3 --input_shape="image:1,3,352,352" --log=debug --soc_version=Ascend310 > atc.log

conda activate py392
. ~/mindx_dir/mxVision/set_env.sh
"""