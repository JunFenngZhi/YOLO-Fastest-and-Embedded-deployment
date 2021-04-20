import numpy as np
import cv2
from rknn.api import RKNN
import os
import torch
from yolo_fastest import YoloFastest
from _config import config_params


# 将pytorch参数模型导出，保存为torchscript格式
def export_pytorch_model(model_path, save_path, input_tensor_shape):
    net = YoloFastest(io_params=config_params["io_params"]).eval()
    net_param = torch.load(model_path, map_location="cpu")
    net.load_state_dict(net_param)  # 导入模型参数

    trace_model = torch.jit.trace(net, torch.Tensor(input_tensor_shape[0],input_tensor_shape[1],input_tensor_shape[2],input_tensor_shape[3]))
    trace_model.save(save_path)

# 将pytorch模型转化为rknn格式，并保存
def pytoch_to_rknn(model_path, target_path, input_tensor_shape):

    print('--> export pytorch model')
    dir_path, filename = os.path.split(os.path.abspath(__file__))
    temp_path = os.path.join(dir_path, 'temp.pt')
    if os.path.exists(temp_path) is False:
        export_pytorch_model(model_path, temp_path, input_tensor_shape)

    rknn = RKNN()  # 构建RKNN模型

    print('--> config RKNN')
    rknn.config(channel_mean_value='128.0 255.0', reorder_channel='0 1 2',  # 输入图片为单通道，减128再除255
                target_platform=['rk3399pro'])

    print('--> Loading model')
    input_size_list = [list(input_tensor_shape[1:])]
    ret = rknn.load_pytorch(model=temp_path, input_size_list=input_size_list)
    if ret != 0:
        print('Load pytorch model failed!')
        exit(ret)

    print('--> Building model')
    ret = rknn.build(do_quantization=False)  # 不做量化
    if ret != 0:
        print('Build pytorch failed!')
        exit(ret)

    print('--> Export RKNN model')
    ret = rknn.export_rknn(target_path)
    if ret != 0:
        print('Export resnet_18.rknn failed!')
        exit(ret)
    print("Finish!")


if __name__ == '__main__':
    pytoch_to_rknn(model_path='/home/toybrick/RKNN_project/pytorch_model/YOLO-Fastest_epoch_29.pth',
                   target_path='/home/toybrick/RKNN_project/RKNN_model/YOLO-Fastest_epoch_29.rknn',
                   input_tensor_shape=config_params["io_params"]["input_tensor_shape"])


