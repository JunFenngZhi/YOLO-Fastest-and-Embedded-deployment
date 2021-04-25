import numpy as np
import cv2
from rknn.api import RKNN
import os
import torch
from yolo_fastest import YoloFastest
from _config import config_params
from loss.yolo_loss import YOLOLossV3


# 将pytorch参数模型导出，保存为torchscript格式
def export_pytorch_model(model_path, save_path, input_tensor_shape):
    net = YoloFastest(io_params=config_params["io_params"]).eval()
    net_param = torch.load(model_path, map_location="cpu")
    net.load_state_dict(net_param)  # 导入模型参数

    trace_model = torch.jit.trace(net, torch.Tensor(input_tensor_shape[0],input_tensor_shape[1],input_tensor_shape[2],input_tensor_shape[3]))
    trace_model.save(save_path)

# 将pytorch模型转化为rknn格式，并保存
def pytoch_to_rknn(model_path, target_path, input_tensor_shape, rknn):
    if os.path.exists(target_path):
        print("---> Model is already transfered into rknn format")
        return

    print('--> Export pytorch model')
    dir_path, filename = os.path.split(os.path.abspath(__file__))
    temp_path = os.path.join(dir_path, 'temp.pt')
    if os.path.exists(temp_path) is False:
        export_pytorch_model(model_path, temp_path, input_tensor_shape)

    print('--> Loading pytorch model')
    input_size_list = [list(input_tensor_shape[1:])]
    ret = rknn.load_pytorch(model=temp_path, input_size_list=input_size_list)
    if ret != 0:
        print('Load pytorch model failed!')
        exit(ret)

    print('--> Building rknn model')
    ret = rknn.build(do_quantization=False)  # 不做量化
    if ret != 0:
        print('Build rknn failed!')
        exit(ret)

    print('--> Export rknn model')
    ret = rknn.export_rknn(target_path)
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)
    print("Done!")



if __name__ == '__main__':
    pytorch_model_path = '/home/toybrick/RKNN_project/pytorch_model/YOLO-Fastest_epoch_29.pth'
    rknn_model_path = '/home/toybrick/RKNN_project/RKNN_model/YOLO-Fastest_epoch_29.rknn'

    rknn = RKNN()
    print('--> Config rknn')
    rknn.config(channel_mean_value='128.0 255.0', reorder_channel='0 1 2',  # 输入图片为单通道。预处理操作为各像素减128再除255
                target_platform=['rk3399pro'])

    # 转换模型格式
    pytoch_to_rknn(model_path=pytorch_model_path, rknn=rknn, target_path=rknn_model_path,
                   input_tensor_shape=config_params["io_params"]["input_tensor_shape"])

    print('--> Loading rknn model')
    ret = rknn.load_rknn(rknn_model_path)
    if ret != 0:
        print('Loading rknn model failed')
        exit(ret)

    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk3399pro')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)

    ori_img = cv2.imread(filename='')   # BGR格式
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)   # 转灰度图

    outputs = rknn.inference(inputs=[img], data_format='nhwc')
    '''查看outputs输出的结构和类型'''
    '''
        输出转为torch tensor类型。 调用yolo_loss进行边界框的恢复。然后进行NMS。根据结果是否存在进行处理，画框，保存结果。
        抽象一个yolo_detect的类出来，只负责后处理模块，对output进行处理。
    '''














