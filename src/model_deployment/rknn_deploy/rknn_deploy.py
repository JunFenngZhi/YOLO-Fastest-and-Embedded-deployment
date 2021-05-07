import numpy as np
import cv2
from rknn.api import RKNN
import os
import time
import torch
from model_training.model.yolo_fastest import YoloFastest
from model_training._config import config_params
from model_training.loss.yolo_loss import YOLOLossV3
from model_training.utils.general import non_max_suppression,plot_one_box


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


# 后处理模块
class PostProcessing():
    def __init__(self, device, config_params):
        pred_branch = len(config_params["io_params"]["strides"])  # 预测分支
        self.model_loss = []  # 使用loss类中的函数，对预测结果进行坐标还原
        for i in range(pred_branch):
            self.model_loss.append(YOLOLossV3(anchors=config_params["io_params"]["anchors"][i],
                                              num_classes=config_params["io_params"]["num_cls"],
                                              img_size=config_params["io_params"]["input_size"], device=device))
        self.class_names = config_params["io_params"]["class_names"]
        self.config_params = config_params
        self.nms_thres = config_params["io_params"]["nms_thre"]
        self.conf_thres = config_params["io_params"]["conf_thre"]
        self.colors = [[106, 90, 205], [199, 97, 20], [112, 128, 105]]

    def process(self, pred, ori_img, file_name, result_path):
        output_list = []
        for i, item_pred in enumerate(pred):  # 获取不同尺度的预测结果
            output_list.append(self.model_loss[i](torch.from_numpy(item_pred)))  # 返回的是predict出来的所有bounding box（已反向还原）
        output = torch.cat(output_list, 1)  # 不同尺度的边界框合在一起
        output = non_max_suppression(output, config_params["io_params"]["num_cls"],
                                     conf_thres=self.conf_thres, nms_thres=self.nms_thres)

        output = output[0]  # 一次只处理一张图，所以只取第一个
        post_process_time = float(time.time()-time_mark)*1000  # 后处理用时
        if output is None:
            print("image_name:{} -> no targets, infer time:{:.2f}ms, post_process time:{:.2f}ms, total time:{:.2f}ms".format(file_name, infer_time, post_process_time,
                                                                infer_time + post_process_time))
            return

        # 画框
        for *xyxy, conf, cls_score, cls_pred in reversed(output):
            label = '%s %.2f' % (self.class_names[int(cls_pred)], conf * cls_score)  # conf*cls_score为类别目标置信度
            plot_one_box(xyxy, ori_img, label=label, color=self.colors[int(cls_pred)], line_thickness=3)

        if os.path.exists(result_path) is False :
            os.makedirs(result_path)

        cv2.imwrite(os.path.join(result_path, 'result_' + file_name), ori_img)  # 保存结果
        print("image_name:{} -> detect finished, infer time:{:.2f}ms, post_process time:{:.2f}ms, total time:{:.2f}ms".format(file_name, infer_time, post_process_time,
                                                              infer_time + post_process_time))
        return





if __name__ == '__main__':
    pytorch_model_path = '/home/toybrick/RKNN_project/pytorch_model/YOLO-Fastest_epoch_29.pth'
    rknn_model_path = '/home/toybrick/RKNN_project/RKNN_model/YOLO-Fastest_epoch_29.rknn'
    data_path = '/home/toybrick/RKNN_project/test_data'
    result_path = '/home/toybrick/RKNN_project/test_result'

    rknn = RKNN()
    post = PostProcessing(device=torch.device("cpu"), config_params=config_params)

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

    print('--> Init runtime environment\n')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)

    img_list = os.listdir(data_path)
    for file_name in img_list:
        img_path = os.path.join(data_path, file_name)  # 每张图片的路径
        ori_img = cv2.imread(filename=img_path)   # BGR格式
        img_grey = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)   # 转灰度图
        
        start_time = time.time()
        outputs = rknn.inference(inputs=[img_grey], data_format='nhwc')
        time_mark = time.time()
        infer_time = float(time_mark-start_time)*1000   # NPU推理时间
        post.process(pred=outputs, ori_img=ori_img, file_name=file_name, result_path=result_path)

    rknn.release()


















