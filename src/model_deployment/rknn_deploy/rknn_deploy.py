import numpy as np
import cv2
from rknn.api import RKNN
import os
import time
import torch
import math
from yolo_fastest import YoloFastest
from _config import config_params
from general import plot_one_box


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


# 输入图片预处理，调整大小
def pre_process(img_path, input_shape, origin_img_shape):
    ori_img = cv2.imread(img_path)  # BGR格式读入

    if input_shape[2] == 1 and origin_img_shape[2] != 1:  # 网络要求单通道输入
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    else:
        img = ori_img  # 深度拷贝，不会影响原图

    if input_shape[0:2] != origin_img_shape[0:2]:  # 输入图片大小调整至满足网络要求
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    return img, ori_img


# 调整bbox坐标至原始图片坐标系
def adjust_coord(origin_img_shape, input_shape, all_bbox_rects):
    scale_h = origin_img_shape[0]/input_shape[0] # 原始图像和网络输入图像的高度放缩倍数
    scale_w = origin_img_shape[1]/input_shape[1] # 原始图像和网络输入图像的宽度放缩倍数

    for i in range(len(all_bbox_rects)):
        all_bbox_rects[i][0] = round(all_bbox_rects[i][0] * scale_w)
        all_bbox_rects[i][2] = round(all_bbox_rects[i][2] * scale_w)
        all_bbox_rects[i][1] = round(all_bbox_rects[i][1] * scale_h)
        all_bbox_rects[i][3] = round(all_bbox_rects[i][3] * scale_h)


# 后处理模块
'''根据pred的通道分布，进行调整。'''
class YOLO_post_process:
    def __init__(self, conf_thres, nms_thres, num_anchors, num_class, anchors, input_shape):
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.num_anchors = num_anchors
        self.bbox_attrs = 5 + num_class  # 每个boxes对应的预测值数量
        self.anchors = anchors
        self.input_shape = input_shape

    @staticmethod
    def __sigmoid(x):
        return 1. / (1. + math.exp(-x))

    @staticmethod
    def __cal_iou(box_1, box_2):
        inter_area = 0

        inter_w = min(box_1[2], box_2[2]) - max(box_1[0], box_2[0])
        inter_h = min(box_1[3], box_2[3]) - max(box_1[1], box_2[1])
        if inter_w > 0 and inter_h > 0:
            inter_area = inter_h * inter_w   # 交集

        union_area = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1]) +\
        (box_2[2] - box_2[0]) * (box_2[3] - box_2[1]) - inter_area   # 并集

        return inter_area/union_area

    def decode_box(self, pred):
        all_bbox_rects = []
        for head, pred_head in enumerate(pred):
            pred_head = pred_head[0]
            in_w = pred_head.shape[2]
            in_h = pred_head.shape[1]
            scale_h = self.input_shape[0]/in_h   # 高度上的输出特征图缩小倍数(相对于网络输入图片坐标系而言)
            scale_w = self.input_shape[1]/in_w   # 宽度上的输出特征图缩小倍数
            anchors = self.anchors[head]

            pred_head = pred_head.reshape((self.num_anchors, self.bbox_attrs, in_h, in_w))
            for pp in range(self.num_anchors):
                for i in range(in_h):
                    for j in range(in_w):
                        conf = self.__sigmoid(pred_head[pp, 4, i, j])
                        if conf > self.conf_thres:  # 大于置信度阈值，保留
                            cls_index = np.argmax(pred_head[pp, 5:, i, j])
                            cls_score = self.__sigmoid(np.max(pred_head[pp, 5:, i, j]))
                            x = (j + self.__sigmoid(pred_head[pp, 0, i, j]))*scale_w
                            y = (i + self.__sigmoid(pred_head[pp, 1, i, j]))*scale_h
                            w = math.exp(pred_head[pp, 2, i, j]) * anchors[pp][0]
                            h = math.exp(pred_head[pp, 3, i, j]) * anchors[pp][1]
                            all_bbox_rects.append(
                                [round(x - w / 2), round(y - h / 2), round(x + w / 2), round(y + h / 2), conf, cls_score, cls_index])
        return all_bbox_rects

    def non_maxium_supression(self, bbox_list):
        results = []
        while len(bbox_list) != 0:
            results.append(bbox_list[0])  # 选择conf最大的作为检测结果
            if len(bbox_list) == 1:
                break
            bbox_list.pop(0)  # 去掉已选择的bbox
            i = 0
            while i <= len(bbox_list)-1:
                iou = self.__cal_iou(bbox_list[i], results[-1])
                if iou > self.nms_thres:
                    bbox_list.pop(i)  # NMS去除冗余bbox
                else:
                    i += 1

        return results





if __name__ == '__main__':
    pytorch_model_path = '/home/toybrick/RKNN_project/pytorch_model/256x320/YOLO-Fastest_epoch_28.pth'
    rknn_model_path = '/home/toybrick/RKNN_project/RKNN_model/256x320/YOLO-Fastest_epoch_28.rknn'
    data_path = '/home/toybrick/RKNN_project/test_data'
    result_path = '/home/toybrick/RKNN_project/test_result/256x320'

    input_shape = config_params["io_params"]["input_shape"]
    origin_img_shape = config_params["io_params"]["origin_img_shape"]
    num_cls = config_params["io_params"]["num_cls"]
    class_names = config_params["io_params"]["class_names"]
    colors = [[106, 90, 205], [199, 97, 20], [112, 128, 105]]

    rknn = RKNN()
    post_process = YOLO_post_process(conf_thres=config_params["io_params"]["conf_thre"],
                                      nms_thres=config_params["io_params"]["nms_thre"],
                                      num_anchors=config_params["io_params"]["num_anchors"],
                                      anchors=config_params["io_params"]["anchors"],
                                      input_shape=input_shape, num_class=num_cls)

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

    print('--> Begin...\n')
    img_list = os.listdir(data_path)
    avg_time = 0
    num = len(img_list)
    for file_name in img_list:
        img_path = os.path.join(data_path, file_name)  # 每张图片的路径
        print(img_path)
        img, ori_img = pre_process(img_path=img_path, input_shape=input_shape, origin_img_shape=origin_img_shape)

        # 网络推理
        start_time = time.time()
        pred = rknn.inference(inputs=[img], data_format='nhwc')
        time_mark = time.time()
        infer_time = float(time_mark-start_time)*1000   # NPU推理时间

        # 后处理
        all_bbox_rects = post_process.decode_box(pred)  # 解码，获取所有有效边界框
        bbox_rects_class = [[] for i in range(num_cls)]
        for bbox in all_bbox_rects:
            bbox_rects_class[bbox[-1]].append(bbox)  # 按类别分开存储
        all_bbox_rects.clear()
        for cls in range(num_cls):  # 每一类单独做NMS
            if len(bbox_rects_class[cls]) == 0:
                continue

            def __get_conf(item):
                return item[4]

            bbox_rects_class[cls].sort(key=__get_conf, reverse=True)  # 根据置信度降序排列
            results = post_process.non_maxium_supression(bbox_rects_class[cls])  # NMS
            all_bbox_rects.extend(results)
        post_process_time = float(time.time() - time_mark) * 1000
        total_time = infer_time + post_process_time
        avg_time += total_time

        # 无检测结果下的日志记录
        if len(all_bbox_rects) == 0:
            cv2.imwrite(os.path.join(result_path, 'result_' + file_name), ori_img)  # 保存结果
            print("image_name:{} -> no targets, infer time:{:.2f}ms, post_process time:{:.2f}ms, total time:{:.2f}ms".format(
                       file_name, infer_time, post_process_time, total_time))
            continue

        # 坐标调整
        if input_shape[0:2] != origin_img_shape[0:2]:  # bbox坐标调整，从网络输入图片坐标系调整到实际图片坐标系
            print("adjust")
            adjust_coord(origin_img_shape, input_shape, all_bbox_rects)

        # 画框
        for *xyxy, conf, cls_score, cls_pred in all_bbox_rects:
            label = '%s %.2f' % (class_names[int(cls_pred)], conf * cls_score)  # conf*cls_score为类别目标置信度
            plot_one_box(xyxy, ori_img, label=label, color=colors[int(cls_pred)], line_thickness=3)

        cv2.imwrite(os.path.join(result_path, 'result_' + file_name), ori_img)  # 保存结果
        print("image_name:{} -> detect finished, infer time:{:.2f}ms, post_process time:{:.2f}ms, total time:{:.2f}ms".format(
                file_name, infer_time, post_process_time, total_time))
    print("avg_time:{:.2f}ms".format(avg_time/num))

rknn.release()


















