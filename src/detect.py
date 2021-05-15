import os
import cv2
import time
import torch
import numpy as np
import math
from model_training.model.yolo_fastest import YoloFastest
from model_training.utils.general import plot_one_box
from model_training._config import config_params
from model_training.train import config_logger


# 避免使用pytorch库函数，使用numpy库完成后处理
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
            pred_head = pred_head.numpy()[0]  # 转换为numpy
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


class Detect_YOLO():
    def __init__(self, device, model_path, config_params,logger):
        self.model = YoloFastest(config_params["io_params"]).to(device).eval()
        net_param = torch.load(model_path, map_location=device)
        self.model.load_state_dict(net_param)  # 导入模型参数
        self.logger = logger
        self.device = device

        self.class_names = config_params["io_params"]["class_names"]
        self.num_cls = config_params["io_params"]["num_cls"]
        self.nms_thres = config_params["io_params"]["nms_thre"]
        self.conf_thres = config_params["io_params"]["conf_thre"]
        self.input_shape = config_params["io_params"]["input_shape"]  # 网络的输入图像尺寸
        self.origin_img_shape = config_params["io_params"]["origin_img_shape"]  # 数据集原始图片尺寸
        self.post_process = YOLO_post_process(conf_thres=self.conf_thres, nms_thres=self.nms_thres,
                                              num_anchors=config_params["io_params"]["num_anchors"],
                                              anchors=config_params["io_params"]["anchors"],
                                              input_shape=self.input_shape, num_class=self.num_cls)
        self.colors = [[106, 90, 205], [199, 97, 20], [112, 128, 105]]

    def __pre_process(self, img_path):
        ori_img = cv2.imread(img_path)  # BGR格式读入

        if self.input_shape[2] == 1 and self.origin_img_shape[2] != 1:  # 网络要求单通道输入
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
        else:
            img = ori_img  # 深度拷贝，不会影响原图

        if self.input_shape[0:2] != self.origin_img_shape[0:2]:  # 输入图片大小调整至满足网络要求
            img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))

        if self.input_shape[2] == 1:
            img = np.expand_dims(img, -1)  # 对于灰度图，在最后多加一个维度，shape变为【h,w,1】

        img = img[:, :, ::-1].copy().transpose(2, 0, 1)  # （h,w,chanel）->(chanel,h,w)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float()
        img = (img - 128.0) / 255.0  # 归一化

        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 最外层加一个维度->[1,h,w,1]

        return img, ori_img

    def __adjust_coord(self, all_bbox_rects):
        scale_h = self.origin_img_shape[0]/self.input_shape[0] # 原始图像和网络输入图像的高度放缩倍数
        scale_w = self.origin_img_shape[1]/self.input_shape[1] # 原始图像和网络输入图像的宽度放缩倍数

        for i in range(len(all_bbox_rects)):
            all_bbox_rects[i][0] = round(all_bbox_rects[i][0] * scale_w)
            all_bbox_rects[i][2] = round(all_bbox_rects[i][2] * scale_w)
            all_bbox_rects[i][1] = round(all_bbox_rects[i][1] * scale_h)
            all_bbox_rects[i][3] = round(all_bbox_rects[i][3] * scale_h)

    def batch_detect(self, data_path, result_path):
        with torch.no_grad():
            img_list = os.listdir(data_path)
            num = len(img_list)  # 待检测图片总数
            avg_time = 0  # 记录检测平均用时
            for filename in img_list:
                img_path = os.path.join(data_path, filename)  # 每张图片的路径
                img, ori_img = self.__pre_process(img_path=img_path)  # 图片预处理，调整格式

                # 网路推理
                start_time = time.time()
                pred = self.model(img)
                time_mark = time.time()
                infer_time = float(time_mark - start_time) * 1000  # 推理时间

                # 后处理
                all_bbox_rects = self.post_process.decode_box(pred)  # 解码，获取所有有效边界框
                bbox_rects_class = [[] for i in range(self.num_cls)]
                for bbox in all_bbox_rects:
                    bbox_rects_class[bbox[-1]].append(bbox)  # 按类别分开存储
                all_bbox_rects.clear()
                for cls in range(self.num_cls):  # 每一类单独做NMS
                    if len(bbox_rects_class[cls]) == 0:
                        continue
                    def __get_conf(item):
                        return item[4]
                    bbox_rects_class[cls].sort(key=__get_conf, reverse=True)  # 根据置信度降序排列
                    results = self.post_process.non_maxium_supression(bbox_rects_class[cls])  # NMS
                    all_bbox_rects.extend(results)
                post_process_time = float(time.time() - time_mark) * 1000
                total_time = infer_time + post_process_time
                avg_time += total_time

                # 无检测结果下的日志记录
                if len(all_bbox_rects) == 0:
                    cv2.imwrite(os.path.join(result_path, 'result_'+filename), ori_img)  # 保存结果
                    self.logger.info("image_name:%s -> no targets, infer time:%.2fms, post_process time:%.2fms, total time:%.2fms" % (filename, infer_time, post_process_time, total_time))
                    continue

                # 坐标调整
                if self.input_shape[0:2] != self.origin_img_shape[0:2]:  # bbox坐标调整，从网络输入图片坐标系调整到实际图片坐标系
                    self.__adjust_coord(all_bbox_rects)

                # 画框
                for *xyxy, conf, cls_score, cls_pred in all_bbox_rects:
                    label = '%s %.2f' % (self.class_names[int(cls_pred)], conf*cls_score)  # conf*cls_score为类别目标置信度
                    plot_one_box(xyxy, ori_img, label=label, color=self.colors[int(cls_pred)], line_thickness=3)

                cv2.imwrite(os.path.join(result_path, 'result_'+filename), ori_img)  # 保存结果
                self.logger.info("image_name:%s -> detect finished, infer time:%.2fms, post_process time:%.2fms, total time:%.2fms" % (filename, infer_time, post_process_time, total_time))

            self.logger.info("detect avg_time: %.2fms" % (avg_time/num))


if __name__ == '__main__':
    logger = config_logger(log_dir='E:\Graduate_Design\YOLO-Fastest\\test_result\\256x320',
                              log_name='cpu-test.log', tensorboard=False)  # 加载日志模块

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备选择
    detect = Detect_YOLO(device, model_path="E:\Graduate_Design\YOLO-Fastest\models\pytorch\\256x320\YOLO-Fastest_epoch_28.pth",
                         config_params=config_params, logger=logger)
    detect.batch_detect(data_path="E:\Graduate_Design\YOLO-Fastest\\test_data",
                        result_path="E:\Graduate_Design\YOLO-Fastest\\test_result\\256x320")





