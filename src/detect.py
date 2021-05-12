import os
import cv2
import time
import torch
import numpy as np
from model_training.model.yolo_fastest import YoloFastest
from model_training.loss.yolo_loss import YOLOLossV3
from model_training.utils.general import non_max_suppression, scale_coords, plot_one_box
from model_training._config import config_params
from model_training.train import config_logger


class Detect_YOLO():
    def __init__(self, device, model_path, config_params,logger):
        self.model = YoloFastest(config_params["io_params"]).to(device).eval()
        net_param = torch.load(model_path, map_location=device)
        self.model.load_state_dict(net_param)  # 导入模型参数

        pred_branch = len(config_params["io_params"]["strides"])  # 预测分支
        self.model_loss = []  # 使用loss类中的函数，对预测结果进行坐标还原
        for i in range(pred_branch):
            self.model_loss.append(YOLOLossV3(anchors=config_params["io_params"]["anchors"][i],
                                         num_classes=config_params["io_params"]["num_cls"],
                                         img_size=config_params["io_params"]["input_size"], device=device))
        self.class_names = config_params["io_params"]["class_names"]
        self.device = device
        self.config_params = config_params
        self.nms_thres = config_params["io_params"]["nms_thre"]
        self.conf_thres = config_params["io_params"]["conf_thre"]
        self.target_shape = config_params["io_params"]["input_size"][0:2]  # 网络输入图像的目标尺寸
        self.logger = logger
        self.colors = [[106, 90, 205], [199, 97, 20], [112, 128, 105]]

    def batch_detect(self, root_path, result_path):
        with torch.no_grad():
            img_list = os.listdir(root_path)
            num = len(img_list)  # 待检测图片总数
            avg_time = 0  # 记录检测平均用时
            for filename in img_list:
                img_path = os.path.join(root_path, filename)  # 每张图片的路径
                img_origin = cv2.imread(img_path)
                img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)

                # resize to target shape
                if img_origin.shape[0:2] == tuple(self.target_shape):
                    img = img_gray  # 假如shape一样可以跳过
                else:
                    img = self.__resize_img(img_gray, new_shape=self.target_shape)

                # pre-processing
                img = np.expand_dims(img, -1)  # 在最后增加一个维度
                img = img[:, :, ::-1].copy().transpose(2, 0, 1)  # （h,w,1）->(1,h,w)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device).float()
                img = (img - 128.0) / 255.0  # 归一化

                if img.ndimension() == 3:
                    img = img.unsqueeze(0)  # 最外层加一个维度->[1,h,w,1]

                start_time = time.time()
                pred = self.model(img)
                time_mark = time.time()
                infer_time = float(time_mark - start_time) * 1000  # 推理时间

                output_list = []
                for i, item_pred in enumerate(pred):  # 获取不同尺度的预测结果
                    output_list.append(self.model_loss[i](item_pred))  # 返回的是predict出来的所有bounding box（已反向还原）
                output = torch.cat(output_list, 1)  # 不同尺度的边界框合在一起
                output = non_max_suppression(output, config_params["io_params"]["num_cls"],
                                             conf_thres=self.conf_thres, nms_thres=self.nms_thres)

                output = output[0]  # 一次只处理一张图，所以只取第一个
                post_process_time = float(time.time()-time_mark)*1000  # 后处理用时
                total_time = post_process_time + infer_time
                avg_time += total_time

                if output is None:
                    cv2.imwrite(os.path.join(result_path, 'result_'+filename), img_origin)  # 保存结果
                    self.logger.info("image_name:%s -> no targets, infer time:%.2fms, post_process time:%.2fms, total time:%.2fms" % (filename, infer_time, post_process_time, total_time))
                    continue

                if img_origin.shape[0:2] != tuple(self.target_shape):
                    self.logger.info("adjust coordiantes")  # 坐标调整，从网络输入图片尺寸调整到实际图片尺寸
                    output[:, :4] = scale_coords(img.shape[1:3], output[:, :4], img_origin.shape).round()  # 这个函数有问题

                # 画框
                for *xyxy, conf, cls_score, cls_pred in reversed(output):
                    label = '%s %.2f' % (self.class_names[int(cls_pred)], conf*cls_score)  # conf*cls_score为类别目标置信度
                    plot_one_box(xyxy, img_origin, label=label, color=self.colors[int(cls_pred)], line_thickness=3)

                if os.path.exists(result_path) is False:
                    os.makedirs(result_path)

                cv2.imwrite(os.path.join(result_path, 'result_'+filename), img_origin)  # 保存结果
                self.logger.info("image_name:%s -> detect finished, infer time:%.2fms, post_process time:%.2fms, total time:%.2fms" % (filename, infer_time, post_process_time, total_time))

            self.logger.info("detect avg_time: %.2fms" % (avg_time/num))

    def __resize_img(self, img0, new_shape, color=128):
        '''
        保持原图的宽高比进行调整至目标图像尺寸, 两边填充（缩放的是宽或者高，取最接近的那个）
        和DetectDataset重load rect处理方式一样
        :param img0: 输入待处理原图
        :param new_shape: 目标尺寸
        :param color: 填充颜色
        :return: 处理后的图像
        '''
        shape = img0.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        unpad_shape = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - unpad_shape[0], new_shape[0] - unpad_shape[1]  # wh padding

        img = cv2.resize(img0, unpad_shape, interpolation=cv2.INTER_LINEAR)

        dw, dh = dw / 2, dh / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img



if __name__ == '__main__':
    logger = config_logger(log_dir='E:\Graduate_Design\YOLO-Fastest\\test_result',
                              log_name='cpu-test.log', tensorboard=False)  # 加载日志模块

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备选择
    detect = Detect_YOLO(device, model_path="E:\Graduate_Design\YOLO-Fastest\models\pytorch\YOLO-Fastest_epoch_27.pth",
                         config_params=config_params, logger=logger)
    detect.batch_detect(root_path="E:\Graduate_Design\YOLO-Fastest\\test_data",
                        result_path="E:\Graduate_Design\YOLO-Fastest\\test_result")





