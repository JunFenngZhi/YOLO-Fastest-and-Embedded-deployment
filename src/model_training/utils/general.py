# -*- coding=utf-8 -*-
import cv2
import torch
import random
import numpy as np


def xyxy2xywh(x):
    y = torch.zeros_like(x) if isinstance(  # 如果x是tensor类型，则生成的y也是tensor。否则生成numpy类型
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    y = torch.zeros_like(x) if isinstance(   # 如果x是tensor类型，则生成的y也是tensor。否则生成numpy类型
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y[:,0:1]是边框左上角顶点
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y[:,2:3]是边框右下角顶点
    return y


# 批量计算IOU.
def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


# Plots one bounding box on image img
def plot_one_box(xyxy, img, color=None, label=None, line_thickness=None):
    # 调节tf,tl可以控制边框和字体粗细
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))  # 左上角顶点坐标， 右下角顶点坐标
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)  # 画出目标框
    if label:
        tf = min(tl - 1, 2)  # font thickness字体粗细
        t_size = cv2.getTextSize(label, fontFace=0, fontScale=tl / 5, thickness=tf)[0]  # fontFace字体， fontScale字体大小
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, thickness=-1, lineType=cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 5, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# 删除tensor中的某个元素(必须要用值接返回值，不能直接修改arr)
def del_tensor_element(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1, arr2), dim=0)

'''
流程如下； 
1、置信度阈值去掉一部分检测结果√
2、剩余检测结果中，根据分类结果，选择概率最大的类别作为该边界框的检测类别√
3、本项目采用分类别进行非极大值抑制的方法。（也有不区分类别都做NMS的方法。区分类别容易错检，不区分类别容易漏检。）
4、首先挑出该类别下置信度最大的那个框，然后用他和该类别剩余所有框进行IOU计算
5、去掉IOU>NMS_threshold的那些结果  重复上述步骤直到该类别没有剩余边界框
6、返回每个图片所有类别的NMS处理后的检测结果。每个检测结果存储形式为(x1, y1, x2, y2, object_conf, class_score, class_pred)
注： class_pred值对应的类别和数据集导入时数组下标对应的类别一致
'''
def non_max_suppression(prediction, num_classes,conf_thres=0.5, nms_thres=0.4):

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]  # 转换后结果赋值回prediction

    output = [None for _ in range(len(prediction))]  # len(prediction)：只遍历第一个维度，所以值为bs
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()  # 保留置信度符合要求的结果下标，封装成list
        image_pred = image_pred[conf_mask]  # 由下标list进行筛选

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # 获取最大的类别概率
        # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。dim=1 指每行的最大值
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], dim=1,  keepdim=True)

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        # Iterate through all predicted classes
        unique_labels = detections[:, -1].unique()  # 获取类别列表
        for c in unique_labels:
            # 获取指定类别的检测结果
            detections_class = detections[detections[:, 6] == c]

            # 根据目标置信度降序排列（）
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)  # 返回下标和值
            detections_class = detections_class[conf_sort_index]  # 根据下标重排序

            # 非极大值抑制过程：每次挑出置信度最大的边界框，作为候选。然后计算剩余所有边界框和这个候选框的IOU。
            # 对于那些重叠率过高的，认为是对同一个物体的检测，去除。然后再重复上述过程所有框都被决定去留。
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold（去掉重叠率过高的）
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data  # ？

            # Add max detections to outputs 因为每一个类别都要做一次非极大抑制，所以要把后面结果和前面结果拼接起来
            output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))

    return output