import torch
from torch.utils.data import DataLoader
from dataloader.detect_dataset import DetectDataset
from utils.general import xywh2xyxy, non_max_suppression, bbox_iou, del_tensor_element
import numpy as np


class Validation():
    def __init__(self, params, logger, dataset, device, model_loss):
        self.logger = logger
        self.model_loss = model_loss
        self.device = device
        self.bs = params["train_params"]["batch_size"]
        self.n_anchors = params["io_params"]["num_anchors"]
        self.img_size = params["io_params"]["input_size"]  # 网络输入图片的尺寸大小
        self.num_cls = params["io_params"]["num_cls"]
        self.cls_name = params["io_params"]["class_names"]
        self.IOU_threshold = params["train_params"]["IOU_val_thre"]  # 预测位置和真实位置匹配阈值
        self.conf_thres = params["io_params"]["conf_thre"]
        self.nms_thres = params["io_params"]["nms_thre"]
        self.dataloader = DataLoader(dataset, batch_size=self.bs, num_workers=0, drop_last=True,
                                     pin_memory=True, shuffle=True, collate_fn=DetectDataset.collate_fn)
        self.target_num = torch.zeros((self.num_cls))  # 统计各类别目标的总数
        self.match_list = []    # 存储预测bbox和真实目标位置的匹配结果
        for _ in range(self.num_cls):
            self.match_list.append([])

    def get_mAP(self, model, epoch):
        self.clear()  # 清空所有记录
        model.eval()  # 防止BN 和dropout对验证结果的影响
        with torch.no_grad():
            # 遍历验证集，将检测结果和真实结果匹配
            for batch_id, (imgs, targets) in enumerate(self.dataloader):
                #self.logger.info("—————— val batch: %d  —————" % (batch_id))
                targets = targets.to(self.device).float()
                targets = self.__recover_targets(targets)  # 真实目标边框坐标恢复  shape(bs,64,6)

                imgs = imgs.to(self.device).float()
                pred = model(imgs)
                output_list = []
                for i, item_pred in enumerate(pred):  # 各个scale的输出单独计算统计
                    output_list.append(self.model_loss[i](item_pred))  # 返回predict的所有bounding box（已反向还原为真实坐标）
                output = torch.cat(output_list, 1)  # 不同尺度的边界框合在一起
                output = non_max_suppression(output, self.num_cls, conf_thres=self.conf_thres,device=self.device,
                                             nms_thres=self.nms_thres)  # NMS  output_shape(bs,:,7) 每张图检测数量不确定

                for img_id, img_pred in enumerate(output):  # 遍历预测的每一张图片
                    img_target = targets[img_id]  # 本图中对应的真实目标位置
                    img_target = img_target[img_target[:, 5] > 1]  # 去除无效数据

                    for t in img_target:
                        self.target_num[int(t[4])] += 1 # 统计本图各类别真实目标数量

                    if img_pred is None:  # 该图片没有可靠的预测结果（不能用==）
                        continue

                    unique_labels = img_pred[:, 6].to(self.device).unique()  # 获取本图预测结果中包含的类别列表
                    for c in unique_labels:   # 对预测出来的每一个类别中的所有结果进行验证
                        target_c = img_target[img_target[:, 4] == c]
                        img_pred_c = img_pred[img_pred[:, 6] == c]
                        c = int(c)
                        for t in img_pred_c:
                            if target_c.size(0) == 0:  # 如果真实结果没有这个类别，则可以直接判定是误检，加FP。
                                self.match_list[c].append(np.array([t[4], 'FP']))
                                continue
                            ious = bbox_iou(t.unsqueeze(0), target_c)  # 一个pred对所有target求IOU
                            match = False
                            for index, iou in enumerate(ious):
                                if iou > self.IOU_threshold:
                                    self.match_list[c].append(np.array([t[4], 'TP']))  # 匹配成功
                                    match = True
                                    target_c = del_tensor_element(target_c, index)  # 去除已经匹配过的，防止重复TP.（这样处理可能会引起误差，因为保留的不一定是conf最大那个）
                                    break
                            if match == False:  # 匹配失败
                                self.match_list[c].append(np.array([t[4], 'FP']))

            for i in range(self.num_cls):
                self.match_list[i].sort(key=lambda x: x[0], reverse=True)  # 根据conf降序排列

            mAP = 0
            self.logger.info("—————— epoch: %d validation results —————" % (epoch))
            for c in range(self.num_cls):
                AP = self.__calculate_AP(cls=c)
                self.logger.info("class: %s, target_num = %d, AP = %.3f" % (self.cls_name[c], self.target_num[c], AP))
                mAP += AP
            mAP /= self.num_cls
            self.logger.info("mean AP: %.3f" % (mAP))
            self.logger.info("——————————————————————————")

        model.train()
        return mAP

    def __calculate_AP(self, cls):
        # 计算P-R列表
        match_list = self.match_list[cls]
        PR_list = []
        for i in range(len(match_list)):  # 选取不同的置信度阈值。因为已经降序排列，所以按下标取值即可
            TP = FP = 0
            for index in range(i+1):  # 遍历所有置信度在阈值之上的匹配项
                if match_list[index][1] == 'TP':
                    TP += 1
                else:
                    FP += 1
            FN = self.target_num[cls] - TP
            if FN < 0:
                self.logger.error("error: FN less than 0!")
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            if i > 0 and recall == PR_list[-1][1]:  # recall和前一个重复
                if precision > PR_list[-1][0]:  # 更新precision为最大值
                    PR_list[-1][0] = precision
            else:
                PR_list.append(np.array([precision, recall]))

        # 计算P-R曲线下包围的面积（AP）
        AP = 0
        pre_recall = 0  # 前一项的recall值
        PR_list = np.array(PR_list)
        for i in range(PR_list.shape[0]):  # 遍历每一项
            max_precision = np.max(PR_list[i:], axis=0)[0]  # 从第i项之后最大的precision值
            AP += (PR_list[i][1]-pre_recall) * max_precision  # 多个矩形面积近似逼近曲线下面积
            pre_recall = PR_list[i][1]

        return AP

    def clear(self):
        self.match_list.clear()
        for c in range(self.num_cls):
            self.match_list.append([])
            self.target_num[c] = 0.

    # 将归一化的目标边框坐标恢复为真实坐标，并转换坐标形式（x1,y1,x2,y2,cls_id,255）
    def __recover_targets(self, targets):
        in_h = self.img_size[0]
        in_w = self.img_size[1]
        targets[:, :, (0, 2)] = targets[:, :, (0, 2)] * in_w
        targets[:, :, (1, 3)] = targets[:, :, (1, 3)] * in_h
        for b in range(self.bs):
            targets[b, :, 0:4] = xywh2xyxy(targets[b, :, 0:4])

        return targets














