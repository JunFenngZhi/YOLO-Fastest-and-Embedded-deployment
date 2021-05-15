import numpy as np
import torch
import torch.nn as nn
import math
from utils.general import bbox_iou
from _config import config_params


def clip_by_tensor(t,t_min,t_max):
    t=t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def MSELoss(pred,target):
    return (pred-target)**2


def BCELoss(pred,target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


class YOLOLossV3(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, device):
        super(YOLOLossV3, self).__init__()
        self.anchors = anchors   # size(3,2) 其中一组（一个scale）
        self.num_anchors = len(anchors)  # 每个尺度下的anchor boxes数量
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes  # 每个boxes对应的预测量
        self.input_shape = input_shape
        self.ignore_threshold = config_params["train_params"]["IOU_loss_thre"]  # anchor box 筛选标准（只根据shape） IOU ignore
        self.device = device

        # loss加权系数
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    # 任意继承了nn.Module的类对象都可以简写调用类来调用forward函数
    def forward(self, input, targets=None):
        bs = input.size(0)  # batch_size
        in_h = input.size(2)  # predict特征图的高
        in_w = input.size(3)  # predict特征图的宽
        stride_h = self.input_shape[0] / in_h  # 特征图相对原图缩小倍数
        stride_w = self.input_shape[1] / in_w

        # 原_ah/(原_h/特_h)=(原_ah/原_h)*特_h  在特征图坐标系下anchor_box的长宽
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        # 变换后（bs, num_anchors, in_h, in_w, bbox_attrs）
        prediction = input.view(bs,  self.num_anchors,
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs（特征图各个像素预测的各个anchor box的tx,ty,tw,th,conf,pred_cls）
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = torch.sigmoid(prediction[..., 4])       # Confidence
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Class pred.(yolov3处理方法）

        if targets is not None:
            #  build target（返回的是初步筛选后认为具体位置可能包含目标的anchor_box的对应参数）
            mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.get_target(targets, scaled_anchors,
                                                                           in_w, in_h,
                                                                           self.ignore_threshold)
            mask, noobj_mask = mask.to(self.device), noobj_mask.to(self.device)
            tx, ty, tw, th = tx.to(self.device), ty.to(self.device), tw.to(self.device), th.to(self.device)
            tconf, tcls = tconf.to(self.device), tcls.to(self.device)

            # （x,y,w,h）的loss只考虑对应位置有目标的特定anchor box
            loss_x = self.bce_loss(x * mask, tx * mask)
            loss_y = self.bce_loss(y * mask, ty * mask)
            loss_w = self.mse_loss(w * mask, tw * mask)
            loss_h = self.mse_loss(h * mask, th * mask)

            # confidence的loss要加权有目标的anchor box和无目标的anchor box.有目标，则confidence应该趋向1，无目标则趋向0。
            loss_conf = self.bce_loss(conf * mask, mask) + 0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)

            # 类别概率loss只考虑对应位置有目标的特定anchor box
            loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1])

            # 总loss 为各部分loss加权
            loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

            return loss, loss_x.item(), loss_y.item(), loss_w.item(),\
                   loss_h.item(), loss_conf.item(), loss_cls.item()
        else:  # 最后detection的时候用来计算真实bound box（考虑移走新开一个函数）
            FloatTensor = torch.FloatTensor
            LongTensor = torch.LongTensor
            
            # Calculate offsets for each grid
            #grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(
            #    bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
            #grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(
            #    bs * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

            # 生成网格。存储的是网格左上角坐标（特征图坐标系下的） 3->num_anchors
            grid_x = torch.arange(in_w).repeat(bs, 3, in_h, 1).type(FloatTensor)  # repeat()：将原有tensor自增到指定的shape
            grid_y = torch.arange(in_h).repeat(bs, 3, in_w, 1).permute(0, 1, 3, 2).type(FloatTensor)

            # Calculate anchor w, h
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))  # 对shape[3,2]的anchors进行列挑选，只选择第一列（w）
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))  # 对shape[3,2]的anchors进行列挑选，只选择第二列（h）
            anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
            anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)  # shape?
            
            # 存储还原后的预测值
            pred_boxes = FloatTensor(prediction[..., :4].shape)  # 存储计算得出的（w,h,x_cen,y_cen）（特征图坐标系下）

            # scale
            _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)  # 在末尾重复一次，不是数值乘2

            if x.is_cuda:
                anchor_w = anchor_w.to(self.device)
                anchor_h = anchor_h.to(self.device)
                grid_x = grid_x.to(self.device)
                grid_y = grid_y.to(self.device)
                pred_boxes = pred_boxes.to(self.device)
                _scale = _scale.to(self.device)

            # Add offset and scale with anchors. pred_boxes.shape:(bs, num_anchors, in_h, in_w,4)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

            # Results
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,  # 将in_w,in_h这两个维度合在一起（尺度恢复至网络输入图片坐标系）
                                conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)  # -1 指的是在最后一个维度上拼接
            return output.data

    # 根据目标的位置和真实边界框，筛选出有效的anchor box，进一步计算loss
    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):
        bs = target.size(0)

        mask = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)  # 标记哪些anchor_boxes可能对应有目标
        noobj_mask = torch.ones(bs, self.num_anchors, in_h, in_w, requires_grad=False)  # 标记哪些anchor_boxes一定无目标对应
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False)

        for b in range(bs):  # 遍历batch中的每个图
            for t in range(target.shape[1]):  # 遍历图中每个目标
                if target[b, t, 5] < 1:  # 最后一位标记这个是否为有效目标
                    break
                # Convert to position relative to box（将目标信息转换到输出特征图坐标系上）
                gx = target[b, t, 0] * in_w
                gy = target[b, t, 1] * in_h
                gw = target[b, t, 2] * in_w
                gh = target[b, t, 3] * in_h

                if gw <= 0 or gh <= 0:  # 该目标不存在
                    continue

                # Get grid box indices（对应特征图上的像素坐标）
                gi = int(gx)
                gj = int(gy)
                # Get shape of gt box  【【0，0，gw，gh】】
                gt_box = torch.FloatTensor(np.array([0.0, 0.0, gw, gh], dtype=np.float32)).unsqueeze(0)
                # Get shape of anchor box 【【0，0，a_w，a_h】。。。】
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(anchors)), 1))
                # Calculate iou between gt and anchor boxes（单纯根据shape计算IOU）
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                # Where the overlap is larger than threshold set mask to zero (ignore)
                noobj_mask[b, anch_ious > ignore_threshold, gj, gi] = 0
                # Find the best matching anchor box
                best_n = np.argmax(anch_ious)  # 对于该目标，专门指定一个anchor_box负责优化（根据目标坐标及shape匹配度）
                mask[b, best_n, gj, gi] = 1

                # Coordinates（相对cell左上角坐标的偏移量，方便后面直接和predict出来的x,y作差求loss）
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                # Width and height
                tw[b, best_n, gj, gi] = math.log(gw/anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = math.log(gh/anchors[best_n][1] + 1e-16)
                # object
                tconf[b, best_n, gj, gi] = 1
                # One-hot encoding of label
                tcls[b, best_n, gj, gi, int(target[b, t, 4])] = 1

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls

