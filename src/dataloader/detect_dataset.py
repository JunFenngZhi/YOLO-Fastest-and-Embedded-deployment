import os
import random
import cv2
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as xmlET  # xml.etree.ElementTree模块实现了一个简单而高效的API用于解析和创建XML数据
from torch.utils.data import DataLoader
import numpy as np
from _config import config_params
import logging
from utils.general import xywh2xyxy, xyxy2xywh


# 数据集中的目标类别
classes = config_params["io_params"]["class_names"]  # 优化时，0->carrier, 1->defender, 2->destroyer 计算loss用的是one-hoe code


def draw_box(img, bboxes):
    for b in bboxes:
        img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    return img

class DetectDataset(Dataset):
    def __init__(self, img_size, logger, augment=True, aug_params=None, max_boxes=64, val=False):
        self.aug_params = aug_params  # data_augmentation参数
        self.img_size = img_size  # 网络的输入图像shape.要将训练集中的图转换成该尺寸
        self.fliplr = self.aug_params['fliplr']  # 左右翻转概率
        if val == True:
            logger.info(" Val Datasest Loading..")
            self.dataset_dir = self.aug_params['val_dataset_dir']
        else:
            logger.info("Training Datasest Loading..")
            self.dataset_dir = self.aug_params['train_dataset_dir']
        self.gussian_filter = self.aug_params['gussian_filter']  # 高斯滤波概率
        self.max_boxes = max_boxes  # 每张图最大允许目标数
        self.augment = augment
        self.logger = logger

        self.file_path_img = os.path.join(self.dataset_dir, "img")  # 自动补齐斜杠
        self.file_path_xml = os.path.join(self.dataset_dir, "xml")
        self.dataset_dict = {}  # 保存image及其对应label

        pathDir = os.listdir(self.file_path_xml)  # 返回指定的文件夹包含的文件名字列表（所有的图和label）
        for idx in range(len(pathDir)):   # 遍历每一张图片及其标签
            if idx % 1000 == 0:
                self.logger.info("Loading:%d/%d" % (idx, len(pathDir)))
            filename = pathDir[idx]
            tree = xmlET.parse(os.path.join(self.file_path_xml, filename))  # 打开文件（对应一张图的label）
            objs = tree.findall('object')  # 返回所有tag为'object'的元素
            num_objs = len(objs)

            _labels = []  # 存储本图对应的目标标签  一张图可以有多个目标
            for ix, obj in enumerate(objs): # 遍历标签中的每个目标
                _bbox = obj.find('bndbox')
                x1 = float(_bbox.find('xmin').text)
                y1 = float(_bbox.find('ymin').text)
                x2 = float(_bbox.find('xmax').text)
                y2 = float(_bbox.find('ymax').text)
                _cls_name = obj.find('name').text  # 获取类别名字（字符串）
                _cls_index = classes.index(_cls_name) # 根据名字，得出对应下标
                _labels.append([_cls_index, x1, y1, x2, y2])  # 转换成(cls_id,x_min,y_min,x_max,y_max)

            _image_name = os.path.splitext(filename)[0] + ".jpg"  # spilt_text将文件名和扩展名分开 (标签xml和图片同名)
            _image_name = os.path.join(self.file_path_img, _image_name)  # 图片的绝对路径
            self.dataset_dict.update({_image_name: _labels})  # 更新字典，是图片和标签对应

        self.img_list = list(self.dataset_dict.keys())  # 图像列表
        self.logger.info("Loading finish！ dataset contain %d items" % (self.__len__()))

    # 保持原图的宽高比进行调整至目标图像尺寸, 两边填充（缩放的是宽或者高，取最接近的那个）
    def load_rect(self, index, color=128):
        new_shape = self.img_size  # 目标图像尺寸
        img_path = self.img_list[index]
        ori_img = cv2.imread(img_path)  # 原始图
        labels = np.array(self.dataset_dict[img_path])  # 获取该图片对应的label

        if new_shape[2] == 1:
            img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
        else:
            img = ori_img  # 深度拷贝，不会影响原图

        shape = img.shape[0:2]
        if shape == tuple(new_shape[0:2]):
            return img, labels, ori_img  # 已经满足要求，不需要resize

        # 目标框尺寸除以原图图像的长和宽，归一化。方便后续补齐边缘后调整边框坐标。因为不确定是宽调整还是高调整。
        labels[:, [2, 4]] = labels[:, [2, 4]] / shape[0]
        labels[:, [1, 3]] = labels[:, [1, 3]] / shape[1]

        # resize
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 目标宽高和原始宽高的最小比例因子
        unpad_shape = int(round(shape[1] * r)), int(round(shape[0] * r))  # 将原图向最接近的宽/高变换过去（尽量减少图像变形）
        dw, dh = new_shape[1] - unpad_shape[0], new_shape[0] - unpad_shape[1]  # wh padding 宽或高和目标宽高的差值
        img = cv2.resize(img, unpad_shape, interpolation=cv2.INTER_LINEAR)

        # 边缘填充
        dw, dh = dw/2, dh/2  # 两边对称补齐
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 四舍五入取整（单位为像素）
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        # 修正label中边界框的坐标
        labels[:, [2, 4]] = (labels[:, [2, 4]] * shape[0]) * r + top  # 还原坐标，加上边缘补齐部分。
        labels[:, [1, 3]] = (labels[:, [1, 3]] * shape[1]) * r + left

        return img, labels, ori_img

    @staticmethod  # 将逐个样本组合起来形成batch返回。组合方式自定义
    def collate_fn(batch):
        images = []
        bboxes = []
        for img, box in batch:  # batch是一个list,里面每个元素为一个元组（data,label)。batch长度等于batch_size
            images.append([img])
            bboxes.append([box])
        images = np.concatenate(images, axis=0)  # 将原本成对的img和label拆开。各自组合成一个大的numpy,形成batch
        bboxes = np.concatenate(bboxes, axis=0)
        images = images.transpose(0, 3, 1, 2)  # reshape将各维度度重新组合。【num.h,w,1】->【num,1,h,w】
        images = torch.from_numpy(images).div(255.0)   # 将numpy转为tensor,两者共享内存
        bboxes = torch.from_numpy(bboxes)
        return images, bboxes

    def __len__(self):
        return len(self.img_list)

    # 返回一个（data,label）
    def __getitem__(self, index):
        img, labels, ori_img = self.load_rect(index)  # 调整图片大小至目标尺寸

        if len(labels):
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # 转换成（x_cen,y_cen,w,h）
            labels[:, [2, 4]] /= self.img_size[0]  # 归一化
            labels[:, [1, 3]] /= self.img_size[1]

        # 数据随机增强(配置参数中只用了模糊和左右翻转。因此其它变换没有实现)
        if self.augment:
            if random.random() < self.gussian_filter:
                _ret = random.random()  # 随机高斯滤波
                if _ret < 0.4:
                    img = cv2.GaussianBlur(img, (7, 7), 0)
                elif _ret < 0.2:
                    img = cv2.GaussianBlur(img, (5, 5), 0)
                else:
                    img = cv2.GaussianBlur(img, (3, 3), 0)
            if random.random() < self.fliplr:
                img = np.fliplr(img)  # 随机左右翻转
                labels[:, 1] = 1 - labels[:, 1]  # 翻转后中心x坐标变化

        # 调整顺序，将类别id移到末尾  调整后（x_cen,y_cen,w,h,cls_id）
        nL = len(labels)  # 该图片对应的目标个数
        if nL:
            cls_id = labels[:, 0].copy()
            labels[:, 0:4] = labels[:, 1:5]
            labels[:, 4] = cls_id

        if self.img_size[2] == 1:
            img = np.expand_dims(img, -1)  # 在最后多加一个维度，shape变为【h,w,1】
        
        img = img - 128.0
        img = np.ascontiguousarray(img)  # 将一个内存不连续存储的数组转换为内存连续存储的数组,使得运行速度更快

        out_bboxes1 = np.zeros([self.max_boxes, 6])
        out_bboxes1[:min(labels.shape[0], self.max_boxes), 0:5] = labels[:min(labels.shape[0], self.max_boxes)]
        out_bboxes1[:min(labels.shape[0], self.max_boxes), 5] = 255.0  # 最后是（x_cen,y_cen,w,h,cls_id,255.0）坐标已经归一化

        return img, out_bboxes1

def config_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_dir = config_params["io_params"]["log_path"]  # 生成日志文件夹
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_dir+'info.log',mode='w')  # 创建一个handler，用于写入日志文件
    ch = logging.StreamHandler()   # 再创建一个handler，用于输出到控制台
    formatter = logging.Formatter('%(asctime)s——%(message)s')  # 定义handler的输出格式formatter
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    #tbwriter = SummaryWriter(log_dir)  # 调用tensorboard
    return logger #tbwriter

if __name__ == '__main__':
    logger = config_logger()
    dataset = DetectDataset(img_size=config_params["io_params"]["input_size"], max_boxes=64,
                            aug_params=config_params["augment_params"], logger=logger)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=0, sampler=None, pin_memory=True,  # num_workers != 0可能报错
                            collate_fn=DetectDataset.collate_fn)  # collate_fn是自定义batch，假如不指定则调用默认的collate_fn

    for i in range(1):
        imgs, targets = dataset.__getitem__(i)
        print("target.shape:", targets.shape)
        print("imgs.shape:", imgs.shape)
        targets[:, 0:4] = xywh2xyxy(targets[:, 0:4])  # 原来(x_cen,y_cen,w,h,cls_id,255.0)
        targets[:, [1, 3]] *= imgs.shape[0]  # 反归一化
        targets[:, [0, 2]] *= imgs.shape[1]

        a = imgs.astype(np.float32).copy()  # tensor转为numpy
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)  # 将灰度空间转回彩色空间，方便边界框标注。
        a = draw_box(a, targets.astype(np.int32))
        cv2.imwrite("./test_o.jpg", a)  # 将数据解码显示出来

    for batch_id, (imgs, targets) in enumerate(dataloader):
        print("target.shape:", targets.shape)
        print("imgs.shape:", imgs.shape)
        break


'''
处理后的img每个像素减去128,并除了255
label都是归一化后的
'''

'''
数据集格式要求：xml和jpg文件同名（除后缀）
'''

