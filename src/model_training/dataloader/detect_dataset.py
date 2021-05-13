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
from tensorboardX import SummaryWriter


# 数据集中的目标类别
classes = config_params["io_params"]["class_names"]  # 优化时，0->carrier, 1->defender, 2->destroyer 计算loss用的是one-hoe code

def config_logger(log_dir, log_name, tensorboard=True):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # INFO或者比INFO等级高的都显示
    if os.path.exists(log_dir) is False:
        os.makedirs(log_dir)
    fh = logging.FileHandler(os.path.join(log_dir,log_name), mode='w')  # 创建一个handler，用于写入日志文件
    ch = logging.StreamHandler()   # 再创建一个handler，用于输出到控制台
    formatter = logging.Formatter('%(asctime)s——%(message)s')  # 定义handler的输出格式formatter
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    if tensorboard:
        tbwriter = SummaryWriter(log_dir)  # 调用tensorboard
        return logger, tbwriter
    else:
        return logger

def draw_box(img, bboxes):
    for b in bboxes:
        img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    return img

class DetectDataset(Dataset):
    def __init__(self, input_shape, origin_img_shape, logger, augment=True, aug_params=None, max_boxes=64, val=False):
        self.aug_params = aug_params  # data_augmentation参数
        self.origin_img_shape = origin_img_shape  # 数据集原始图像shape
        self.input_shape = input_shape  # 网络的输入图像shape。要将训练集中的图转换成该尺寸
        if val == True:
            logger.info(" Val Datasest Loading..")
            self.dataset_dir = self.aug_params['val_dataset_dir']
        else:
            logger.info("Training Datasest Loading..")
            self.dataset_dir = self.aug_params['train_dataset_dir']
        self.fliplr = self.aug_params['fliplr']  # 左右翻转概率
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

            _labels = []  # 存储本图对应的目标标签  一张图可以有多个目标
            for ix, obj in enumerate(objs):  # 遍历标签中的每个目标
                _bbox = obj.find('bndbox')
                x1 = float(_bbox.find('xmin').text)
                y1 = float(_bbox.find('ymin').text)
                x2 = float(_bbox.find('xmax').text)
                y2 = float(_bbox.find('ymax').text)
                _cls_name = obj.find('name').text  # 获取类别名字（字符串）
                _cls_index = classes.index(_cls_name)  # 根据名字，得出对应下标
                _labels.append([_cls_index, x1, y1, x2, y2])  # 转换成(cls_id,x_min,y_min,x_max,y_max)

            _image_name = os.path.splitext(filename)[0] + ".jpg"  # spilt_text将文件名和扩展名.xml分开 (标签xml和图片同名)
            _image_name = os.path.join(self.file_path_img, _image_name)  # 图片的绝对路径
            self.dataset_dict.update({_image_name: _labels})  # 更新字典，是图片和标签对应

        self.img_list = list(self.dataset_dict.keys())  # 图像列表
        self.logger.info("Loading finish！ dataset contain %d items" % (self.__len__()))

    # 调整图像大小
    def load_rect(self, index):
        img_path = self.img_list[index]
        ori_img = cv2.imread(img_path)  # 原始图（BGR格式）
        labels = np.array(self.dataset_dict[img_path])  # 获取该图片对应的label

        if self.input_shape[2] == 1 and self.origin_img_shape[2] != 1:  # 网络要求单通道输入
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
        else:
            img = ori_img  # 深度拷贝，不会影响原图

        if self.input_shape[0:2] != self.origin_img_shape[0:2]:  # 输入图片大小调整至满足网络要求
            img = cv2.resize(img,(self.input_shape[1], self.input_shape[0]))

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
        images = images.transpose(0, 3, 1, 2)  # reshape将各维度度重新组合。【num.h,w,chanel】->【num,chanel,h,w】
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
            labels[:, [2, 4]] /= self.origin_img_shape[0]  # 归一化
            labels[:, [1, 3]] /= self.origin_img_shape[1]

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

        if self.input_shape[2] == 1:
            img = np.expand_dims(img, -1)  # 在最后多加一个维度，shape变为【h,w,1】
        
        img = img - 128.0
        img = np.ascontiguousarray(img)  # 将一个内存不连续存储的数组转换为内存连续存储的数组,使得运行速度更快

        out_bboxes1 = np.zeros([self.max_boxes, 6])
        out_bboxes1[:min(labels.shape[0], self.max_boxes), 0:5] = labels[:min(labels.shape[0], self.max_boxes)]
        out_bboxes1[:min(labels.shape[0], self.max_boxes), 5] = 255.0  # 最后是（x_cen,y_cen,w,h,cls_id,255.0）坐标已经归一化

        return img, out_bboxes1


if __name__ == '__main__':
    logger = config_logger(log_dir=config_params["io_params"]["log_path"],
                           log_name='info.log', tensorboard=False)
    dataset = DetectDataset(input_shape=config_params["io_params"]["input_shape"], max_boxes=64,
                            origin_img_shape=config_params["io_params"]["origin_img_shape"],
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
        cv2.imwrite(".\\test_o.jpg", a)  # 将数据解码显示出来

    for batch_id, (imgs, targets) in enumerate(dataloader):
        print("target.shape:", targets.shape)
        print("imgs.shape:", imgs.shape)
        break


'''
处理后的img每个像素减去128,并除了255
label中的目标坐标都归一化为0-1
'''

'''
数据集格式要求：xml和jpg文件同名（除后缀）
'''

