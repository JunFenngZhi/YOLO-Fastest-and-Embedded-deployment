import math
import os
import time
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from model.yolo_fastest import YoloFastest
from dataloader.detect_dataset import DetectDataset
from loss.yolo_loss import YOLOLossV3
from _config import config_params
from validate import Validation


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


def train(params, device, tbwriter):
    # 参数加载
    pretrained_pth = params["train_params"]["pretrained_pth"]  # 预训练模型的路径
    save_path = params["io_params"]["save_path"]  # 设定保存模型的路径
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    pred_branch = len(params["io_params"]["strides"])  # 网络输出分支数
    total_epochs = params["train_params"]["total_epochs"]
    batch_size = params["train_params"]["batch_size"]

    # loss类
    model_loss = []
    for i in range(pred_branch):  # 每一个尺度的anchor对应一个loss类
        model_loss.append(YOLOLossV3(anchors=params["io_params"]["anchors"][i], num_classes=params["io_params"]["num_cls"],
                                     input_shape=params["io_params"]["input_shape"], device=device))

    # 模型实例化
    model = YoloFastest(params["io_params"]).to(device)

    # 模型导入或初始化
    if os.path.exists(pretrained_pth):
        logger.info("Load pretrained model %s" % pretrained_pth)
        net_param = torch.load(pretrained_pth, map_location=device)  # 加载预训练模型的参数
        model.load_state_dict(net_param)   # 将参数设置到model中
    else:
        logger.info("initialize model")
        model.initialize_weights()

    # 分批读入数据集，并作数据增强
    train_dataset = DetectDataset(input_shape=config_params["io_params"]["input_shape"], max_boxes=64,
                                  origin_img_shape=config_params["io_params"]["origin_img_shape"],
                                  aug_params=config_params["augment_params"], logger=logger)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,  # 分批读取数据用于训练
                            drop_last=True, pin_memory=True, shuffle=True,
                            collate_fn=DetectDataset.collate_fn)  # collate_fn 自定义batch输出
    val_dataset = DetectDataset(input_shape=config_params["io_params"]["input_shape"], max_boxes=64, augment=False,
                                origin_img_shape=config_params["io_params"]["origin_img_shape"],
                                aug_params=config_params["augment_params"], logger=logger, val=True)
    val = Validation(params=config_params, logger=logger, dataset=val_dataset,
                     device=device, model_loss=model_loss)  # 用于模型验证，计算mAP

    batch_per_epoch = len(dataloader)  # 一个epoch包含的batch数量
    num_warm = max(3 * batch_per_epoch, 1000)

    # 优化器设置
    train_params = params["train_params"]
    optimizer = optim.Adam(model.parameters(), lr=train_params['lr0'], betas=(0.9, 0.999), eps=1e-08)

    def lf(epoch):
        return ((1+math.cos(epoch*math.pi/total_epochs))/2)*0.8+0.2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 自定义调整学习率（每个epoch调整一次）

    start_epoch = 0
    scheduler.last_epoch = start_epoch - 1
    total_steps = (total_epochs - start_epoch) * batch_per_epoch  # 所有epochs中一共要训练的batch数量
    step_count = 0  # 记录是第几个batch

    logger.info("Start training.")

    for epoch in range(start_epoch, total_epochs):
        model.train()
        for batch_id, (imgs, targets) in enumerate(dataloader):
            start_time = time.time()

            imgs = imgs.to(device).float()  # train_imgs
            targets = targets.to(device).float()   # targets

            # epoch内渐变修改lr
            iteration = batch_id + batch_per_epoch * epoch  # 当前训练的迭代次数
            if iteration <= num_warm:
                xi = [0, num_warm]
                for x in optimizer.param_groups:
                    x['lr'] = np.interp(iteration, xi, [0.0, x['initial_lr'] * lf(epoch)])  # 每个iteration线性插值修改学习率

            # 每个batch计算梯度前先将梯度初始化为0
            optimizer.zero_grad()

            # 预测结果（两个scale）
            pred = model(imgs)

            # 计算loss
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = []   # 存储每次迭代时每一项loss的数值
            for _ in range(len(losses_name)):
                losses.append([])
            for i, item_pred in enumerate(pred):  # 各个scale的输出单独计算loss
                _loss_item = model_loss[i](item_pred, targets)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)

            losses = [sum(l) for l in losses]  # 对于每一项，将两个pred的loss加在一起
            loss = losses[0]  # 获取total loss
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 根据梯度更新权值

            # 每更新10个batch打印当前训练信息
            step_count += 1

            if step_count > 0 and step_count % 10 == 0:
                _loss = loss.item()
                duration = float(time.time() - start_time)  # 1个batch的训练用时
                example_per_second = batch_size / duration

                _remain = (total_steps - step_count) * duration  # 剩余训练时间估计
                _m, _s = divmod(_remain, 60)
                _h, _m = divmod(_m, 60)

                lr = optimizer.param_groups[0]['lr']  # 当前学习率
                logger.info(  # 日志打印
                    "epoch [%d]: current_batch = %d/%d, total_iter = %d, loss = %.5f, example/sec = %.3f, lr = %.5f, remain = %d:%02d:%02d"%
                    (epoch, batch_id+1, batch_per_epoch, step_count, _loss, example_per_second, lr, _h, _m, _s)
                )
                tbwriter.add_scalar("lr", lr, step_count)  # 在tensorboard上记录参数
                tbwriter.add_scalar("example/sec", example_per_second, step_count)
                for i, name in enumerate(losses_name):
                    value = _loss if i == 0 else losses[i]  # if (i==0) value=_loss;else value=losses[i]
                    tbwriter.add_scalar(name, value, step_count)  # 记录每一类的loss

        scheduler.step()  # 本epoch结束，更新lr
        if epoch > 4:
            val.get_mAP(epoch=epoch, model=model)  # 计算一次mAP，评估模型性能
        torch.save(model.state_dict(), os.path.join(save_path, "YOLO-Fastest_epoch_{}.pth".format(str(epoch))))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    logger, tbwriter = config_logger(log_dir=config_params["io_params"]["log_path"],
                                     log_name='train_info.log')  # 加载日志模块
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # 设备选择

    # cudnn benchmark
    cudnn.deterministic = False  # 保证运算结果稳定性
    cudnn.benchmark = True

    logger.info("Start....")
    train(config_params, device, tbwriter)  # 训练
