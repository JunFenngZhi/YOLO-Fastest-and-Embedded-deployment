config_params = {
        "io_params": {
            "save_path": '/home/gpu/zhijunfeng/YOLO-Fastest/models/',  # 模型存储路径
            "log_path": '/home/gpu/zhijunfeng/YOLO-Fastest/logs/',     # 程序日志存储路径
            "anchors": [
                    [[10, 13], [16, 30], [33, 23]],
                    [[150, 75], [100, 100], [75, 150]],
                    [[300, 150], [200, 200], [150, 300]],
                         ],  # 每组三个，对应三个不同的尺度  每个指的是【w,h】  256x320用前两个，512x640用后两个
            "input_channel": 1,
            "input_shape": [256, 320, 1],  # 网络的输入图像尺寸(行和列必须为32的倍数)  【行，列，通道数】 网络在输入图坐标系预测目标，最后再把结果恢复到原始图坐标系下
            "origin_img_shape": [512, 640, 3],  # 数据集中原始输入图片尺寸  【行，列，通道数】
            "input_tensor_shape": (1, 1, 512, 640),
            "num_cls": 3,  # 类别数
            "num_anchors": 3,
            "anchor_mask": [[0, 1, 2], [3, 4, 5]],
            "strides": [16, 32],  # 指的是两个尺度下的放缩倍数。即输出特征图的长和宽相对输入缩小多少倍。
            "conf_thre": 0.5,  # 后处理中使用，去除无效检测结果
            "nms_thre": 0.2,    # NMS  结合数据集特点可以适当减少NMS_thre。阈值越大，说明能容忍的重叠程度更高
            "class_names": ['carrier', 'defender', 'destroyer']  # 数据集中类别信息
        },

        "augment_params": {
            "train_dataset_dir": "/home/gpu/zhijunfeng/Data/train_data/",  # 训练集存储路径
            "val_dataset_dir": "/home/gpu/zhijunfeng/Data/val_data/",  # 验证集存储路径
            "degrees": 0.0,    # image rotation (+/- deg)
            "translate": 0.0,  # image translation (+/- fraction)
            "scale": 1.0,      # image scale (+/- gain)
            "shear": 0.0,      # image shear (+/- deg)
            # image perspective (+/- fraction), range 0-0.001
            "perspective": 0.0,
            "flipud": 0.0,     # image flip up-down (probability)
            "fliplr": 0.5,     # image flip left-right (probability)
            "mixup": 0.0,      # image mixup (probability)
            "gussian_filter": 0.3
        },

        "train_params": {
            "pretrained_pth": "/home/hjh-rog/0_workspace/6_detect/yolo-fastest/pretrained/epoch_0.pt",
            "total_epochs": 30,
            "batch_size": 16,
            # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lr0": 0.001,
            "momentum": 0.937,          # SGD momentum/Adam beta1
            "weight_decay": 0.0005,
            "branch_weight": [1.0, 1.0],
            "IOU_loss_thre": 0.5,  # 在loss计算时，通过目标形状和anchor box形状的相似度，提前去除那些不可能包含目标的预测结果
            "IOU_val_thre": 0.5,  # 在验证计算map时，预测位置和真实位置匹配阈值

        },
    }

