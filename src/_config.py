config_params = {
        "io_params": {
            "save_path": '/home/gpu/zhijunfeng/YOLO-Fastest/models/',
            "log_path": '/home/gpu/zhijunfeng/YOLO-Fastest/logs/',
            "anchors": [
                    [[150, 75], [100, 100], [75, 150]],
                    [[300, 150], [200, 200], [150, 300]],
                    [[10, 13], [16, 30], [33, 23]]
                         ],  # 每组三个，对应三个不同的尺度  每个指的是【w,h】
            "input_channel": 1,
            "input_size": [512, 640, 1],  # 网络的输入图像尺寸。假如原始图片不符合，则要变成该尺寸 【行，列，通道数】
            "input_tensor_shape": (1, 1, 512, 640),
            "num_cls": 3,  # 类别数
            "num_anchors": 3,
            "anchor_mask": [[0, 1, 2], [3, 4, 5]],
            "strides": [16, 32],  # 指的是两个尺度下的放缩倍数。即最后特征图中一个像素对应原图多少像素。
            "conf_thre": 0.5,  # NMS中使用
            "nms_thre": 0.7,    # NMS  结合数据集特点可以适当增大NMS_thre
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
            "total_epochs": 20,
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

