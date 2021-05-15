# -*- coding=utf-8 -*-
import torch 
import torch.nn as nn


def group_norm(out_channels):
    assert out_channels % 16 == 0
    num = 32  if out_channels % 32 == 0 else 16
    return nn.GroupNorm(num, out_channels)


norm_dict = {"BN": nn.BatchNorm2d, "GN": group_norm}
norm_func = norm_dict["BN"]

# 这两个都是默认padding的
def conv_norm_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, norm_func=norm_func):
    if padding is None:  # 如果不指定padding值，则根据卷积核大小自行计算padding值（保证卷积后大小不变）
        assert kernel_size % 2 != 0
        padding = (kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                  bias=False, groups=groups),
        norm_func(out_channels),  # Batch_Normalize
        nn.ReLU(inplace=False)  # inplace为True，将会改变输入的数据。即修改原输入
    )


def conv_norm(in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, norm_func=norm_func):
    if padding is None:
        assert kernel_size % 2 != 0
        padding = (kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                  bias=False, groups=groups),
        norm_func(out_channels)
    )


# 反卷积，两倍上采样
def deconv_norm_relu(in_channels, out_channels, kernel_size=2, stride=2, padding=0, groups=1, norm_func=norm_func):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=False, groups=groups),
        norm_func(out_channels),
        nn.ReLU()
    )


# 三个卷积层的残差块
class BasicResBlock(nn.Module):
    def __init__(self, io_channels, inner_channels):
        super(BasicResBlock, self).__init__()

        self.conv1 = conv_norm_relu(io_channels, inner_channels, kernel_size=1, stride=1)
        self.conv2 = conv_norm_relu(inner_channels, inner_channels, kernel_size=3, stride=1, groups=inner_channels)
        self.conv3 = conv_norm(inner_channels, io_channels, kernel_size=1, stride=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.conv3(x)
        out += residual # 残差叠加
        return out


class YoloFastest(nn.Module):
    def __init__(self, io_params):
        super(YoloFastest, self).__init__()
        self.num_cls = io_params["num_cls"]   # 类别数
        self.input_channel = io_params["input_channel"]  # 输入通道数

        num_anchor = io_params["num_anchors"]
        self.num_out = num_anchor * (5 + self.num_cls)  # 5 指的是（x_cen,y_cen,w,h,confidence）

        self.conv0 = conv_norm_relu(self.input_channel, 8, kernel_size=3, stride=2)

        self.conv1_2 = conv_norm_relu(8, 8, kernel_size=1, stride=1)
        self.conv1_3 = conv_norm_relu(8, 8, kernel_size=3, stride=1, groups=8)  # depth-wise conv
        self.conv1_4 = conv_norm(8, 4, kernel_size=1, stride=1)

        self.res1_1 = BasicResBlock(io_channels=4, inner_channels=8)  # 残差块

        self.conv1_8 = conv_norm_relu(4, 24, kernel_size=1, stride=1)
        self.conv1_9 = conv_norm_relu(24, 24, kernel_size=3, stride=2)

        self.conv2_1 = conv_norm(24, 8, kernel_size=1, stride=1)

        self.res2_1 = BasicResBlock(io_channels=8, inner_channels=32)
        self.res2_2 = BasicResBlock(io_channels=8, inner_channels=32)

        self.conv2_2 = conv_norm_relu(8, 32, kernel_size=1, stride=1)
        self.conv2_3 = conv_norm_relu(32, 32, kernel_size=3, stride=2, groups=32)  # depth-wise conv

        self.conv3_1 = conv_norm(32, 8, kernel_size=1, stride=1)

        self.res3_1 = BasicResBlock(io_channels=8, inner_channels=48)
        self.res3_2 = BasicResBlock(io_channels=8, inner_channels=48)

        self.conv3_2 = conv_norm_relu(8, 48, kernel_size=1, stride=1)
        self.conv3_3 = conv_norm_relu(48, 48, kernel_size=3, stride=1, groups=48)  # depth-wise conv
        self.conv3_4 = conv_norm(48, 16, kernel_size=1, stride=1)
        
        self.res3_3 = BasicResBlock(io_channels=16, inner_channels=96)
        self.res3_4 = BasicResBlock(io_channels=16, inner_channels=96)
        self.res3_5 = BasicResBlock(io_channels=16, inner_channels=96)
        self.res3_6 = BasicResBlock(io_channels=16, inner_channels=96)

        self.conv3_5 = conv_norm_relu(16, 96, kernel_size=1, stride=1)
        self.conv3_6 = conv_norm_relu(96, 96, kernel_size=3, stride=2, groups=96)

        self.conv4_1 = conv_norm(96, 24, kernel_size=1, stride=1)  # depth-wise conv
        
        self.res4_1 = BasicResBlock(io_channels=24, inner_channels=136)
        self.res4_2 = BasicResBlock(io_channels=24, inner_channels=136)
        self.res4_3 = BasicResBlock(io_channels=24, inner_channels=136)
        self.res4_4 = BasicResBlock(io_channels=24, inner_channels=136)
        
        self.conv4_2 = conv_norm_relu(24, 136, kernel_size=1, stride=1)  # 出结果
        self.conv4_3 = conv_norm_relu(136, 136, kernel_size=3, stride=2, groups=136)

        self.conv5_1 = conv_norm_relu(136, 48, kernel_size=1, stride=1)
        
        self.res5_1 = BasicResBlock(io_channels=48, inner_channels=224)
        self.res5_2 = BasicResBlock(io_channels=48, inner_channels=224)
        self.res5_3 = BasicResBlock(io_channels=48, inner_channels=224)
        self.res5_4 = BasicResBlock(io_channels=48, inner_channels=224)
        self.res5_5 = BasicResBlock(io_channels=48, inner_channels=224)

        self.conv5_2 = conv_norm_relu(48, 96, kernel_size=1, stride=1)  # upsample
        self.conv5_3 = conv_norm_relu(96, 96, kernel_size=5, stride=1, groups=96)
        self.conv5_4 = conv_norm(96, 128, kernel_size=1, stride=1)
        self.conv5_5 = conv_norm_relu(128, 128, kernel_size=5, stride=1, groups=128)
        self.conv5_6 = conv_norm(128, 128, kernel_size=1, stride=1)

        self.head_5 = nn.Conv2d(128, self.num_out, kernel_size=1, stride=1)

        self.deconv5_1 = deconv_norm_relu(96, 96)

        self.conv4_1_1 = conv_norm_relu(232, 96, kernel_size=1, stride=1)
        self.conv4_1_2 = conv_norm_relu(96, 96, kernel_size=5, stride=1, groups=96)
        self.conv4_1_3 = conv_norm(96, 96, kernel_size=1, stride=1)
        self.conv4_1_4 = conv_norm_relu(96, 96, kernel_size=5, stride=1, groups=96)
        self.conv4_1_5 = conv_norm(96, 96, kernel_size=1, stride=1)

        self.head_4 = nn.Conv2d(96, self.num_out, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv1_4(x)

        x = self.res1_1(x)

        x = self.conv1_8(x)
        x = self.conv1_9(x)
        x = self.conv2_1(x)

        x = self.res2_1(x)
        x = self.res2_2(x)

        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.res3_1(x)
        x = self.res3_2(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        x = self.res3_3(x)
        x = self.res3_4(x)
        x = self.res3_5(x)
        x = self.res3_6(x)
        
        x = self.conv3_5(x)
        x = self.conv3_6(x)

        x = self.conv4_1(x)

        x = self.res4_1(x)
        x = self.res4_2(x)
        x = self.res4_3(x)
        x = self.res4_4(x)

        conv4_2 = self.conv4_2(x)
        x = self.conv4_3(conv4_2)
        
        x = self.conv5_1(x)
        x = self.res5_1(x)
        x = self.res5_2(x)
        x = self.res5_3(x)
        x = self.res5_4(x)
        x = self.res5_5(x)
        
        conv5_2 = self.conv5_2(x)
        x = self.conv5_3(conv5_2)
        x = self.conv5_4(x)
        x = self.conv5_5(x)
        x = self.conv5_6(x)
        
        head_small = self.head_5(x)  # 输出（小特征图适合于大目标检测）

        deconv5_1 = self.deconv5_1(conv5_2)  # 反卷积，扩大特征图尺寸（原版用的是直接resize upsample）
        x = torch.cat((conv4_2, deconv5_1), 1)

        x = self.conv4_1_1(x)
        x = self.conv4_1_2(x)
        x = self.conv4_1_3(x)
        x = self.conv4_1_4(x)
        x = self.conv4_1_5(x)
        head_large = self.head_4(x)  # 融合高层次特征后，做出的预测输出（大特征图适合于小目标检测）

        return head_large, head_small

    def initialize_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif t is nn.BatchNorm2d:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True


class YoloFastest_lite(nn.Module):
    def __init__(self, io_params=None):
        super(YoloFastest_lite, self).__init__()
        self.num_cls = io_params["num_cls"]
        self.input_channel = io_params["input_channel"]

        num_anchor = io_params["num_anchors"] * self.num_cls
        self.num_out = num_anchor * (5 + self.num_cls)

        self.conv0 = conv_norm_relu(self.input_channel, 8, kernel_size=3, stride=2)

        self.conv1_2 = conv_norm_relu(8, 8, kernel_size=1, stride=1)
        self.conv1_3 = conv_norm_relu(8, 8, kernel_size=3, stride=1, groups=8) # 可分离卷积
        self.conv1_4 = conv_norm(8, 4, kernel_size=1, stride=1, groups=1)

        self.res1_1 = BasicResBlock(4, 8)

        self.conv1_8 = conv_norm_relu(4, 24, kernel_size=1, stride=1)
        self.conv1_9 = conv_norm_relu(24, 24, kernel_size=3, stride=2)

        self.conv2_1 = conv_norm(24, 8, kernel_size=1, stride=1)
        
        self.res2_1 = BasicResBlock(8, 32)
        self.res2_2 = BasicResBlock(8, 32)

        self.conv2_2 = conv_norm_relu(8, 32, kernel_size=1, stride=1)
        self.conv2_3 = conv_norm_relu(32, 32, kernel_size=3, stride=2, groups=32)

        self.conv3_1 = conv_norm(32, 8, kernel_size=1, stride=1)

        self.res3_1 = BasicResBlock(8, 48)
        self.res3_2 = BasicResBlock(8, 48)

        self.conv3_2 = conv_norm_relu(8, 48, kernel_size=1, stride=1)
        self.conv3_3 = conv_norm_relu(48, 48, kernel_size=3, stride=1, groups=48)
        self.conv3_4 = conv_norm(48, 16, kernel_size=1, stride=1)
        
        self.res3_3 = BasicResBlock(16, 96)
        self.res3_4 = BasicResBlock(16, 96)
        self.res3_5 = BasicResBlock(16, 96)
        self.res3_6 = BasicResBlock(16, 96)

        self.conv3_5 = conv_norm_relu(16, 96, kernel_size=1, stride=1)
        self.conv3_6 = conv_norm_relu(96, 96, kernel_size=3, stride=2, groups=96)

        self.conv4_1 = conv_norm(96, 24, kernel_size=1, stride=1)
        
        self.res4_1 = BasicResBlock(24, 136)
        self.res4_2 = BasicResBlock(24, 136)
        self.res4_3 = BasicResBlock(24, 136)
        self.res4_4 = BasicResBlock(24, 136)
        
        self.conv4_2 = conv_norm_relu(24, 136, kernel_size=1, stride=1)  # 出结果
        self.conv4_3 = conv_norm_relu(136, 136, kernel_size=3, stride=2, groups=136)

        self.conv5_1 = conv_norm_relu(136, 48, kernel_size=1, stride=1)
        
        self.res5_1 = BasicResBlock(48, 224)
        self.res5_2 = BasicResBlock(48, 224)
        self.res5_3 = BasicResBlock(48, 224)
        self.res5_4 = BasicResBlock(48, 224)
        self.res5_5 = BasicResBlock(48, 224)

        self.conv5_2 = conv_norm_relu(48, 96, kernel_size=1, stride=1)  # upsample
        self.conv5_3 = conv_norm_relu(96, 96, kernel_size=5, stride=1, groups=96)
        self.conv5_4 = conv_norm(96, 128, kernel_size=1, stride=1)
        self.conv5_5 = conv_norm_relu(128, 128, kernel_size=5, stride=1, groups=128)
        self.conv5_6 = conv_norm(128, 128, kernel_size=1, stride=1)

        self.head_5 = nn.Conv2d(128, self.num_out, kernel_size=1, stride=1)

        self.deconv5_1 = deconv_norm_relu(96, 96)

        self.conv4_1_1 = conv_norm_relu(232, 96, kernel_size=1, stride=1)
        self.conv4_1_2 = conv_norm_relu(96, 96, kernel_size=5, stride=1, groups=96)
        self.conv4_1_3 = conv_norm(96, 96, kernel_size=1, stride=1)
        self.conv4_1_4 = conv_norm_relu(96, 96, kernel_size=5, stride=1, groups=96)
        self.conv4_1_5 = conv_norm(96, 96, kernel_size=1, stride=1)

        self.head_4 = nn.Conv2d(96, self.num_out, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv1_4(x)

        x = self.res1_1(x)

        x = self.conv1_8(x)
        x = self.conv1_9(x)
        x = self.conv2_1(x)

        x = self.res2_1(x)
        x = self.res2_2(x)

        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.res3_1(x)
        x = self.res3_2(x)
        x = self.conv3_2(x)
        x = self.conv3_4(x)

        x = self.res3_3(x)
        x = self.res3_4(x)
        x = self.res3_5(x)
        x = self.res3_6(x)
        
        x = self.conv3_5(x)
        x = self.conv3_6(x)

        x = self.conv4_1(x)

        x = self.res4_1(x)
        x = self.res4_2(x)
        x = self.res4_3(x)
        x = self.res4_4(x)

        conv4_2 = self.conv4_2(x)
        x = self.conv4_3(conv4_2)
        
        x = self.conv5_1(x)
        x = self.res5_1(x)
        x = self.res5_2(x)
        x = self.res5_3(x)
        x = self.res5_4(x)
        x = self.res5_5(x)
        
        conv5_2 = self.conv5_2(x)
        x = self.conv5_3(conv5_2)
        x = self.conv5_4(x)
        x = self.conv5_5(x)
        x = self.conv5_6(x)
        
        head_5 = self.head_5(x)

        return head_5

    def initialize_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                #t.weight.data.normal_(0.0, 0.02) 
                #pass
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True


'''
conv2D()中groups参数含义：分组卷积。C个输入通道，分成G组，每组负责C/G个，所以卷积核尺寸为C/G * K * K。输出N个通道，分成G组，每组负责N/G个特征图的输出,所以一组有N/G个滤波器。
                    不同组间复用相同的滤波器，所以分组可以减少卷积层参数数量。要求：C,N都能被G整除。
depth_wise conv: G = N = C。一个输入通道为一组，形成N组。卷积核尺寸为1 * K * K。相当于用同样的卷积核重复对所有输入特征图作卷积，生成各个通道的输出输出。
'''

if __name__ == "__main__":

    print(torch.cuda.is_available())
    input = torch.randn(1, 1, 512, 640).cuda()

    io_params = {
        "num_cls" : 1,
        "anchors" :  [
                    [[12, 18],  [37, 49],  [52,132]],
                    [[115, 73], [119,199], [242,238]]
                    ],
        "input_channel": 1}

    net = YoloFastest_lite(io_params).cuda()
    net.eval()
    
    head_5 = net(input)
    
    print (head_5.shape)