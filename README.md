# YOLO-Fastest & Embedded deployment
## Introduction
*   YOLO-Fastest 是现在已知开源最快的最轻量的改进版yolo通用目标检测算法。它基于yolov3，具有轻量化、简单易移植、推理快等特点。其设计初衷是为了打破算力的瓶颈，能在更多的低成本的边缘端设备实时运行目标检测算法。YOLO-Fastest原版是基于darknet框架训练的，对新手而言较难安装使用。本项目优化了原有的pytorch版YOLO-Fastest并加以改进，更方便大家训练部署自己的网络。   
 
*   本项目使用两种方法将YOLO-Fastest部署到嵌入式设备上，并对比了这两种部署方法的性能。  
1、 基于ncnn推理框架部署。ncnn是一个为嵌入式端极致优化的高性能神经网络前向计算框架。因为本项目使用的是pytorch版本下的模型，转换模型格式后无法直接使用ncnn中自带的yolo后处理层(yolov3detectionoutput),所以本项目参照ncnn源码自行实现了yolo后处理。  
2、基于NPU模块部署。NPU模块是专门用于神经网络推理计算的计算单元，通常推理速度会远高于CPU。本项目使用RK3399proD开发板上搭载的NPU模块并通过配套的rknn-toolkit API进行网络部署。  

*   本项目使用的数据集为红外光海面背景下小目标舰船模拟数据集。

*   为了对比，本项目训练了两个版本的模型。第一个模型网络输入图片的分辨率是640x512,第二个是320x256。结果表明，适当降低分辨率可以在保证性能的同时有效提升检测速度。  

## Performance
|部署方式|运行设备|输入图像分辨率|mAP|检测率|推理时间(ms)|后处理时间(ms)|检测时间(ms)|
|---|---|---|---|---|---|---|---|
|基于NCNN框架部署|双核Cortex-A72|640*512|?|77.1%|139.56|0.17|139.73|
|||320*256|?|85.7%|52.90|0.1|53.00|
|基于NPU部署|RK3399proD搭载的NPU|640*512|?|94.3%|110.23|25.64|135.87|
|||320*256|?|97.1%|36.68|6.89|43.56|
|PC端直接运行|Intel i5-7200U|640*512|0.897|94.3%|228.89|3.1|231.99|
|||320*256|0.859|97.1%|53.90|3.9|57.80|


## User Guide
*  进行src/model_training目录，修改_config.py中的参数，然后运行train.py即可开始训练网络。注意：需要在train.py的主函数中指定调用的GPU。   

*  网络部署整体流程如下：模型格式转换——>推理环境参数配置——>模型推理——>推理结果后处理。具体各阶段的操作，可以查阅相关文档或在相关扣扣群中询问。两种部署方法的代码在src_model_deployment目录中。

## Environment
* Hardware:  
deploy: RK3399proD开发板(Debian 10.1系统).  
train: one NVIDIA TITAN GPU for nearly 3 hours.   
* Package:  python3.6, pytorch 1.2.0, rknn-toolkit 1.4.0, ncnn 20210322_release

## Thanks
https://github.com/dog-qiuqiu/Yolo-Fastest   
https://github.com/Tencent/ncnn   
http://t.rock-chips.com/wiki.php?mod=view&id=25   
