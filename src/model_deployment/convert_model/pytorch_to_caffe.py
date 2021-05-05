import sys
import torch
from torch.autograd import Variable
from torchvision.models import resnet
from model_training.utils import pytorch_to_caffe
from model_training.model.yolo_fastest import YoloFastest, YoloFastest_lite
from model_training._config import config_params

if __name__ == '__main__':
    name = 'YoloFastest'
    net = YoloFastest(config_params["io_params"])
    net.eval()
    ckpt = torch.load("/home/hjh-rog/0_workspace/6_detect/yolo-fastest/pretrained/epoch_5.pth")
    net.load_state_dict(ckpt)
    input = torch.ones([1, 1, 512, 640])
    pytorch_to_caffe.trans_net(net, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))