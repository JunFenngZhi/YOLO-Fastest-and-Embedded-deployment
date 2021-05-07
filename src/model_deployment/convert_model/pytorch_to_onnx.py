import torch
import torch.onnx
from model_training.model.yolo_fastest import YoloFastest
from model_training._config import config_params


# An instance of your model
net = YoloFastest(io_params=config_params["io_params"]).eval()
net_param = torch.load("E:\Graduate_Design\YOLO-Fastest\models\pytorch\YOLO-Fastest_epoch_27.pth", map_location="cpu")
net.load_state_dict(net_param)  # 导入模型参数

# An example input you would normally provide to your model's forward() method
input_tensor_shape = config_params["io_params"]["input_tensor_shape"]
x = torch.rand(input_tensor_shape[0], input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3])

# Export the model
torch_out = torch.onnx._export(net, x, "YOLO-Fastest_epoch_27.onnx", export_params=True)

'''
下一步使用onnx-simplifier优化导出的onnx模型。（https://github.com/daquexian/onnx-simplifier）
推荐在linux系统下安装运行。
最后使用convertmodel.com将onnx模型转化为其它格式
'''