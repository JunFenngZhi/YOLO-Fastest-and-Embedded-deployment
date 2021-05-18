#include "YOLO_ncnn.h"
#include "utils.h"




int main()
{
    //路径参数
	string resultPath = "D:\\Graduate_Design\\YOLO-Fastest\\test_result";
	string dataPath = "D:\\Graduate_Design\\YOLO-Fastest\\test_data\\infra_red";
	string param_path = "D:\\Graduate_Design\\YOLO-Fastest\\models\\ncnn\\256x320\\YOLO-Fastest_epoch_28-opt.param";
	string bin_path = "D:\\Graduate_Design\\YOLO-Fastest\\models\\ncnn\\256x320\\YOLO-Fastest_epoch_28-opt.bin";

	//YOLO检测参数
	vector<string>class_name = { "carrier", "defender", "destroyer" };
	vector<cv::Scalar> box_color = { cv::Scalar(106, 90, 205),cv::Scalar(199, 97, 20),cv::Scalar(112, 128, 105) };
	int num_class = 3;
	float conf_thres = 0.5;
	float nms_thres = 0.2; 
	vector<int> input_shape = {256,320,1};  //网络输入图片的尺寸(行，列，通道数)
	vector<int> ori_img_shape = {512,640,3};  //数据集原始图片尺寸
	vector<vector<float>>anchors = { {10, 13, 16, 30, 33, 23} ,{150, 75, 100, 100, 75, 150} };

	Detect_YOLO YOLO(param_path.c_str(),bin_path.c_str(), input_shape, num_class, conf_thres, nms_thres, anchors);
	vector<cv::String> file_path;
	cv::glob(dataPath, file_path);  //读取文件夹下的所有图片名
	float avg_time = 0.f;
	int num = file_path.size();
	for (int i = 0; i < file_path.size(); i++) 
	{
		//路径分割
		size_t found = file_path[i].find_last_of("\\"); 
		cv::String path = file_path[i].substr(0, found);
		cv::String file_name = file_path[i].substr(found + 1);

		//读入图片，检测
		cv::Mat img = cv::imread(file_path[i]); //BGR格式读入
		vector<Detect_YOLO::BBoxRect> results;
		float infer_time = 0.f, post_process_time = 0.f;
		YOLO.detect(img, results, infer_time, post_process_time);
		avg_time += (infer_time + post_process_time);

		//print log
		if (results.empty()) {
			printf("image_name:%s -> no targets, infer time:%.2fms, post_process time:%.2fms, total time:%.2fms\n", file_name.c_str(), infer_time,
				post_process_time, infer_time + post_process_time);
		}
		else {
			draw_box(img, results, class_name, box_color, input_shape);
			printf("image_name:%s -> detect finished, infer time:%.2fms, post_process time:%.2fms, total time:%.2fms\n", file_name.c_str(), infer_time,
				post_process_time, infer_time + post_process_time);
		}
		//cv::imwrite(resultPath + "\\result_" + file_name, img);

	}
	printf("avg_time:%.2fms\n", avg_time / num);
	return 0;
}
