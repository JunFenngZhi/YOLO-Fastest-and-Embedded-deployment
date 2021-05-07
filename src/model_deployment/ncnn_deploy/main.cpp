#include "YOLO_ncnn.h"
#include "utils.h"



int main()
{
    //YOLO检测参数
	string resultPath = "/home/toybrick/ncnn_project/test_result";
	string dataPath = "/home/toybrick/ncnn_project/test_data";
	string param_path = "/home/toybrick/ncnn_project/model/YOLO-Fastest_epoch_27-opt.param";
	string bin_path = "/home/toybrick/ncnn_project/model/YOLO-Fastest_epoch_27-opt.bin";
	vector<string>class_name = { "carrier", "defender", "destroyer" };
	vector<cv::Scalar> box_color = { cv::Scalar(106, 90, 205),cv::Scalar(199, 97, 20),cv::Scalar(112, 128, 105) };
	int num_class = 3;
	float conf_thres = 0.5;
	float nms_thres = 0.2;
	vector<int> img_size = { 512, 640, 1 };
	vector<vector<float>>anchors = { {150, 75, 100, 100, 75, 150}, {300, 150, 200, 200, 150, 300} };

	Detect_YOLO YOLO(param_path.c_str(),bin_path.c_str(), img_size, num_class, conf_thres, nms_thres, anchors);

	vector<cv::String> file_path;
	cv::glob(dataPath, file_path);  //读取文件夹下的所有图片名
	for (int i = 0; i < file_path.size(); i++) 
	{
		//路径分割
		size_t found = file_path[i].find_last_of("/"); 
		cv::String path = file_path[i].substr(0, found);
		cv::String file_name = file_path[i].substr(found + 1);

		//读入图片，检测
		cv::Mat img = cv::imread(file_path[i], CV_LOAD_IMAGE_COLOR); //BGR格式读入
		vector<Detect_YOLO::BBoxRect> results;
		float infer_time = 0.f, post_process_time = 0.f;
		YOLO.detect(img, results,infer_time, post_process_time);

		//print log
		if (results.empty()) {
			printf("image_name:%s -> no targets, infer time:%.2fms, post_process time:%.2fms, total time:%.2fms\n", file_name.c_str(), infer_time,
				post_process_time, infer_time + post_process_time);
		}
		else {
			draw_box(img, results,class_name, box_color);
			printf("image_name:%s -> detect finished, infer time:%.2fms, post_process time:%.2fms, total time:%.2fms\n", file_name.c_str(), infer_time,
				post_process_time, infer_time + post_process_time);
		}
		cv::imwrite(resultPath + "/result_" + file_name, img);

	}

	return 0;
}
