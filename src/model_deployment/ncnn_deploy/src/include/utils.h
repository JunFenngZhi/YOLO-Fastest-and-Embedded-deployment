#pragma once
#include "YOLO_ncnn.h"


//显示ncnn::Mat中的内容
void pretty_print(const ncnn::Mat& m);

//sigmoid激活函数
float sigmoid(float x);

//画出检测结果
void draw_box(cv::Mat& ori_img, const vector<Detect_YOLO::BBoxRect>& results, 
	const vector<string>&class_name, const vector<cv::Scalar>& color, const vector<int>& input_shape);