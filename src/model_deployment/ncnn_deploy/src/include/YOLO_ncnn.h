#pragma once
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <iostream>

#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "gpu.h"

using namespace std;


//cpu num threads
#define NUMTHREADS 8

class Detect_YOLO
{
public:
	struct BBoxRect
	{
		float cls_score; //类别概率
		float conf; //目标置信度
		int xmin;
		int ymin;
		int xmax;
		int ymax;
		int label;  //类别标签
	};

public:
	Detect_YOLO(const char* paramPath, const char* binPath, const vector<int>&img_size,
		int num_class, float conf_thres, float nms_thres, const vector<vector<float>>&anchors);
	~Detect_YOLO();
	void detect(const cv::Mat& ori_img, vector<BBoxRect>& all_bbox_rects, 
		float& infer_time, float& post_process_time) const;  //输入待检测图片，返回处理后的检测结果
	
public:
	ncnn::Net net;
	vector<vector<float>>anchors;//每个头对应一行，每行排列为（w_1,h_1,w_2,h_2...）
	vector<int> img_size; //(w,h,c)
	float conf_thres;
	float nms_thres;
	int num_anchors;
	int num_class;
	
private:
	int decode_bbox(const vector<ncnn::Mat>& pred, vector<BBoxRect>& all_bbox_rects) const;//对检测结果进行解码，恢复为真实坐标
	void qsort_descent(vector<BBoxRect>& datas, int left, int right) const; //根据conf降序排列
	void non_maxium_supression(vector<BBoxRect>& bbox_rects) const; //对已排序的bbox进行非极大值抑制
	float cal_IOU(const BBoxRect& box_1, const BBoxRect& box_2) const;

};

