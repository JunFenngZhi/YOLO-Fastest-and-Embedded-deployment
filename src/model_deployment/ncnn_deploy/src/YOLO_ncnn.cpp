#include "utils.h"
#include "YOLO_ncnn.h"
#include <omp.h>


Detect_YOLO::Detect_YOLO(const char* paramPath, const char* binPath, const vector<int>&img_size, 
	int num_class, float conf_thres, float nms_thres, const vector<vector<float>>&anchors):conf_thres(conf_thres),
	img_size(img_size),num_class(num_class),anchors(anchors),nms_thres(nms_thres)
{
	net.opt.use_packing_layout = true;
	net.opt.use_bf16_storage = true;
	net.opt.use_fp16_arithmetic = true;
	net.load_param(paramPath); //必须先读取param
	net.load_model(binPath);
	num_anchors = anchors[0].size() / 2;
}

Detect_YOLO::~Detect_YOLO()
{
}


void Detect_YOLO::detect(const cv::Mat& ori_img, vector<BBoxRect>& all_bbox_rects,
	float& infer_time, float& post_process_time) const
{
	//网络推理模式设置
	ncnn::Extractor extractor = net.create_extractor();
	extractor.set_light_mode(true);  //轻量运行模式，网络不会保存中间结果，只会输出最后结果
	extractor.set_num_threads(NUMTHREADS);    //设置运行线程数

	//输入图片预处理
	cv::Mat img_grey;
	cvtColor(ori_img, img_grey, CV_BGR2GRAY);
	ncnn::Mat in = ncnn::Mat::from_pixels(img_grey.data, ncnn::Mat::PIXEL_GRAY, img_grey.cols, img_grey.rows);
	float mean[1] = { 128.f };
	float norm[1] = { 1 / 255.f };
	in.substract_mean_normalize(mean, norm);  //输入图像减去均值，并乘以归一化系数。

	//输入
	double start = ncnn::get_current_time();
	extractor.input("data", in);

	//输出预测结果
	ncnn::Mat head_large, head_small;
	extractor.extract("head_large", head_large);
	extractor.extract("head_small", head_small);
	vector<ncnn::Mat> pred = { head_large, head_small };
	double time_mark = ncnn::get_current_time();
	infer_time = time_mark - start;

	//后处理
	decode_bbox(pred, all_bbox_rects);
	vector<vector<BBoxRect>>bbox_rects_class(num_class); //检测结果分类别存储
	for (int i = 0; i < all_bbox_rects.size(); i++)
	{
		int cls = all_bbox_rects[i].label;
		bbox_rects_class[cls].push_back(all_bbox_rects[i]);
	}
	all_bbox_rects.clear();
	for (int i = 0; i < num_class; i++) //按类别进行NMS
	{
		if (bbox_rects_class[i].empty())
			continue;

		qsort_descent(bbox_rects_class[i], 0, static_cast<int>(bbox_rects_class[i].size() - 1));
		non_maxium_supression(bbox_rects_class[i]);
		all_bbox_rects.insert(all_bbox_rects.end(), bbox_rects_class[i].begin(), bbox_rects_class[i].end()); //各类别检测结果合并
	}
	post_process_time = ncnn::get_current_time() - time_mark;
}

int Detect_YOLO::decode_bbox(const vector<ncnn::Mat>& pred, vector<BBoxRect>& all_bbox_rects) const
{
	for (int b = 0; b < pred.size(); b++) //每次处理一个输出头
	{
		vector<vector<BBoxRect> > all_bbox_rects_head(num_anchors);  //保存这个头的输出的所有bbox，按类型分开存储

		int w = pred[b].w;  
		int h = pred[b].h;
		int channel = pred[b].c;
		float scale_h = img_size[0] / h; //高度上的输出特征图缩小倍数
		float scale_w = img_size[1] / w; //宽度上的输出特征图缩小倍数
		
		//check output format
		const int channel_per_box = channel / num_anchors;
		if (channel_per_box != 4 + 1 + num_class) {
			cout << "the number of channel per box is wrong!" << endl;  //检查输出
			return -1;
		}
			
		//按anchors类别分开遍历
		#pragma omp parallel for num_threads(NUMTHREADS) //OPENmp并行加速
		for (int pp = 0; pp < num_anchors; pp++)  
		{	
			//划分通道，便于单独访问
			int p = pp * channel_per_box; 
			const float* xptr = pred[b].channel(p);   
			const float* yptr = pred[b].channel(p + 1);
			const float* wptr = pred[b].channel(p + 2);
			const float* hptr = pred[b].channel(p + 3);
			const float* conf_ptr = pred[b].channel(p + 4);
			const ncnn::Mat cls_scores = pred[b].channel_range(p + 5, num_class);

			for (int i = 0; i < h; i++)
			{
				for (int j = 0; j < w; j++) 
				{
					float conf = sigmoid(conf_ptr[0]);  //参考yolov2
					if (conf > conf_thres)   //使用置信度阈值筛选掉无效的bbox
					{  
						//确定bbox预测的类别
						int cls_index = 0;
						float max_score = -FLT_MAX;
						for (int c = 0; c < num_class; c++) 
						{
							float score = sigmoid(cls_scores.channel(c).row(i)[j]);  //yolov3处理方法，不使用softmax
							if (score > max_score) {
								max_score = score;
								cls_index = c;
							}
						}

						//decode & recover to real scale
						float bbox_x = (j + sigmoid(xptr[0]))*scale_w;
						float bbox_y = (i + sigmoid(yptr[0]))*scale_h;
						float bbox_w = static_cast<float>(exp(wptr[0]) * anchors[b][pp * 2] / scale_w)*scale_w;
						float bbox_h = static_cast<float>(exp(hptr[0]) * anchors[b][pp * 2 + 1] / scale_h)*scale_h;

						//transfer format(x_cen,y_cen,w,h)->(x1,y1,x2,y2)
						int bbox_xmin = static_cast<int>(bbox_x - bbox_w * 0.5f);
						int bbox_ymin = static_cast<int>(bbox_y - bbox_h * 0.5f);
						int bbox_xmax = static_cast<int>(bbox_x + bbox_w * 0.5f);
						int bbox_ymax = static_cast<int>(bbox_y + bbox_h * 0.5f);

						BBoxRect temp = { max_score,conf,bbox_xmin,bbox_ymin,bbox_xmax,bbox_ymax,cls_index };
						all_bbox_rects_head[pp].push_back(temp);

					}
					
					xptr++;
					yptr++;
					wptr++;
					hptr++;
					conf_ptr++;
				}
			}
		}

		//检测结果合并
		for (int i = 0; i < num_anchors; i++)
		{
			all_bbox_rects.insert(all_bbox_rects.end(),all_bbox_rects_head[i].begin(),all_bbox_rects_head[i].end());
		}
		return 0;
	}
}

void Detect_YOLO::qsort_descent(vector<BBoxRect>& datas, int left, int right) const
{
	int i = left;
	int j = right;
	float p = datas[(left + right) / 2].conf;

	while (i <= j)
	{
		while (datas[i].conf > p)
			i++;

		while (datas[j].conf < p)
			j--;

		if (i <= j)
		{
			// swap
			std::swap(datas[i], datas[j]);

			i++;
			j--;
		}
	}

	if (left < j)
		qsort_descent(datas, left, j);

	if (i < right)
		qsort_descent(datas, i, right);
}

void Detect_YOLO::non_maxium_supression(vector<BBoxRect>& bbox_rects) const
{
	vector<BBoxRect> results;  //存储NMS后的bbox	
	while (!bbox_rects.empty())
	{
		results.push_back(bbox_rects[0]); //选取conf最大的bbox为检测结果
		if (bbox_rects.size() == 1)  
			break;
		bbox_rects.erase(bbox_rects.begin()); //去掉已经添加的结果
		for (auto it = bbox_rects.begin(); it != bbox_rects.end();) //计算指定bbox和剩余所有bbox的IOU,进行筛选
		{
			float iou = cal_IOU(results.back(), *it);
			if (iou > nms_thres)
				it = bbox_rects.erase(it);
			else
				it++;
		}
	}
	bbox_rects = results;
}

float Detect_YOLO::cal_IOU(const BBoxRect& box_1, const BBoxRect& box_2) const
{
	float inter_area = 0.f,union_area = 0.f;
	float inter_w = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin);
	float inter_h = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin);

	if (inter_w > 0 && inter_h > 0)
		inter_area = inter_h * inter_w;// 交集

	union_area = (box_1.xmax - box_1.xmin)*(box_1.ymax - box_1.ymin)+
		(box_2.xmax - box_2.xmin)*(box_2.ymax - box_2.ymin)-inter_area; //并集

	return inter_area / union_area;

}
