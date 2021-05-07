#include "utils.h"




void pretty_print(const ncnn::Mat& m)
{
	for (int q = 0; q < m.c; q++)
	{
		const float* ptr = m.channel(q);
		for (int y = 0; y < m.h; y++)
		{
			for (int x = 0; x < m.w; x++)
			{
				printf("%f ", ptr[x]);
			}
			ptr += m.w;
			printf("\n");
		}
		printf("------------------------\n");
	}
}

float sigmoid(float x)
{
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

void draw_box(cv::Mat& ori_img, const vector<Detect_YOLO::BBoxRect>& results, 
	const vector<string>&class_name, const vector<cv::Scalar>& color)
{
	for (auto it = results.begin(); it != results.end(); it++)
	{
		int xmin = it->xmin;
		int ymin = it->ymin;
		int xmax = it->xmax;
		int ymax = it->ymax;

		//处理坐标越界
		if (xmin < 0) xmin = 0;
		if (ymin < 0) ymin = 0;
		if (xmax < 0) xmax = 0;
		if (ymax < 0) ymax = 0;
		if (xmin > ori_img.cols) xmin = ori_img.cols;
		if (ymin > ori_img.rows) ymin = ori_img.rows;
		if (xmax > ori_img.cols) xmax = ori_img.cols;
		if (ymax > ori_img.rows) ymax = ori_img.rows;

		//draw box
		cv::rectangle(ori_img, cv::Point(xmin, ymin), cv::Point(xmax, ymax), color[it->label], 3, 1, 0);

		//draw label
		char text[64];
		sprintf(text, "%s %.2f", class_name[it->label].c_str(), it->conf * it->cls_score);
		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		cv::putText(ori_img, text, cv::Point(xmin, ymin - label_size.height + 4),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255),1);
	}

	//cv::imshow("test", ori_img);
	//cv::waitKey(0);
}
