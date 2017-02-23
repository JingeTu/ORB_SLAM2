
#ifndef UTIL_M_H
#define UTIL_M_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM2_M{

	struct ORB_PARAM
	{
		int features_n;
		float scale_factor_f;
		int levels_n;
		int ini_th_FAST_n;
		int min_th_FAST_n;
		int rgb_n;
	};

	namespace Util{
		cv::Mat ConvertToGray(const cv::Mat &im, int nRGB);
		ORB_PARAM ReadORBParameters(std::string file_path);
		cv::Mat ReadCameraMatrix(std::string file_path);
		cv::Mat ReadDistortCoefficient(std::string file_path);

	}
}

#endif