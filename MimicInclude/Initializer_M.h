#ifndef INITIALIZER_M_H
#define INITIALIZER_M_H

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM2_M {
	class Initializer {
	public:
		Initializer(const std::vector<cv::KeyPoint> &key_point_1_vec,
		 const std::vector<cv::KeyPoint> &key_point_2_vec,
		 const std::vector<cv::DMatch> &matches_1_2_vec,
		 int max_iterations_n,
		 float sigma_f);

		bool Initialize();

	private:

		void FindHomography(std::vector<bool> &match_inliers_vec, float &score, cv::Mat &H21);

		std::vector<cv::KeyPoint> key_point_1_vec_;
		std::vector<cv::KeyPoint> key_point_2_vec_;

		std::vector<cv::DMatch> matches_1_2_vec_;

		int max_iterations_n_;
		float sigma_f_;

		std::vector<std::vector<size_t>> ransac_vec;
	}
}

#endif // INITIALIZER_M_H