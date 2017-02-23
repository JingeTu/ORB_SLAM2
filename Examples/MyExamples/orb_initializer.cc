#include "Initializer_M.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {

	cv::Mat rawL = cv::imread(argv[3]);
	cv::Mat rawR = cv::imread(argv[4]);

    cv::FileStorage fSettings(argv[2], cv::FileStorage::READ); // Examples/Monocular/TUM1.yaml

    ORB_PARAM orb_parameters = ReadORBParameters(argv[2]);

    // Camera Parameters
    cv::Mat K = ReadCameraMatrix(argv[2]);

    cv::Mat DistCoef = ReadDistortCoefficient(argv[2]);

    // Undistort Images.
    cv::Mat udL;
    cv::undistort(rawL, udL, K, DistCoef, K);
    cv::Mat udR;
    cv::undistort(rawR, udR, K, DistCoef, K);

    cv::Mat grayL = ORB_SLAM2_M::Util::ConvertToGray(udL, nRGB);
    cv::Mat grayR = ORB_SLAM2_M::Util::ConvertToGray(udR, nRGB);

    std::vector<cv::KeyPoint> keypoints_left_vec;
    std::vector<cv::KeyPoint> keypoints_right_vec;
    cv::Mat descriptors_left_mat;
    cv::Mat descriptors_right_mat;

    ORB_SLAM2_M::ORBextractor* pORBExtractor = new ORB_SLAM2_M::ORBextractor(orb_parameters.features_n,
        orb_parameters.scale_factor_f,orb_parameters.levels_n,
        orb_parameters.ini_th_FAST_n,orb_parameters.min_th_FAST_n);

    (*pORBExtractor)(grayL, cv::Mat(), keypoints_left_vec, descriptors_left_mat);
    (*pORBExtractor)(grayR, cv::Mat(), keypoints_right_vec, descriptors_right_mat);

    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_L2, true);
    matcher.match(descriptors_left_mat, descriptors_right_mat, matches); // `queryIdx` for descriptorsL, `trainIdx` for descriptorsR.

    std::vector<cv::DMatch> inlier_matches;
    ORB_SLAM2_M::Util::MyRANSACMatch(K, keypoints_left_vec, keypoints_right_vec, matches, 0.5, inlier_matches);

    ORB_SLAM2_M::Initializer* initializer = new ORB_SLAM2_M::Initializer(keypoints_left_vec, keypoints_right_vec, inlier_matches);

    initializer->initialize();





	return 0;
}