// orb_match.cc

#include <iostream>
#include "ORBVocabulary.h"
#include "ORBextractor.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat ConvertToGray(const cv::Mat &im, int nRGB) {
	cv::Mat retGray;

	if (im.channels() == 3) {
        if (nRGB == 1)  cv::cvtColor(im, retGray, CV_RGB2GRAY);
        else  cv::cvtColor(im, retGray, CV_BGR2GRAY);

	}
	else if (im.channels() == 4) {
        if (nRGB == 1) cv::cvtColor(im, retGray, CV_RGBA2GRAY);
        else  cv::cvtColor(im, retGray, CV_BGRA2GRAY);
    }
    return retGray;
}

int main(int argc, char** argv) {

	cv::Mat rgbL = cv::imread(argv[3]); // /home/jg/Downloads/rgbd_dataset_freiburg1_desk2/rgb/1305031526.671473.png
	cv::Mat rgbR = cv::imread(argv[4]); // /home/jg/Downloads/rgbd_dataset_freiburg1_desk2/rgb/1305031526.871446.png

	ORB_SLAM2::ORBVocabulary *pORBVocabulary = new ORB_SLAM2::ORBVocabulary();
    std::cout << "Read ORBVocabulary from `" << argv[1] << "`." << std::endl;
	bool bVocLoad = pORBVocabulary->loadFromTextFile(argv[1]); // Vocabulary/ORBvoc.txt
	if(!bVocLoad) {
        std::cerr << "Wrong path to vocabulary. " << std::endl;
        std::cerr << "Falied to open at: " << argv << std::endl;
        exit(-1);
    }
    std::cout << "ORBVocabulary loaded." << std::endl;

    cv::FileStorage fSettings(argv[2], cv::FileStorage::READ); // Examples/Monocular/TUM1.yaml

    // int nFeatures = fSettings["ORBextractor.nFeatures"];
    int nFeatures = 100;
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];
    int nRGB = fSettings["Camera.RGB"];

    ORB_SLAM2::ORBextractor* pORBExtractor = new ORB_SLAM2::ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    // (*pORBExtractor)();

    std::vector<cv::KeyPoint> keypointsL;
    std::vector<cv::KeyPoint> keypointsR;
    cv::Mat descriptorsL;
    cv::Mat descriptorsR;

    cv::Mat grayL = ConvertToGray(rgbL, nRGB);
    cv::Mat grayR = ConvertToGray(rgbR, nRGB);

    (*pORBExtractor)(grayL, cv::Mat(), keypointsL, descriptorsL);
    (*pORBExtractor)(grayR, cv::Mat(), keypointsR, descriptorsR);

    std::cout << "descriptorsL.rows : " << descriptorsL.rows
    << "\tdescriptorsL.cols : " << descriptorsL.cols << std::endl;
    std::cout << "descriptorsR.rows : " << descriptorsR.rows
    << "\tdescriptorsR.cols : " << descriptorsR.cols << std::endl;

    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_L2, true);
    matcher.match(descriptorsL, descriptorsR, matches);
    // cv::Mat imgShow;
    // cv::drawKeypoints(rgbL, keypointsL, imgShow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // cv::imshow("Keypoints of Left", imgShow);
    cv::Mat imgMatches;
    cv::drawMatches( rgbL, keypointsL, rgbR, keypointsR, matches, imgMatches );
    cv::imshow( "matches", imgMatches );
    cv::waitKey(0);


	return 0;
}