#include <iostream>
#include "ORBVocabulary_M.h"
#include "ORBextractor_M.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>

#include <cstdlib>
#include <ctime>
#include <limits>

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

size_t *Permutation(size_t N) {
    size_t * a = new size_t[N];
    for (size_t i = 0; i < N; ++i)
        a[i] = i;
    for (size_t i = 0; i < N; ++i)
        std::swap(a[i], a[std::rand() % N]);
    return a;
}

cv::Mat SolveF(const cv::Mat &A) {
    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt);
    cv::Mat Frls = vt.row(vt.rows - 1).t();
    cv::Mat Frl(3, 3, CV_32F);
    cv::Mat Fc1 = Frls.rowRange(0, 3);
    Fc1.copyTo(Frl.col(0));
    cv::Mat Fc2 = Frls.rowRange(3, 6);
    Fc2.copyTo(Frl.col(1));
    cv::Mat Fc3 = Frls.rowRange(6, 9);
    Fc3.copyTo(Frl.col(2));
    return Frl;
}

void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat & R2, cv::Mat &t) {
    cv::Mat u, w, vt;
    cv::SVD::compute(E, w, u, vt);

    u.col(2).copyTo(t);
    t = t / cv::norm(t);

    cv::Mat W(3, 3, CV_32F, cv::Scalar(0));
    W.at<float>(0, 1) = -1;
    W.at<float>(1, 0) = 1;
    W.at<float>(2, 2) = 1;

    R1 = u * W * vt;
    if (cv::determinant(R1) < 0)
        R1 = -R1;

    R2 = u * W.t() * vt;
    if (cv::determinant(R2) < 0)
        R2 = -R2;
}

cv::Mat LinearTriangulation(const cv::Mat &x1, const cv::Mat &x2, const cv::Mat &P1, const cv::Mat &P2) {
    cv::Mat A(4,4,CV_32F);
    A.row(0) = x1.at<float>(0)*P1.row(2)-P1.row(0);
    A.row(1) = x1.at<float>(1)*P1.row(2)-P1.row(1);
    A.row(2) = x2.at<float>(0)*P2.row(2)-P2.row(0);
    A.row(3) = x2.at<float>(1)*P2.row(2)-P2.row(1);
    cv::Mat w,u,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    cv::Mat X = vt.row(3).t();
    return X;
}

int CheckRT(const cv::Mat &K, const cv::Mat &R, const cv::Mat &t, const std::vector<cv::KeyPoint> &keypointsL, const std::vector<cv::KeyPoint> &keypointsR, const std::vector<cv::DMatch> &inlierMatches) {
    int inlierNum = 0;

    for (int i = 0, iend = inlierMatches.size(); i < iend; ++i) {
        cv::Mat xl = (cv::Mat_<float>(3, 1) << keypointsL[inlierMatches[i].queryIdx].pt.x, keypointsL[inlierMatches[i].queryIdx].pt.y, 1.0f);
        cv::Mat xr = (cv::Mat_<float>(3, 1) << keypointsR[inlierMatches[i].trainIdx].pt.x, keypointsR[inlierMatches[i].trainIdx].pt.y, 1.0f);
        cv::Mat Pl(3, 4, CV_32F, cv::Scalar(0));
        Pl.at<float>(0, 0) = 1.0f;Pl.at<float>(1, 1) = 1.0f;Pl.at<float>(2, 2) = 1.0f;
        Pl = K * Pl;
        cv::Mat Pr(3, 4, CV_32F);
        R.copyTo(Pr.rowRange(0, 3).colRange(0, 3));
        t.copyTo(Pr.rowRange(0, 3).col(3));
        Pr = K * Pr;
        cv::Mat Xl = LinearTriangulation(xl, xr, Pl, Pr);
        Xl = Xl.rowRange(0,3)/Xl.at<float>(3);
        cv::Mat Xr = R * Xl + t;
        if (Xl.at<float>(2) > 0 && Xr.at<float>(2) > 0)
            inlierNum ++;
    }

    return inlierNum;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr MyRANSACMatch(const cv::Mat &K, const std::vector<cv::KeyPoint> &keypointsL, const std::vector<cv::KeyPoint> &keypointsR,
 const std::vector<cv::DMatch> &matches, const float outlierRatio, std::vector<cv::DMatch> &inlierMatches) {

    std::srand(std::time(0));

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

    float fOutlierRatio = outlierRatio;
    int nMatchNum = 8; // Use 8 point to solve analytic solutions.
    float fConfidenceRatio = 0.999;
    int nIterationNum = (int) (std::log(1 - fConfidenceRatio) / std::log(1 - std::pow(1 - fOutlierRatio, nMatchNum))); // RANSAC iteration number.
    int nTotalMatchNum = matches.size();
    int minOutlier = std::numeric_limits<int>::max();
    cv::Mat bestMatchFrl;
    for (int iter = 0; iter < nIterationNum; ++iter) {

        // Run a permutation to choose 8 matches to solve fundamental matrix.
        std::vector<size_t> vChoosenMatches;
        vChoosenMatches.resize(nMatchNum);
        size_t* permutation = Permutation(nTotalMatchNum);
        for (int j = 0; j < nMatchNum; ++j) {
            vChoosenMatches[j] = permutation[j];
        }
        delete[] permutation;

        cv::Mat A(nMatchNum, 9, CV_32F);
        for (int j = 0; j < nMatchNum; ++j) {
            cv::Point2f xl = keypointsL[matches[vChoosenMatches[j]].queryIdx].pt;
            cv::Point2f xr = keypointsR[matches[vChoosenMatches[j]].trainIdx].pt;
            cv::Mat a = (cv::Mat_<float>(1, 9) << xl.x * xr.x, xl.x * xr.y, xl.x * 1.0f, xl.y * xr.x, xl.y * xr.y, xl.y * 1.0f, 1.0f * xr.x, 1.0f * xr.y, 1.0f * 1.0f);
            a.copyTo(A.row(j));
        }

        // Solve the fundamental matrix.
        cv::Mat Frl = SolveF(A);

        // Use fundamental matrix to determine which match pair is a outlier.
        int outlierNum = 0;
        std::vector<bool> outlier;
        outlier.resize(nTotalMatchNum);
        for (int j = 0; j < nTotalMatchNum; ++j) {
            cv::Point2f xl = keypointsL[matches[j].queryIdx].pt;
            cv::Point2f xr = keypointsR[matches[j].trainIdx].pt;
            float a = Frl.at<float>(0, 0) * xl.x + Frl.at<float>(0, 1) * xl.y + Frl.at<float>(0, 2) * 1.0f;
            float b = Frl.at<float>(1, 0) * xl.x + Frl.at<float>(1, 1) * xl.y + Frl.at<float>(1, 2) * 1.0f;
            float c = Frl.at<float>(2, 0) * xl.x + Frl.at<float>(2, 1) * xl.y + Frl.at<float>(2, 2) * 1.0f;
            float d2 = ((a * xr.x + b * xr.y + c * 1.0f) * (a * xr.x + b * xr.y + c * 1.0f)) / (a * a + b * b);
            if (d2 > 25.0f) {
                outlierNum++;
                outlier[j] = true;
            }
            else
                outlier[j] = false;
        }
        if (outlierNum < fOutlierRatio * nTotalMatchNum) {
            if (outlierNum < minOutlier)
                minOutlier = outlierNum;
            int point_count = nTotalMatchNum - outlierNum;
            int count = 0;
            std::vector<cv::Point2f> pointsL(point_count);
            std::vector<cv::Point2f> pointsR(point_count);
            for (int j = 0; j < nTotalMatchNum; ++j) {
                if (!outlier[j]) {
                    pointsL[count] = keypointsL[matches[j].queryIdx].pt;
                    pointsR[count] = keypointsR[matches[j].trainIdx].pt;
                    ++count;
                }
            }
            bestMatchFrl = cv::findFundamentalMat(pointsL, pointsR, cv::FM_RANSAC, 3, 0.99);
        }
    }

    bestMatchFrl.convertTo(bestMatchFrl, CV_32F);


    // Use the best fundamental matrix to exclude outliers.
    for (int j = 0; j < nTotalMatchNum; ++j) {
        cv::Point2f xl = keypointsL[matches[j].queryIdx].pt;
        cv::Point2f xr = keypointsR[matches[j].trainIdx].pt;
        float a = bestMatchFrl.at<float>(0, 0) * xl.x + bestMatchFrl.at<float>(0, 1) * xl.y + bestMatchFrl.at<float>(0, 2) * 1.0f;
        float b = bestMatchFrl.at<float>(1, 0) * xl.x + bestMatchFrl.at<float>(1, 1) * xl.y + bestMatchFrl.at<float>(1, 2) * 1.0f;
        float c = bestMatchFrl.at<float>(2, 0) * xl.x + bestMatchFrl.at<float>(2, 1) * xl.y + bestMatchFrl.at<float>(2, 2) * 1.0f;
        float d2 = std::pow(a * xr.x + b * xr.y + c * 1.0f, 2) / (a * a + b * b);
        if (d2 < 25.0f) {
            inlierMatches.push_back(matches[j]);
        }
    }

    // Project fundamental matrix
    cv::Mat w, u, vt;
    cv::SVD::compute(bestMatchFrl, w, u, vt);
    w.at<float>(2, 0) = 0;
    bestMatchFrl = u * cv::Mat::diag(w) * vt;

    // Extract R and t from essential matrix. `left` -> `right`
    cv::Mat E = K.t() * bestMatchFrl * K;
    cv::Mat R1, R2, t1, t2;
    DecomposeE(E, R1, R2, t1);
    t2 = -t1;

    // Check if the two z is both positive.
    int good1 = CheckRT(K, R1, t1, keypointsL, keypointsR, inlierMatches);
    int good2 = CheckRT(K, R1, t2, keypointsL, keypointsR, inlierMatches);
    int good3 = CheckRT(K, R2, t1, keypointsL, keypointsR, inlierMatches);
    int good4 = CheckRT(K, R2, t2, keypointsL, keypointsR, inlierMatches);

    int maxGood = std::max(good1, std::max(good2, std::max(good3, good4)));

    cv::Mat R, t;
    if (maxGood == good1) {
        R = R1;
        t = t1;
    }
    else if (maxGood == good2) {
        R = R1;
        t = t2;
    }
    else if (maxGood == good3) {
        R = R2;
        t = t1;
    }
    else if (maxGood == good4) {
        R = R2;
        t = t2;
    }
    
    std::vector<cv::DMatch> inlierMatchesTemp;

    int outlierNum = 0;

    cv::Mat Pl(3, 4, CV_32F, cv::Scalar(0));
    Pl.at<float>(0, 0) = 1.0f;Pl.at<float>(1, 1) = 1.0f;Pl.at<float>(2, 2) = 1.0f;
    Pl = K * Pl;
    cv::Mat Pr(3, 4, CV_32F);
    R.copyTo(Pr.rowRange(0, 3).colRange(0, 3));
    t.copyTo(Pr.rowRange(0, 3).col(3));
    Pr = K * Pr;
    for (int i = 0, iend = inlierMatches.size(); i < iend; ++i) {
        cv::Mat xl = (cv::Mat_<float>(3, 1) << keypointsL[inlierMatches[i].queryIdx].pt.x, keypointsL[inlierMatches[i].queryIdx].pt.y, 1.0f);
        cv::Mat xr = (cv::Mat_<float>(3, 1) << keypointsR[inlierMatches[i].trainIdx].pt.x, keypointsR[inlierMatches[i].trainIdx].pt.y, 1.0f);
        cv::Mat Xl = LinearTriangulation(xl, xr, Pl, Pr);
        Xl = Xl.rowRange(0,3)/Xl.at<float>(3);
        cv::Mat Xr = R * Xl + t;
        if (Xl.at<float>(2) > 0 && Xr.at<float>(2) > 0) {
            inlierMatchesTemp.push_back(inlierMatches[i]);
            pcl::PointXYZRGB point;
            point.x = Xl.at<float>(0);
            point.y = Xl.at<float>(1);
            point.z = Xl.at<float>(2);
            point_cloud_ptr->points.push_back(point);
        }
        else
            outlierNum++;
    }

    inlierMatches = inlierMatchesTemp;

    return point_cloud_ptr;
}

int main(int argc, char** argv) {

    // Read the raw data
	cv::Mat rawL = cv::imread(argv[3]);
	cv::Mat rawR = cv::imread(argv[4]);

	// ORB_SLAM2_M::ORBVocabulary *pORBVocabulary = new ORB_SLAM2_M::ORBVocabulary();
 //    std::cout << "Read ORBVocabulary from `" << argv[1] << "`." << std::endl;
	// bool bVocLoad = pORBVocabulary->loadFromTextFile(argv[1]); // Vocabulary/ORBvoc.txt
	// if(!bVocLoad) {
 //        std::cerr << "Wrong path to vocabulary. " << std::endl;
 //        std::cerr << "Falied to open at: " << argv << std::endl;
 //        exit(-1);
 //    }
 //    std::cout << "ORBVocabulary loaded." << std::endl;

    cv::FileStorage fSettings(argv[2], cv::FileStorage::READ); // Examples/Monocular/TUM1.yaml

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];
    int nRGB = fSettings["Camera.RGB"];

    // Camera Parameters
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }

    // Undistort Images.
    cv::Mat udL;
    cv::undistort(rawL, udL, K, DistCoef, K);
    cv::Mat udR;
    cv::undistort(rawR, udR, K, DistCoef, K);

    // Extract features.
    ORB_SLAM2_M::ORBextractor* pORBExtractor = new ORB_SLAM2_M::ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    std::vector<cv::KeyPoint> keypointsL;
    std::vector<cv::KeyPoint> keypointsR;
    cv::Mat descriptorsL;
    cv::Mat descriptorsR;

    cv::Mat grayL = ConvertToGray(udL, nRGB);
    cv::Mat grayR = ConvertToGray(udR, nRGB);

    (*pORBExtractor)(grayL, cv::Mat(), keypointsL, descriptorsL);
    (*pORBExtractor)(grayR, cv::Mat(), keypointsR, descriptorsR);

    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_L2, true);
    matcher.match(descriptorsL, descriptorsR, matches); // `queryIdx` for descriptorsL, `trainIdx` for descriptorsR.

    std::vector<cv::DMatch> inlierMatches;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Reconstruction"));
    viewer->setBackgroundColor(255, 255, 255);
    viewer->addCoordinateSystem(1.0);

    viewer->addPointCloud<pcl::PointXYZRGB>(MyRANSACMatch(K, keypointsL, keypointsR, matches, 0.5, inlierMatches), "Reconstruction Cloud");

    cv::Mat imgMatches;
    cv::drawMatches( udL, keypointsL, udR, keypointsR, inlierMatches, imgMatches );
    cv::imshow( "matches", imgMatches );

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    cv::waitKey(0);

	return 0;
}