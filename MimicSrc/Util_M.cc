#include "Util_M.h"
#include <time>
#include <limits>

namespace ORB_SLAM2_M {

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

    ORB_PARAM ReadORBParameters(std::string file_path) {
        ORB_PARAM ret_orb_param;
        cv::FileStorage fSettings(file_path, cv::FileStorage::READ);
        ret_orb_param.features_n = fSettings["ORBextractor.nFeatures"];
        ret_orb_param.scale_factor_f = fSettings["ORBextractor.scaleFactor"];
        ret_orb_param.levels_n = fSettings["ORBextractor.nLevels"];
        ret_orb_param.ini_th_FAST_n = fSettings["ORBextractor.iniThFAST"];
        ret_orb_param.min_th_FAST_n = fSettings["ORBextractor.minThFAST"];
        ret_orb_param.rgb_n = fSettings["Camera.RGB"];

        return ret_orb_param;
    }

    cv::Mat ReadCameraMatrix(std::string file_path) {
        cv::FileStorage fSettings(file_path, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];
        cv::Mat K = cv::Mat::eye(3,3,CV_32F);
        K.at<float>(0,0) = fx;
        K.at<float>(1,1) = fy;
        K.at<float>(0,2) = cx;
        K.at<float>(1,2) = cy;

        return K;
    }

    cv::Mat ReadDistortCoefficient(std::string file_path) {
        cv::FileStorage fSettings(file_path, cv::FileStorage::READ);
        cv::Mat dist_coef(4,1,CV_32F);
        dist_coef.at<float>(0) = fSettings["Camera.k1"];
        dist_coef.at<float>(1) = fSettings["Camera.k2"];
        dist_coef.at<float>(2) = fSettings["Camera.p1"];
        dist_coef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if(k3!=0)
        {
            dist_coef.resize(5);
            dist_coef.at<float>(4) = k3;
        }

        return dist_coef;
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

    void MyRANSACMatch(const cv::Mat &K, const std::vector<cv::KeyPoint> &keypointsL, const std::vector<cv::KeyPoint> &keypointsR,
     const std::vector<cv::DMatch> &matches, const float outlierRatio, std::vector<cv::DMatch> &inlierMatches) {

        std::srand(std::time(0));

        float outlier_ratio_f = outlierRatio;
        int min_match_num_n = 8; // Use 8 point to solve analytic solutions.
        float confidence_ratio_f = 0.999;
        int max_iteration_num_n = (int) (std::log(1 - confidence_ratio_f) / std::log(1 - std::pow(1 - outlier_ratio_f, min_match_num_n))); // RANSAC iteration number.
        int matches_size_n = matches.size();
        int min_outlier_num_n = std::numeric_limits<int>::max();
        cv::Mat bestMatchFrl;
        for (int iter = 0; iter < max_iteration_num_n; ++iter) {

            // Run a permutation to choose 8 matches to solve fundamental matrix.
            std::vector<size_t> vChoosenMatches;
            vChoosenMatches.resize(min_match_num_n);
            size_t* permutation = Permutation(matches_size_n);
            for (int j = 0; j < min_match_num_n; ++j) {
                vChoosenMatches[j] = permutation[j];
            }
            delete[] permutation;

            cv::Mat A(min_match_num_n, 9, CV_32F);
            for (int j = 0; j < min_match_num_n; ++j) {
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
            outlier.resize(matches_size_n);
            for (int j = 0; j < matches_size_n; ++j) {
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
            if (outlierNum < outlier_ratio_f * matches_size_n) {
                if (outlierNum < min_outlier_num_n)
                    min_outlier_num_n = outlierNum;
                int point_count = matches_size_n - outlierNum;
                int count = 0;
                std::vector<cv::Point2f> pointsL(point_count);
                std::vector<cv::Point2f> pointsR(point_count);
                for (int j = 0; j < matches_size_n; ++j) {
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
        for (int j = 0; j < matches_size_n; ++j) {
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
            }
            else{
                outlierNum++;
            }
        }

        inlierMatches = inlierMatchesTemp;

    }


}