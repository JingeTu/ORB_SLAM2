#include "Initializer_M.h"
#include <thread>

namespace ORB_SLAM2_M {

	Initializer::Initializer(const std::vector<cv::KeyPoint> &key_point_1_vec,
		const std::vector<cv::KeyPoint> &key_point_2_vec,
		const std::vector<cv::DMatch> &matches_1_2_vec,
		int max_iterations_n, float sigma_f)
	: key_point_1_vec_{key_point_1_vec},
	key_point_2_vec_{key_point_2_vec},
	matches_1_2_vec_{matches_1_2_vec},
	max_iterations_n_{max_iterations_n}, sigma_f_{sigma_f}{

	}

	bool Initializer::Initialize() {
		// We need to calculate these two
		cv::Mat R21, t21;

		ransac_vec = std::vector<std::vector<int>>(max_iterations_n_, std::vector(8, 0));

		std::vector<size_t> all_indices_vec;
		all_indices_vec.reserve(N);
		for (size_t i = 0; i < N; ++i) {
			all_indices_vec.push_back(i);
		}

		std::vector<size_t> avaliable_indices_vec;

		DUtils::Random::SeedRandOnce(0);

		for (int i = 0, iend = max_iterations_n_; i < iend; ++i) {
			avaliable_indices_vec = all_indices_vec;
			for (int j = 0; j < 8; ++j) {
				int rand = DUtils::Random::RandomInt(0, avaliable_indices_vec.size()-1);
				ransac_vec[i][j] = rand;

				avaliable_indices_vec[rand] = avaliable_indices_vec[avaliable_indices_vec.size() - 1];
				avaliable_indices_vec.pop_back();
			}
		}

		vector<bool> vbMatchesInliersH, vbMatchesInliersF;
		float SH, SF;
		cv::Mat H, F;

		std::thread threadH(&Initializer::FindHomography,this,ref(vbMatchesInliersH), ref(SH), ref(H));
		std::thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

		threadH.join();
		threadF.join();
	}

	void Initializer::FindHomography(std::vector<bool> &match_inliers_vec, float &score, cv::Mat &H21) {

	}
}