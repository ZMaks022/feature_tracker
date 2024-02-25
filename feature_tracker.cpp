#include "feature_tracker.h"

std::vector<cv::Mat> make_pyramid(const cv::Mat &image, int levels) {
    std::vector<cv::Mat> result;
    result.reserve(levels);

    result.push_back(image);

    for (size_t L = 1; L <= levels; ++L) {
        const cv::Mat &original = result[L - 1];
        cv::Mat prev;
        cv::copyMakeBorder(original, prev, 1, 1, 1, 1, cv::BORDER_REPLICATE);

        cv::Mat temp = cv::Mat::zeros(original.rows / 2, original.cols / 2, original.type());

        for (int y = 0; y < temp.rows; ++y) {
            for (int x = 0; x < temp.cols; ++x) {
                double value = 0;

                value += prev.at<uchar>(2*y, 2*x)/4.0;
                value += (prev.at<uchar>(2*y, 2*x - 1) +
                        prev.at<uchar>(2*y, 2*x + 1) +
                        prev.at<uchar>(2*y - 1, 2*x) +
                        prev.at<uchar>(2*y + 1, 2*x))/8.0;
                value += (prev.at<uchar>(2*y - 1, 2*x - 1) +
                        prev.at<uchar>(2*y + 1, 2*x + 1) +
                        prev.at<uchar>(2*y + 1, 2*x - 1) +
                        prev.at<uchar>(2*y + 1, 2*x + 1))/16.0;

                temp.at<uchar>(y, x) = static_cast<uchar>(value);
            }
        }

        result.push_back(temp);
    }

    return result;
}

bool is_within_image(const cv::Mat &img, const cv::Point2f &pt) {
    return pt.x >= 0 && pt.x < img.cols && pt.y >= 0 && pt.y < img.rows;
}


cv::Point2f optical_flow(const cv::Mat &prev_img, const cv::Mat &next_img, cv::Point2f u, cv::Point2f guss) {
    const int omega_x = 4, omega_y = 4;
    const int K = 100;
    const double THRESHOLD = 0.2;

    // Derivative of IL with respect to x
    cv::Mat I_dx(prev_img.size(), CV_32F);
    for (int y = 0; y < I_dx.rows; ++y) {
        for (int x = 0; x < I_dx.cols; ++x) {
            float value = (x < prev_img.cols - 1) ? prev_img.at<uchar>(y, x + 1) : 0;
            value -= (x > 0) ? prev_img.at<uchar>(y, x - 1) : 0;

            I_dx.at<float>(y, x) = value / 2.0;
        }
    }

    // Derivative of IL with respect to y
    cv::Mat I_dy(prev_img.size(), CV_32F);
    for (int y = 0; y < I_dy.rows; ++y) {
        for (int x = 0; x < I_dy.cols; ++x) {
            float value = (y < prev_img.rows - 1) ? prev_img.at<uchar>(y + 1, x) : 0;
            value -= (y > 0) ? prev_img.at<uchar>(y - 1, x) : 0;

            I_dy.at<float>(y, x) = value / 2.0f;
        }
    }

    // Spatial gradient matrix
    cv::Mat G = cv::Mat::zeros(2, 2, CV_32F);
    // Define the window boundaries
//    int start_x = (int)u.x - omega_x;
//    int end_x = (int)u.x + omega_x + 1;
//    int start_y = (int)u.y - omega_y;
//    int end_y = (int)u.y + omega_y + 1;
    int start_x = std::max(0, (int)u.x - omega_x - 1);
    int end_x = std::min(prev_img.cols, (int)u.x + omega_x + 1);
    int start_y = std::max(0, (int)u.y - omega_y - 1);
    int end_y = std::min(prev_img.rows, (int)u.y + omega_y + 1);

    for (int y = start_y; y < end_y; ++y) {
        for (int x = start_x; x < end_x; ++x) {
            float ix = I_dx.at<float>(y, x);
            float iy = I_dy.at<float>(y, x);
            G.at<float>(0, 0) += ix * ix;
            G.at<float>(0, 1) += ix * iy;
            G.at<float>(1, 0) += ix * iy;
            G.at<float>(1, 1) += iy * iy;
        }
    }

    cv::Mat G_inv = cv::Mat::zeros(G.rows, G.cols, G.type());

    cv::Point2f next_pts = {0.0f, 0.0f};

    cv::Mat eta_k = cv::Mat::zeros(2, 1, CV_32F);
    for (size_t k = 1; k < K; ++k) {
        if (!is_within_image(next_img, u + next_pts + guss)) {
            continue;
        }

        // Image difference
        cv::Mat I_k(prev_img.size(), CV_32F);
        for (int y = 0; y < I_k.rows; ++y) {
            for (int x = 0; x < I_k.cols; ++x) {
                try {
                    float I_val = prev_img.at<uchar>(y, x);
                    float J_val = next_img.at<uchar>(
                            y + next_pts.y + guss.y,
                            x + next_pts.x + guss.x
                    );
                    I_k.at<float>(y, x) = I_val - J_val;
                } catch (const std::exception& e) {
                    std::cout << e.what() << std::endl;
                    return {0, 0};
                }
            }
        }

        // Compute b_k over the window
//        start_x = u.x - omega_x;
//        end_x = u.x + omega_x;
//        start_y = u.y - omega_y;
//        end_y = u.y + omega_y;
        start_x = std::max(0, (int)u.x - omega_x - 1);
        end_x = std::min(prev_img.cols, (int)u.x + omega_x + 1);
        start_y = std::max(0, (int)u.y - omega_y - 1);
        end_y = std::min(prev_img.rows, (int)u.y + omega_y + 1);

        cv::Mat b_k = cv::Mat::zeros(2, 1, G.type());
        for (int y = start_y; y < end_y; ++y) {
            for (int x = start_x; x < end_x; ++x) {
                b_k.at<float>(0, 0) += I_k.at<float>(y, x) * I_dx.at<float>(y, x);
                b_k.at<float>(1, 0) += I_k.at<float>(y, x) * I_dy.at<float>(y, x);
            }
        }

        if (cv::determinant(G) < 0.01) {
            double lambda = 0.1;
            G += cv::Mat::eye(G.size(), G.type()) * lambda;
        }

        cv::invert(G, G_inv);
        eta_k = G_inv * b_k;
        next_pts.x += eta_k.at<float>(0, 0);
        next_pts.y += eta_k.at<float>(1, 0);

        if (cv::norm(eta_k) < THRESHOLD) {
            break;
        }

    }

    return next_pts;
}


std::vector<cv::Point2f> pyramidal_tracking(const cv::Mat &grayI, const cv::Mat &grayJ, const std::vector<cv::Point2f> &us) {
    const int LEVELS = 5;

    std::vector<cv::Mat> pyramidI = make_pyramid(grayI, LEVELS);
    std::vector<cv::Mat> pyramidJ = make_pyramid(grayJ, LEVELS);

    std::vector<cv::Point2f> ds(us.size());

    for (size_t i = 0; i < us.size(); ++i) {
        const auto u = us[i];
        cv::Point2f displacement = cv::Point2f(0, 0);

        for (int level = LEVELS - 1; level >= 0; --level) {
            cv::Point2f scaledU = u / pow(2, level);
            cv::Point2f current_displacement = displacement / pow(2, level);

            cv::Point2f new_displacement;

            try {
                new_displacement = optical_flow(pyramidI[level], pyramidJ[level], scaledU, current_displacement);
            } catch (const std::exception& e) {
                std::cout << "Tracking failed at level " << level << std::endl;
                std::cout << "with error: " << e.what() << std::endl;
                break;
            }
            displacement += new_displacement * pow(2, level);

        }
        ds[i] = displacement;
    }

    return ds;
}
