#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "feature_tracker.h"


int main() {
    const int POINT_RADIUS = 2;
    const int BORDER_SIZE = 15; // size of the border near which we do not select points (optional)
    const int THICKNESS = 1; // connecting point line thickness

    std::string filename1 = "frame_001";

    std::string filename_read_1 = "../inputs/" + filename1 + ".png";
    cv::Mat I = cv::imread(filename_read_1);
    std::vector<cv::Point2f> us = {
            {373, 121},
            {354, 133},
            {349, 145},
            {331, 158},
            {351, 158},
            {400, 133},
            {408, 154},
            {387, 84},
            {353, 106}
    };

    for (size_t i = 5; i < 45; ++i) {
        std::string frame = std::to_string(i);
        std::string formatted_str = std::string(3 - frame.length(), '0') + frame;
        std::string filename2 = "frame_" + formatted_str;

        std::string filename_read_2 = "../inputs/" + filename2 + ".png";

        cv::Mat J = cv::imread(filename_read_2);
        if (I.empty() || J.empty()) {
            std::cout << "Failed to load image" << std::endl;
            return -1;
        }

        cv::Mat grayI, grayJ;
        cv::cvtColor(I, grayI, cv::COLOR_BGR2GRAY);
        cv::cvtColor(J, grayJ, cv::COLOR_BGR2GRAY);

//        std::vector<cv::Point2f> us;
//        cv::goodFeaturesToTrack(grayI, us, MAX_POINTS, 0.07, 10);
        // now points that are near the border are removed here
        {
            std::vector<cv::Point2f> filtered_points;
            for (auto &u: us) {
                int u_x_int = static_cast<int>(u.x);
                int u_y_int = static_cast<int>(u.y);
                if ((u_x_int > BORDER_SIZE) && (u_x_int < grayI.cols - BORDER_SIZE) &&
                    (u_y_int > BORDER_SIZE) && (u_y_int < grayI.rows - BORDER_SIZE)) {
                    filtered_points.push_back(u);
                }
            }
            us = filtered_points;
        }

        // Start main algorithm from article
        auto ds = pyramidal_tracking(grayI, grayJ, us);

        // draw features in the first image
//        for (auto u: us) {
//            cv::circle(I, u, POINT_RADIUS, cv::Scalar(0, 255, 0), THICKNESS);
//        }
//        std::string filename_write_1 = "../outputs/" + filename1 + ".png";
//        cv::imwrite(filename_write_1, I);

        // draw the same features in second image
        for (size_t i = 0; i < us.size(); ++i) {
            const auto u = us[i];
            const auto d = ds[i];
            const cv::Point2f v = u + d;

            cv::circle(J, u, POINT_RADIUS, cv::Scalar(0, 0, 255), THICKNESS);
            cv::circle(J, v, POINT_RADIUS, cv::Scalar(0, 255, 0), THICKNESS);
            cv::line(J, u, v, cv::Scalar(255, 0, 0), THICKNESS);
        }

        std::string filename_write_2 = "../outputs/" + filename2 + ".png";
        cv::imwrite(filename_write_2, J);

        std::cout << filename2 << ".png generated" << std::endl;
    }

    return 0;
}