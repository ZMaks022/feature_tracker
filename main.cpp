#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "feature_tracker.h"


int main() {
    const int POINT_RADIUS = 2;
    const int MAX_POINTS = 50;
    const int BORDER_SIZE = 20; // size of the border near which we do not select points (temporary stub)
    const int THICKNESS = 1; // connecting point line thickness

    std::string filename1 = "frame_050";
    std::string filename2 = "frame_060";

    std::string filename_read_1 = "../inputs/" + filename1 + ".png";
    std::string filename_read_2= "../inputs/" + filename2 + ".png";
    cv::Mat I = cv::imread(filename_read_1);
    cv::Mat J = cv::imread(filename_read_2);
    if (I.empty() || J.empty()) {
        std::cout << "Failed to load image" << std::endl;
        return -1;
    }

    cv::Mat grayI, grayJ;
    cv::cvtColor(I, grayI, cv::COLOR_BGR2GRAY);
    cv::cvtColor(J, grayJ, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> us;
    cv::goodFeaturesToTrack(grayI, us, MAX_POINTS, 0.07, 10);
    // REDO now points that are near the border are removed here
    {
        std::vector<cv::Point2f> filtered_points;
        for (auto & u : us) {
            int u_x_int = static_cast<int>(u.x);
            int u_y_int = static_cast<int>(u.y);
            if ( (u_x_int > BORDER_SIZE) && ( u_x_int < grayI.cols - BORDER_SIZE) &&
                 (u_y_int > BORDER_SIZE) && ( u_y_int < grayI.rows - BORDER_SIZE) ) {
                filtered_points.push_back(u);
            }
        }
        us = filtered_points;
    }

    // Start main algorithm from article
    auto ds = pyramidal_tracking(grayI, grayJ, us);

    // draw features in the first image
    for (auto u : us) {
        cv::circle(I, u, POINT_RADIUS, cv::Scalar(0, 255, 0), THICKNESS);
    }
//    cv::imshow("Test", I);
//    cv::waitKey(0);
    std::string filename_write_1 = "../outputs/" + filename1 + ".png";
    cv::imwrite(filename_write_1, I);

    // draw the same features in second image
    for (size_t i = 0; i < us.size(); ++i) {
        const auto u = us[i];
        const auto d = ds[i];
        const cv::Point2f v = u + d;

        cv::circle(J, u, POINT_RADIUS, cv::Scalar(0, 0, 255), THICKNESS);
        cv::circle(J, v, POINT_RADIUS, cv::Scalar(0, 255, 0), THICKNESS);
        cv::line(J, u, v, cv::Scalar(255, 0, 0), THICKNESS);
    }
//    cv::imshow("Test", J);
//    cv::waitKey(0);
    std::string filename_write_2 = "../outputs/" + filename2 + ".png";
    cv::imwrite(filename_write_2, J);

    return 0;
}