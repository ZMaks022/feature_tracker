#pragma once
#ifndef ANGLEDETECTOR_FEATURE_TRACKER_H
#define ANGLEDETECTOR_FEATURE_TRACKER_H

#include <vector>
#include <opencv2/opencv.hpp>

std::vector<cv::Mat> make_pyramid(const cv::Mat &image, int levels);
cv::Point2f optical_flow(const cv::Mat &prev_img, const cv::Mat &next_img, cv::Point2f u, cv::Point2f guss, int &error);
std::vector<cv::Point2f> pyramidal_tracking(const cv::Mat &I, const cv::Mat &J, const std::vector<cv::Point2f> &us);

#endif //ANGLEDETECTOR_FEATURE_TRACKER_H
