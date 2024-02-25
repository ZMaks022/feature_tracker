cmake_minimum_required(VERSION 3.27)
project(FeatureTracker)

set(CMAKE_CXX_STANDARD 20)

add_executable(FeatureTracker main.cpp
        feature_tracker.h
        feature_tracker.cpp)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(FeatureTracker PRIVATE ${OpenCV_LIBS})

