//
// Created by jm1417 on 06/03/2021.
//

#include "simulator/input/live_input.h"

#include <simulator/base/architecture.h>

#include <chrono>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>

LiveInput::LiveInput() = default;

LiveInput::LiveInput(int rows, int cols) {
    this->rows_ = rows;
    this->cols_ = cols;
    this->capture = std::make_unique<cv::VideoCapture>(0);
    if(!this->capture->isOpened()) {
        std::cerr << "Could not open camera" << std::endl;
        exit(1);
    }

    this->size = std::make_unique<cv::Size>(cols, rows);
    this->frame = cv::Mat(rows, cols, MAT_TYPE);
    this->frame.setTo(0);
}

void LiveInput::read(Register &reg) {
#ifdef USE_RUNTIME_CHECKS
    if(this->capture == nullptr) {
        std::cerr << "No video capture defined" << std::endl;
    }
#endif
    cv::Mat temp(rows_, cols_, CV_32S);
    auto TIME_START = std::chrono::high_resolution_clock::now();
    *this->capture >> temp;
#ifdef USE_RUNTIME_CHECKS
    if(temp.empty()) {
        std::cerr << "ERROR! blank frame grabbed" << std::endl;
    }
#endif
    cv::cvtColor(temp, temp, cv::COLOR_BGR2GRAY);

    int width = temp.cols;
    int height = temp.rows;
    cv::Mat cropFrame =
        temp(cv::Rect((width - height) / 2, 0, height - 1, height - 1));
    cv::resize(cropFrame, cropFrame, *this->size);
    cropFrame.convertTo(temp, MAT_TYPE, 1, -128);
    cv::add(this->frame, temp, this->frame);
    reg.write(this->frame);
    auto TIME_END = std::chrono::high_resolution_clock::now();
    long time_in_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            TIME_END - TIME_START)
                            .count();
    time_taken = time_in_nano * 1e-9;
}

void LiveInput::reset() { this->frame.setTo(0); }

double LiveInput::last_frame_time() { return time_taken; }
