//
// Created by jm1417 on 28/01/2021.
//

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "photodiode.h"
#include "array.h"

Photodiode::Photodiode(int rows, int columns) : rows_(rows), columns_(columns) {
    this->capture = std::make_shared<cv::VideoCapture>(0,  cv::CAP_ANY);
    if (!(*this->capture).isOpened()) {
        std::cerr << "Could not open camera" << std::endl;
        exit(1);
    }
    this->size = std::make_shared<cv::Size>(columns, rows);
    this->frame = cv::Mat(rows_, columns_, MAT_TYPE);
    this->reset();
}

void Photodiode::reset() {
    this->frame.setTo(0);
}

void Photodiode::read(AnalogueRegister& reg) {
    if (this->capture == nullptr) {
        std::cerr << "No video capture defined" << std::endl;
    }
    cv::Mat temp(rows_, columns_, CV_32S);
    auto TIME_START = std::chrono::high_resolution_clock::now();
    *this->capture >> temp;
    if (temp.empty()) {
        std::cerr << "ERROR! blank frame grabbed" << std::endl;
    }

    cv::cvtColor(temp, temp, cv::COLOR_BGR2GRAY);

    int width = temp.cols;
    int height = temp.rows;
    cv::Mat cropFrame = temp(cv::Rect((width-height)/2, 0, height-1, height-1));
    cv::resize(cropFrame, cropFrame, *this->size);
    cropFrame.convertTo(temp, MAT_TYPE);
    cv::add(this->frame, temp, this->frame);
    this->frame.copyTo(reg.value());
    auto TIME_END = std::chrono::high_resolution_clock::now();
    long time_in_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(TIME_END-TIME_START).count();
    time_taken = time_in_nano*1e-9;
}

double Photodiode::last_frame_time() {
    return time_taken;
}

void Photodiode::print_stats(const CycleCounter &counter) {

}

void Photodiode::write_stats(const CycleCounter &counter, json& j) {

}

Photodiode::~Photodiode() {
}



