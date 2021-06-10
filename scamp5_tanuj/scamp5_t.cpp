//
// Created by td1518 on 20/05/2021.
//

#include "scamp5_t.h"

#include <rttr/registration>

void SCAMP5T::init() {
    SCAMP5::init();
}

void SCAMP5T::hard_sigmoid(
        const std::shared_ptr<AnalogueRegister>& dst,
        const std::shared_ptr<AnalogueRegister>& src) {
    pe->analogue_bus.hard_sigmoid(*dst, *src, *FLAG);
}


void SCAMP5T::maxpool(const std::shared_ptr<AnalogueRegister>&reg) {
    cv::Mat& src = reg->read();

    for (int r = 0; r < rows_; r += 2) {
        for (int c = 0; c < cols_; c += 2) {
            float max = std::max({
                src.at<float>(r,c),
                src.at<float>(r + 1,c),
                src.at<float>(r,c + 1),
                src.at<float>(r + 1,c + 1),
            });
            src.at<float>(r,c) = max;
            src.at<float>(r + 1,c) = max;
            src.at<float>(r,c + 1) = max;
            src.at<float>(r + 1,c + 1) = max;
        }
    }
}

void SCAMP5T::maxpool_sparse(const std::shared_ptr<AnalogueRegister>&reg) {
    cv::Mat& src = reg->read();

    for (int r = 0; r < rows_; r += 4) {
        for (int c = 0; c < cols_; c += 4) {
            float max = std::max({
                src.at<float>(r,c),
                src.at<float>(r + 2,c),
                src.at<float>(r,c + 2),
                src.at<float>(r + 2,c + 2),
            });

            // Set entire block to max val
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    src.at<float>(r + i, c + j) = max;
                }
            }
        }
    }
}

RTTR_REGISTRATION {
    using namespace rttr;

    registration::class_<SCAMP5T>("SCAMP5T")
        .constructor()
        .method("init", &SCAMP5T::init)
        .method("hard_sigmoid", &SCAMP5T::hard_sigmoid)
        .method("maxpool", &SCAMP5T::maxpool)
        .method("maxpool_sparse", &SCAMP5T::maxpool_sparse);
};
