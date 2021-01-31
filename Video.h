//
// Created by jm1417 on 25/12/2020.
//

#ifndef SIMULATOR_VIDEO_H
#define SIMULATOR_VIDEO_H

#include <opencv2/core/mat.hpp>
#include "PE/PE.h"

class Video {
public:
    static cv::Mat draw_analogue_register(AREG& reg, const std::string& window);
    static void capture();
};


#endif //SIMULATOR_VIDEO_H
