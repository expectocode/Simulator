//
// Created by jm1417 on 31/01/2021.
//

#ifndef SIMULATOR_REGISTER_H
#define SIMULATOR_REGISTER_H

#include <opencv2/core/mat.hpp>

typedef cv::Mat Data;

class Register {
protected:
    Data value_;
public:
    Register(int rows, int columns, int type);
    Data & value()       { return value_; }
    [[nodiscard]] const Data & value() const { return value_; }
};


#endif //SIMULATOR_REGISTER_H
