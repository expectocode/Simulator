//
// Created by jm1417 on 28/01/2021.
//

#ifndef SIMULATOR_ARRAY_H
#define SIMULATOR_ARRAY_H


#include <utility>

#include "processing_element.h"

class Array {
protected:
    int rows_;
    int columns_;

public:
    Array(int rows, int columns, ProcessingElement  pe);

    ProcessingElement pe;
};


#endif //SIMULATOR_ARRAY_H