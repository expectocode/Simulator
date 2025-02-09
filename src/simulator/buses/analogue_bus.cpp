//
// Created by jm1417 on 28/01/2021.
//

#include "simulator/buses/analogue_bus.h"

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>

void AnalogueBus::bus(AnalogueRegister &a, DigitalRegister &FLAG) {
    // a = 0 + error
    a.write(0, FLAG);
}

void AnalogueBus::bus(AnalogueRegister &a, AnalogueRegister &a0,
                      DigitalRegister &FLAG) {
    // a = -a0 + error
    cv::bitwise_not(a0.read(), a.read(), FLAG.read());
    cv::add(a.read(), 1, a.read(), FLAG.read()); //todo counts
}

void AnalogueBus::bus(AnalogueRegister &a, AnalogueRegister &a0,
                      AnalogueRegister &a1, DigitalRegister &FLAG) {
    // a = -(a0 + a1) + error
    cv::add(a0.read(), a1.read(), scratch, FLAG.read());
    cv::bitwise_not(scratch, a.read(), FLAG.read());
    cv::add(a.read(), 1, a.read(), FLAG.read());
}

void AnalogueBus::bus(AnalogueRegister &a, AnalogueRegister &a0,
                      AnalogueRegister &a1, AnalogueRegister &a2,
                      DigitalRegister &FLAG) {
    // a = -(a0 + a1 + a2) + error
    cv::add(a0.read(), a1.read(), scratch, FLAG.read());
    cv::add(scratch, a2.read(), scratch, FLAG.read());
    cv::bitwise_not(scratch, a.read(), FLAG.read());
    cv::add(a.read(), 1, a.read(), FLAG.read());
}

void AnalogueBus::bus(AnalogueRegister &a, AnalogueRegister &a0,
                      AnalogueRegister &a1, AnalogueRegister &a2,
                      AnalogueRegister &a3, DigitalRegister &FLAG) {
    // a = -(a0 + a1 + a2 + a3) + error
    cv::add(a0.read(), a1.read(), scratch, FLAG.read());
    cv::add(scratch, a2.read(), scratch, FLAG.read());
    cv::add(scratch, a3.read(), scratch, FLAG.read());
    cv::bitwise_not(scratch, a.read(), FLAG.read());
    cv::add(a.read(), 1, a.read(), FLAG.read());
}

void AnalogueBus::bus2(AnalogueRegister &a, AnalogueRegister &b,
                       DigitalRegister &FLAG) {
    // a,b = 0 + error
    a.write(0, FLAG);
    b.write(0, FLAG);
}

void AnalogueBus::bus2(AnalogueRegister &a, AnalogueRegister &b,
                       AnalogueRegister &a0, DigitalRegister &FLAG) {
    // a,b = -0.5*a0 + error + noise
    cv::multiply(a0.read(), 0.5, scratch);
    cv::bitwise_not(scratch, scratch, FLAG.read());
    cv::add(scratch, 1, scratch, FLAG.read());
    a.write(scratch, FLAG.read());
    b.write(scratch, FLAG.read());
}

void AnalogueBus::bus2(AnalogueRegister &a, AnalogueRegister &b,
                       AnalogueRegister &a0, AnalogueRegister &a1,
                       DigitalRegister &FLAG) {
    // a,b = -0.5*(a0 + a1) + error + noise
    cv::add(a0.read(), a1.read(), scratch, FLAG.read());
    cv::multiply(scratch, 0.5, scratch);
    cv::bitwise_not(scratch, scratch, FLAG.read());
    cv::add(scratch, 1, scratch, FLAG.read());
    a.write(scratch, FLAG.read());
    b.write(scratch, FLAG.read());
}

void AnalogueBus::bus3(AnalogueRegister &a, AnalogueRegister &b,
                       AnalogueRegister &c, AnalogueRegister &a0,
                       DigitalRegister &FLAG) {
    // a,b,c = -0.33*a0 + error + noise
    cv::multiply(0.333, a0.read(), scratch);
    cv::bitwise_not(scratch, scratch, FLAG.read());
    cv::add(scratch, 1, scratch, FLAG.read());
    a.write(scratch, FLAG.read());
    b.write(scratch, FLAG.read());
    c.write(scratch, FLAG.read());
}

void AnalogueBus::conditional_positive_set(DigitalRegister &b,
                                           AnalogueRegister &a) {
    // b := 1 if a > 0
    cv::threshold(a.read(), b.read(), 0, 1, cv::THRESH_BINARY);
    b.read().convertTo(b.read(), CV_8U);
}

void AnalogueBus::conditional_positive_set(DigitalRegister &b,
                                           AnalogueRegister &a0,
                                           AnalogueRegister &a1) {
    // b := 1 if (a0 + a1) > 0.
    cv::add(a0.read(), a1.read(), scratch);
    cv::threshold(scratch, b.read(), 0, 1, cv::THRESH_BINARY);
    b.read().convertTo(b.read(), CV_8U);
}

void AnalogueBus::conditional_positive_set(DigitalRegister &b,
                                           AnalogueRegister &a0,
                                           AnalogueRegister &a1,
                                           AnalogueRegister &a2) {
    // b := 1 if (a0 + a1 + a2) > 0.
    cv::add(a0.read(), a1.read(), scratch);
    cv::add(scratch, a2.read(), scratch);
    threshold(scratch, b.read(), 0, 1, cv::THRESH_BINARY);
    b.read().convertTo(b.read(), CV_8U);
}

void AnalogueBus::mov(AnalogueRegister &y, AnalogueRegister &x0,
                      AnalogueRegister &intermediate, DigitalRegister &FLAG) {
    // y = x0
    AnalogueBus::bus(intermediate, x0, FLAG);
    AnalogueBus::bus(y, intermediate, FLAG);
}

void AnalogueBus::add(AnalogueRegister &y, AnalogueRegister &x0,
                      AnalogueRegister &x1, AnalogueRegister &intermediate,
                      DigitalRegister &FLAG) {
    // y = x0 + x1
    AnalogueBus::bus(intermediate, x0, x1, FLAG);
    AnalogueBus::bus(y, intermediate, FLAG);
}

void AnalogueBus::add(AnalogueRegister &y, AnalogueRegister &x0,
                      AnalogueRegister &x1, AnalogueRegister &x2,
                      AnalogueRegister &intermediate, DigitalRegister &FLAG) {
    // y = x0 + x1 + x2
    AnalogueBus::bus(intermediate, x0, x1, x2, FLAG);
    AnalogueBus::bus(y, intermediate, FLAG);
}

void AnalogueBus::sub(AnalogueRegister &y, AnalogueRegister &x0,
                      AnalogueRegister &x1, AnalogueRegister &intermediate,
                      DigitalRegister &FLAG) {
    // y = x0 - x1
    AnalogueBus::bus(intermediate, x0, FLAG);
    AnalogueBus::bus(y, intermediate, x1, FLAG);
}

void AnalogueBus::neg(AnalogueRegister &y, AnalogueRegister &x0,
                      AnalogueRegister &intermediate, DigitalRegister &FLAG) {
    // y = -x0
    AnalogueBus::bus(intermediate, FLAG);
    AnalogueBus::bus(y, intermediate, x0, FLAG);
}

void AnalogueBus::abs(AnalogueRegister &y, AnalogueRegister &x0,
                      AnalogueRegister &intermediate, DigitalRegister &FLAG) {
    // y = |x0|
    AnalogueBus::bus(intermediate, FLAG);
    AnalogueBus::bus(y, intermediate, x0, FLAG);
    AnalogueBus::bus(intermediate, x0, FLAG);
    AnalogueBus::conditional_positive_set(FLAG, x0);
    AnalogueBus::bus(y, intermediate, FLAG);
    FLAG.set();
}

void AnalogueBus::div(AnalogueRegister &y0, AnalogueRegister &y1,
                      AnalogueRegister &y2, AnalogueRegister &intermediate,
                      DigitalRegister &FLAG) {
    // y0 := 0.5*y2; y1 := -0.5*y2 + error, y2 := y2 + error
    AnalogueBus::bus2(y0, y1, y2, FLAG);
    AnalogueBus::bus(intermediate, y2, y1, FLAG);
    AnalogueBus::bus(y2, intermediate, y0, FLAG);
    AnalogueBus::bus2(y0, y1, y2, FLAG);
    AnalogueBus::bus(y0, y1, FLAG);
}

void AnalogueBus::div(AnalogueRegister &y0, AnalogueRegister &y1,
                      AnalogueRegister &y2, AnalogueRegister &x0,
                      AnalogueRegister &intermediate, DigitalRegister &FLAG) {
    // y0 := 0.5*x0; y1 := -0.5*x0 + error, y2 := x0 + error
    AnalogueBus::bus2(y0, y1, x0, FLAG);
    AnalogueBus::bus(intermediate, x0, y1, FLAG);
    AnalogueBus::bus(y2, intermediate, y0, FLAG);
    AnalogueBus::bus2(y0, y1, y2, FLAG);
    AnalogueBus::bus(y0, y1, FLAG);
}

void AnalogueBus::diva(AnalogueRegister &y0, AnalogueRegister &y1,
                       AnalogueRegister &y2, AnalogueRegister &intermediate,
                       DigitalRegister &FLAG) {
    // y0 := 0.5*y0; y1 := -0.5*y0 + error, y2 := -0.5*y0 + error
    AnalogueBus::bus2(y1, y2, y0, FLAG);
    AnalogueBus::bus(intermediate, y1, y0, FLAG);
    AnalogueBus::bus(y0, intermediate, y2, FLAG);
    AnalogueBus::bus2(y1, y2, y0, FLAG);
    AnalogueBus::bus(y0, y1, FLAG);
}

void AnalogueBus::divq(AnalogueRegister &y0, AnalogueRegister &x0,
                       AnalogueRegister &intermediate, DigitalRegister &FLAG) {
    // y0 := 0.5*x0 + error
    AnalogueBus::bus2(y0, intermediate, x0, FLAG);
    AnalogueBus::bus(y0, intermediate, FLAG);
}



void AnalogueBus::get_up(AnalogueRegister &dst, AnalogueRegister &src, int offset) {
    // x, y, width, height
    auto read_chunk =
        cv::Rect(0, 0, src.read().cols, src.read().rows - offset);
    auto write_chunk =
        cv::Rect(0, offset, src.read().cols, src.read().rows - offset);
    src.read()(read_chunk).copyTo(dst.read()(write_chunk));
    auto fill = cv::Rect(0, 0, src.read().cols, offset);
    dst.read()(fill).setTo(cv::Scalar(0));
#ifdef TRACK_STATISTICS
    src.inc_read();
    dst.inc_write();
#endif
}

void AnalogueBus::get_right(AnalogueRegister &dst, AnalogueRegister &src, int offset) {
    // x, y, width, height
    auto read_chunk =
        cv::Rect(offset, 0, src.read().cols - offset, src.read().rows);
    auto write_chunk =
        cv::Rect(0, 0, src.read().cols - offset, src.read().rows);
    src.read()(read_chunk).copyTo(dst.read()(write_chunk));
    auto fill =
        cv::Rect(src.read().cols - offset, 0, offset, src.read().rows);
    dst.read()(fill).setTo(cv::Scalar(0));
#ifdef TRACK_STATISTICS
    src.inc_read();
    dst.inc_write();
#endif
}

void AnalogueBus::get_left(AnalogueRegister &dst, AnalogueRegister &src, int offset) {
    // x, y, width, height
    auto read_chunk =
        cv::Rect(0, 0, src.read().cols - offset, src.read().rows);
    auto write_chunk =
        cv::Rect(offset, 0, src.read().cols - offset, src.read().rows);
    src.read()(read_chunk).copyTo(dst.read()(write_chunk));
    auto fill = cv::Rect(0, 0, offset, src.read().rows);
    dst.read()(fill).setTo(cv::Scalar(0));
#ifdef TRACK_STATISTICS
    src.inc_read();
    dst.inc_write();
#endif
}

void AnalogueBus::get_down(AnalogueRegister &dst, AnalogueRegister &src,
                          int offset) {
    // x, y, width, height
    auto read_chunk =
        cv::Rect(0, offset, src.read().cols, src.read().rows - offset);
    auto write_chunk =
        cv::Rect(0, 0, src.read().cols, src.read().rows - offset);
    src.read()(read_chunk).copyTo(dst.read()(write_chunk));
    auto fill =
        cv::Rect(0, src.read().rows - offset, src.read().cols, offset);
    dst.read()(fill).setTo(cv::Scalar(0));
#ifdef TRACK_STATISTICS
    src.inc_read();
    dst.inc_write();
#endif
}

void AnalogueBus::get_east(AnalogueRegister &dst, AnalogueRegister &src, int offset, Origin origin) {
    switch(origin) {
        case TOP_LEFT:
        case BOTTOM_LEFT: {
            get_right(dst, src, offset);
            break;
        }
        case TOP_RIGHT:
        case BOTTOM_RIGHT: {
            get_left(dst, src, offset);
            break;
        }
    }
}

void AnalogueBus::get_west(AnalogueRegister &dst, AnalogueRegister &src, int offset, Origin origin) {
    switch(origin) {
        case TOP_LEFT:
        case BOTTOM_LEFT: {
            get_left(dst, src, offset);
            break;
        }
        case TOP_RIGHT:
        case BOTTOM_RIGHT: {
            get_right(dst, src, offset);
            break;
        }
    }
}

void AnalogueBus::get_north(AnalogueRegister &dst, AnalogueRegister &src,
                           int offset, Origin origin) {
    switch(origin) {
        case BOTTOM_LEFT:
        case BOTTOM_RIGHT: {
            get_up(dst, src, offset);
            break;
        }
        case TOP_RIGHT:
        case TOP_LEFT: {
            get_down(dst, src, offset);
            break;
        }
    }
}

void AnalogueBus::get_south(AnalogueRegister &dst, AnalogueRegister &src,
                           int offset, Origin origin) {
    switch(origin) {
        case BOTTOM_LEFT:
        case BOTTOM_RIGHT: {
            get_down(dst, src, offset);
            break;
        }
        case TOP_RIGHT:
        case TOP_LEFT: {
            get_up(dst, src, offset);
            break;
        }
    }
}



// Higher level functions

void AnalogueBus::scan(uint8_t *dst, AnalogueRegister &src, uint8_t row_start,
                       uint8_t col_start, uint8_t row_end, uint8_t col_end, uint8_t row_step,
                       uint8_t col_step, Origin origin) {
    PlaneParams p;
    get_fixed_params(p, origin, row_start, col_start, row_end, col_end,
                     row_step, col_step);

    int buf_index = 0;
    for(int col = p.col_start; p.col_op(col, p.col_end); col += p.col_step) {
        for(int row = p.row_start; p.row_op(row, p.row_end); row += p.row_step) {
            dst[buf_index++] = src.read().at<uint8_t>(row, col);
        }
    }
}


void AnalogueBus::blocked_average(uint8_t *result, AnalogueRegister &src,
                                      int block_size, Origin origin) {
    // divide the AREG image into block_size x block_size square blocks, and get the average of each block
    // result - pointer to a buffer to store the results
    // origin - Where 0,0 is
    int rows = src.read().rows;
    int cols = src.read().cols;
    int step = rows / block_size;

    PlaneParams p;
    get_fixed_params(p, origin, 0, 0, rows, cols, step, step);

    int buf_index = 0;
    for(int col = p.col_start; p.col_op(col, p.col_end); col += p.col_step) {
        for(int row = p.row_start; p.row_op(row, p.row_end); row += p.row_step) {
            result[buf_index++] =
                cv::sum(src.read()(
                    cv::Rect(col, row, col + step, row + step)))[0] /
                (step * step);
        }
    }
}
