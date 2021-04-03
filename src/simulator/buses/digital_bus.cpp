//
// Created by jm1417 on 28/01/2021.
//

#include "simulator/buses/digital_bus.h"

#include <opencv2/core.hpp>

#include "simulator/registers/analogue_register.h"

void DigitalBus::OR(DigitalRegister &d, DigitalRegister &d0,
                    DigitalRegister &d1) {
    // d := d0 OR d1
    cv::bitwise_or(d0.value(), d1.value(), d.value());
#ifdef TRACK_STATISTICS
    d.inc_write();
    d0.inc_read();
    d1.inc_read();
#endif
}

void DigitalBus::OR(DigitalRegister &d, DigitalRegister &d0,
                    DigitalRegister &d1, DigitalRegister &d2) {
    // d := d0 OR d1 OR d2
    cv::bitwise_or(d0.value(), d1.value(), d.value());
    cv::bitwise_or(d.value(), d2.value(), d.value());
#ifdef TRACK_STATISTICS
    d.inc_write();
    d0.inc_read();
    d1.inc_read();
    d2.inc_read();
#endif
}

void DigitalBus::OR(DigitalRegister &d, DigitalRegister &d0,
                    DigitalRegister &d1, DigitalRegister &d2,
                    DigitalRegister &d3) {
    // d := d0 OR d1 OR d2 OR d3
    cv::bitwise_or(d0.value(), d1.value(), d.value());
    cv::bitwise_or(d.value(), d2.value(), d.value());
    cv::bitwise_or(d.value(), d3.value(), d.value());
#ifdef TRACK_STATISTICS
    d.inc_write();
    d0.inc_read();
    d1.inc_read();
    d2.inc_read();
    d3.inc_write();
#endif
}

void DigitalBus::NOT(DigitalRegister &d, DigitalRegister &d0) {
    // d := NOT d0
    cv::bitwise_not(d0.value(), d.value());
#ifdef TRACK_STATISTICS
    d.inc_write();
    d0.inc_write();
#endif
}

void DigitalBus::NOR(DigitalRegister &d, DigitalRegister &d0,
                     DigitalRegister &d1) {
    // d := NOT(d0 OR d1)
    DigitalBus::OR(d, d0, d1);
    DigitalBus::NOT(d);
#ifdef TRACK_STATISTICS
    d.inc_write();
    d0.inc_read();
    d1.inc_read();
#endif
}

void DigitalBus::NOR(DigitalRegister &d, DigitalRegister &d0,
                     DigitalRegister &d1, DigitalRegister &d2) {
    // d := NOT(d0 OR d1 OR d2)
    DigitalBus::OR(d, d0, d1, d2);
    DigitalBus::NOT(d);
#ifdef TRACK_STATISTICS
    d.inc_write();
    d0.inc_read();
    d1.inc_read();
    d2.inc_read();
#endif
}

void DigitalBus::NOR(DigitalRegister &d, DigitalRegister &d0,
                     DigitalRegister &d1, DigitalRegister &d2,
                     DigitalRegister &d3) {
    // d := NOT(d0 OR d1 OR d2 OR d3)
    DigitalBus::OR(d, d0, d1, d2, d3);
    DigitalBus::NOT(d);
#ifdef TRACK_STATISTICS
    d.inc_write();
    d0.inc_read();
    d1.inc_read();
    d2.inc_read();
    d3.inc_read();
#endif
}

void DigitalBus::NOT(DigitalRegister &Rl) {
    // Rl := NOT Rl
    cv::bitwise_not(Rl.value(), Rl.value());
#ifdef TRACK_STATISTICS
    Rl.inc_write();
    Rl.inc_read();
#endif
}

void DigitalBus::OR(DigitalRegister &Rl, DigitalRegister &Rx) {
    // Rl := Rl OR Rx
    DigitalBus::OR(Rl, Rl, Rx);
#ifdef TRACK_STATISTICS
    Rl.inc_write();
    Rl.inc_read();
    Rx.inc_read();
#endif
}

void DigitalBus::NOR(DigitalRegister &Rl, DigitalRegister &Rx) {
    // Rl := Rl NOR Rx
    DigitalBus::NOR(Rl, Rl, Rx);
#ifdef TRACK_STATISTICS
    Rl.inc_write();
    Rl.inc_read();
    Rx.inc_read();
#endif
}

void DigitalBus::AND(DigitalRegister &Ra, DigitalRegister &Rx,
                     DigitalRegister &Ry) {
    // Ra := Rx AND Ry
    cv::bitwise_and(Rx.value(), Ry.value(), Ra.value());
#ifdef TRACK_STATISTICS
    Ra.inc_write();
    Rx.inc_read();
    Ry.inc_read();
#endif
}

void DigitalBus::NAND(DigitalRegister &Ra, DigitalRegister &Rx,
                      DigitalRegister &Ry) {
    // Ra = NOT(Rx AND Ry)
    DigitalBus::AND(Ra, Rx, Ry);
    DigitalBus::NOT(Ra);
#ifdef TRACK_STATISTICS
    Ra.inc_write();
    Rx.inc_read();
    Ry.inc_read();
#endif
}

void DigitalBus::IMP(DigitalRegister &Rl, DigitalRegister &Rx,
                     DigitalRegister &Ry) {
    // Rl := Rx IMP Ry (logical implication)
    //    Truth Table:
    //    Rx  Ry    Rl
    //    0   0     1
    //    0   1     0
    //    1   0     1
    //    1   1     1
    DigitalRegister intermediate(Rl.value().rows, Rl.value().cols);
    DigitalBus::NOT(intermediate, Ry);
    DigitalBus::OR(Rl, Rx, intermediate);
#ifdef TRACK_STATISTICS
    Rl.inc_write();
    Rx.inc_read();
    Ry.inc_read();
#endif
}

void DigitalBus::NIMP(DigitalRegister &Rl, DigitalRegister &Rx,
                      DigitalRegister &Ry) {
    // Rl := Rx NIMP Ry
    DigitalRegister intermediate(Rl.value().rows, Rl.value().cols);
    DigitalBus::NOT(intermediate);
    DigitalBus::NOR(Rl, Rx, intermediate);
#ifdef TRACK_STATISTICS
    Rl.inc_write();
    Rx.inc_read();
    Ry.inc_read();
#endif
}

void DigitalBus::XOR(DigitalRegister &Rl, DigitalRegister &Rx,
                     DigitalRegister &Ry) {
    // Rl := Rx XOR Ry
    cv::bitwise_xor(Rx.value(), Ry.value(), Rl.value());
#ifdef TRACK_STATISTICS
    Rl.inc_write();
    Rx.inc_read();
    Ry.inc_read();
#endif
}

void DigitalBus::MOV(DigitalRegister &d, DigitalRegister &d0) {
    // d := d0
    cv::copyTo(d0.value(), d.value(), cv::noArray());
#ifdef TRACK_STATISTICS
    d.inc_write();
    d0.inc_read();
#endif
}

void DigitalBus::MUX(DigitalRegister &Rl, DigitalRegister &Rx,
                     DigitalRegister &Ry, DigitalRegister &Rz) {
    // Rl := Ry IF Rx = 1, Rl := Rz IF Rx = 0.
    // R1 = (Ry.~Rx) + (Rz.Rx)
    DigitalRegister intermediate(Rl.value().rows, Rl.value().cols);
    DigitalRegister intermediate2(Rl.value().rows, Rl.value().cols);
    cv::bitwise_not(Rx.value(), intermediate.value());
    DigitalBus::AND(intermediate2, Ry, intermediate);
    DigitalBus::AND(intermediate, Rz, Rx);
    DigitalBus::OR(Rl, intermediate, intermediate2);
#ifdef TRACK_STATISTICS
    // TODO fix reads
    Rl.inc_write();
    Rx.inc_read();
    Ry.inc_read();
    Rz.inc_read();
#endif
}

void DigitalBus::CLR_IF(DigitalRegister &Rl, DigitalRegister &Rx) {
    // Rl := 0 IF Rx = 1, Rl := Rl IF Rx = 0
    DigitalRegister intermediate(Rl.value().rows, Rl.value().cols);
    DigitalBus::NOT(intermediate, Rl);
    DigitalBus::NOR(Rl, intermediate, Rx);
#ifdef TRACK_STATISTICS
    // TODO fix reads
    Rl.inc_write();
    Rx.inc_read();
#endif
}

// Masked

void DigitalBus::OR_MASKED(DigitalRegister &d, DigitalRegister &d0,
                           DigitalRegister &d1, DigitalRegister &FLAG) {
    // d := d0 OR d1
    cv::bitwise_or(d0.value(), d1.value(), d.value(), FLAG.value());
}

void DigitalBus::OR_MASKED(DigitalRegister &d, DigitalRegister &d0,
                           DigitalRegister &d1, DigitalRegister &d2,
                           DigitalRegister &FLAG) {
    // d := d0 OR d1 OR d2
    cv::bitwise_or(d0.value(), d1.value(), d.value(), FLAG.value());
    cv::bitwise_or(d.value(), d2.value(), d.value(), FLAG.value());
}

void DigitalBus::OR_MASKED(DigitalRegister &d, DigitalRegister &d0,
                           DigitalRegister &d1, DigitalRegister &d2,
                           DigitalRegister &d3, DigitalRegister &FLAG) {
    // d := d0 OR d1 OR d2 OR d3
    cv::bitwise_or(d0.value(), d1.value(), d.value(), FLAG.value());
    cv::bitwise_or(d.value(), d2.value(), d.value(), FLAG.value());
    cv::bitwise_or(d.value(), d3.value(), d.value(), FLAG.value());
}

void DigitalBus::NOT_MASKED(DigitalRegister &d, DigitalRegister &d0,
                            DigitalRegister &FLAG) {
    // d := NOT d0
    cv::bitwise_not(d0.value(), d.value(), FLAG.value());
}

void DigitalBus::NOR_MASKED(DigitalRegister &d, DigitalRegister &d0,
                            DigitalRegister &d1, DigitalRegister &FLAG) {
    // d := NOT(d0 OR d1)
    DigitalBus::OR_MASKED(d, d0, d1, FLAG);
    DigitalBus::NOT_MASKED(d, FLAG);
}

void DigitalBus::NOR_MASKED(DigitalRegister &d, DigitalRegister &d0,
                            DigitalRegister &d1, DigitalRegister &d2,
                            DigitalRegister &FLAG) {
    // d := NOT(d0 OR d1 OR d2)
    DigitalBus::OR_MASKED(d, d0, d1, d2, FLAG);
    DigitalBus::NOT_MASKED(d, FLAG);
}

void DigitalBus::NOR_MASKED(DigitalRegister &d, DigitalRegister &d0,
                            DigitalRegister &d1, DigitalRegister &d2,
                            DigitalRegister &d3, DigitalRegister &FLAG) {
    // d := NOT(d0 OR d1 OR d2 OR d3)
    DigitalBus::OR_MASKED(d, d0, d1, d2, d3, FLAG);
    DigitalBus::NOT_MASKED(d, FLAG);
}

void DigitalBus::NOT_MASKED(DigitalRegister &Rl, DigitalRegister &FLAG) {
    // Rl := NOT Rl
    cv::bitwise_not(Rl.value(), Rl.value(), FLAG.value());
}

void DigitalBus::OR_MASKED(DigitalRegister &Rl, DigitalRegister &Rx,
                           DigitalRegister &FLAG) {
    // Rl := Rl OR Rx
    DigitalBus::OR_MASKED(Rl, Rl, Rx, FLAG);
}

void DigitalBus::NOR_MASKED(DigitalRegister &Rl, DigitalRegister &Rx,
                            DigitalRegister &FLAG) {
    // Rl := Rl NOR Rx
    DigitalBus::NOR_MASKED(Rl, Rl, Rx, FLAG);
}

void DigitalBus::AND_MASKED(DigitalRegister &Ra, DigitalRegister &Rx,
                            DigitalRegister &Ry, DigitalRegister &FLAG) {
    // Ra := Rx AND Ry
    cv::bitwise_and(Rx.value(), Ry.value(), Ra.value(), FLAG.value());
}

void DigitalBus::NAND_MASKED(DigitalRegister &Ra, DigitalRegister &Rx,
                             DigitalRegister &Ry, DigitalRegister &FLAG) {
    // Ra = NOT(Rx AND Ry)
    DigitalBus::AND_MASKED(Ra, Rx, Ry, FLAG);
    DigitalBus::NOT_MASKED(Ra, FLAG);
}

void DigitalBus::IMP_MASKED(DigitalRegister &Rl, DigitalRegister &Rx,
                            DigitalRegister &Ry, DigitalRegister &FLAG) {
    // Rl := Rx IMP Ry (logical implication)
    //    Truth Table:
    //    Rx  Ry    Rl
    //    0   0     1
    //    0   1     0
    //    1   0     1
    //    1   1     1
    DigitalRegister intermediate(Rl.value().rows, Rl.value().cols);
    DigitalBus::NOT_MASKED(intermediate, Ry, FLAG);
    DigitalBus::OR_MASKED(Rl, Rx, intermediate, FLAG);
}

void DigitalBus::NIMP_MASKED(DigitalRegister &Rl, DigitalRegister &Rx,
                             DigitalRegister &Ry, DigitalRegister &FLAG) {
    // Rl := Rx NIMP Ry
    DigitalRegister intermediate(Rl.value().rows, Rl.value().cols);
    DigitalBus::NOR_MASKED(intermediate, Ry, FLAG);
    DigitalBus::NOR_MASKED(Rl, Rx, intermediate, FLAG);
}

void DigitalBus::XOR_MASKED(DigitalRegister &Rl, DigitalRegister &Rx,
                            DigitalRegister &Ry, DigitalRegister &FLAG) {
    // Rl := Rx XOR Ry
    cv::bitwise_xor(Rx.value(), Ry.value(), Rl.value(), FLAG.value());
}

void DigitalBus::MOV_MASKED(DigitalRegister &d, DigitalRegister &d0,
                            DigitalRegister &FLAG) {
    // d := d0
    cv::copyTo(d0.value(), d.value(), FLAG.value());
}

void DigitalBus::MUX_MASKED(DigitalRegister &Rl, DigitalRegister &Rx,
                            DigitalRegister &Ry, DigitalRegister &Rz,
                            DigitalRegister &FLAG) {
    // Rl := Ry IF Rx = 1, Rl := Rz IF Rx = 0.
    // R1 = (Ry.~Rx) + (Rz.Rx)
    DigitalRegister intermediate(Rl.value().rows, Rl.value().cols);
    DigitalRegister intermediate2(Rl.value().rows, Rl.value().cols);
    cv::bitwise_not(Rx.value(), intermediate.value());
    DigitalBus::AND_MASKED(intermediate2, Ry, intermediate, FLAG);
    DigitalBus::AND_MASKED(intermediate, Rz, Rx, FLAG);
    DigitalBus::OR_MASKED(Rl, intermediate, intermediate2, FLAG);
}

void DigitalBus::CLR_IF_MASKED(DigitalRegister &Rl, DigitalRegister &Rx,
                               DigitalRegister &FLAG) {
    // Rl := 0 IF Rx = 1, Rl := Rl IF Rx = 0
    DigitalRegister intermediate(Rl.value().rows, Rl.value().cols);
    DigitalBus::NOT_MASKED(intermediate, Rl, FLAG);
    DigitalBus::NOR_MASKED(Rl, intermediate, Rx, FLAG);
}

// Neighbour Operations

void DigitalBus::get_up(DigitalRegister &dst, DigitalRegister &src, int offset,
                        int boundary_fill) {
    // x, y, width, height
    auto read_chunk =
        cv::Rect(0, 0, src.value().cols, src.value().rows - offset);
    auto write_chunk =
        cv::Rect(0, offset, src.value().cols, src.value().rows - offset);
    src.value()(read_chunk).copyTo(dst.value()(write_chunk));
    auto fill = cv::Rect(0, 0, src.value().cols, offset);
    dst.value()(fill).setTo(cv::Scalar(boundary_fill));
#ifdef TRACK_STATISTICS
    src.inc_read();
    dst.inc_write();
#endif
}

void DigitalBus::get_right(DigitalRegister &dst, DigitalRegister &src,
                           int offset, int boundary_fill) {
    // x, y, width, height
    auto read_chunk =
        cv::Rect(offset, 0, src.value().cols - offset, src.value().rows);
    auto write_chunk =
        cv::Rect(0, 0, src.value().cols - offset, src.value().rows);
    src.value()(read_chunk).copyTo(dst.value()(write_chunk));
    auto fill =
        cv::Rect(src.value().cols - offset, 0, offset, src.value().rows);
    dst.value()(fill).setTo(cv::Scalar(boundary_fill));
#ifdef TRACK_STATISTICS
    src.inc_read();
    dst.inc_write();
#endif
}

void DigitalBus::get_left(DigitalRegister &dst, DigitalRegister &src,
                          int offset, int boundary_fill) {
    // x, y, width, height
    auto read_chunk =
        cv::Rect(0, 0, src.value().cols - offset, src.value().rows);
    auto write_chunk =
        cv::Rect(offset, 0, src.value().cols - offset, src.value().rows);
    src.value()(read_chunk).copyTo(dst.value()(write_chunk));
    auto fill = cv::Rect(0, 0, offset, src.value().rows);
    dst.value()(fill).setTo(cv::Scalar(boundary_fill));
#ifdef TRACK_STATISTICS
    src.inc_read();
    dst.inc_write();
#endif
}

void DigitalBus::get_down(DigitalRegister &dst, DigitalRegister &src,
                          int offset, int boundary_fill) {
    // x, y, width, height
    auto read_chunk =
        cv::Rect(0, offset, src.value().cols, src.value().rows - offset);
    auto write_chunk =
        cv::Rect(0, 0, src.value().cols, src.value().rows - offset);
    src.value()(read_chunk).copyTo(dst.value()(write_chunk));
    auto fill =
        cv::Rect(0, src.value().rows - offset, src.value().cols, offset);
    dst.value()(fill).setTo(cv::Scalar(boundary_fill));
#ifdef TRACK_STATISTICS
    src.inc_read();
    dst.inc_write();
#endif
}

void DigitalBus::get_east(DigitalRegister &dst, DigitalRegister &src,
                          int offset, int boundary_fill, Origin origin) {
    switch(origin) {
        case TOP_LEFT:
        case BOTTOM_LEFT: {
            get_right(dst, src, offset, boundary_fill);
            break;
        }
        case TOP_RIGHT:
        case BOTTOM_RIGHT: {
            get_left(dst, src, offset, boundary_fill);
            break;
        }
    }
}

void DigitalBus::get_west(DigitalRegister &dst, DigitalRegister &src,
                          int offset, int boundary_fill, Origin origin) {
    switch(origin) {
        case TOP_LEFT:
        case BOTTOM_LEFT: {
            get_left(dst, src, offset, boundary_fill);
            break;
        }
        case TOP_RIGHT:
        case BOTTOM_RIGHT: {
            get_right(dst, src, offset, boundary_fill);
            break;
        }
    }
}

void DigitalBus::get_north(DigitalRegister &dst, DigitalRegister &src,
                           int offset, int boundary_fill, Origin origin) {
    switch(origin) {
        case BOTTOM_LEFT:
        case BOTTOM_RIGHT: {
            get_up(dst, src, offset, boundary_fill);
            break;
        }
        case TOP_RIGHT:
        case TOP_LEFT: {
            get_down(dst, src, offset, boundary_fill);
            break;
        }
    }
}

void DigitalBus::get_south(DigitalRegister &dst, DigitalRegister &src,
                           int offset, int boundary_fill, Origin origin) {
    switch(origin) {
        case BOTTOM_LEFT:
        case BOTTOM_RIGHT: {
            get_down(dst, src, offset, boundary_fill);
            break;
        }
        case TOP_RIGHT:
        case TOP_LEFT: {
            get_up(dst, src, offset, boundary_fill);
            break;
        }
    }
}

// SuperPixel Operations

void DigitalBus::superpixel_create(
    DigitalRegister &dst, AnalogueRegister &src,
    const std::unordered_map<int, cv::Point> &locations) {
    // Converts an analogue image to a digital superpixel format
    // For now assume 1 bank

    int bank_size = 4;
    int num_banks = 1;
    int pixel_size = 2;

    for(int col = 0; col < src.value().cols; col += pixel_size) {
        for(int row = 0; row < src.value().rows; row += pixel_size) {
            int sum = cv::sum(
                src.value()(cv::Rect(col, row, pixel_size, pixel_size)))[0];
            sum /= (pixel_size * pixel_size);  // <- this truncates values
            // convert value to bit array of length n. LSB first
            // write out bits to block in correct order - assume 16 bit snake
            // for now Need to be able to access bank. So something like
            for(int i = 0; i < pixel_size * pixel_size; ++i) {
                int bit = (sum >> i) & 1;
                cv::Point relative_pos =
                    locations.at(i + 1);  // bitorder snake starts at 1 not 0
                dst.value().at<uint8_t>(relative_pos.y + row,
                                        relative_pos.x + col) = bit;
            }
        }
    }
}

void DigitalBus::superpixel_shift_right(DigitalRegister &dst,
                                        DigitalRegister &src, Origin origin) {
    int rows = src.value().rows;
    int cols = src.value().cols;
    DigitalRegister east = DigitalRegister(rows, cols);
    DigitalRegister north = DigitalRegister(rows, cols);
    DigitalRegister west = DigitalRegister(rows, cols);
    DigitalRegister south = DigitalRegister(rows, cols);

    DigitalRegister R_NORTH =
        (cv::Mat)(cv::Mat_<uint8_t>(rows, cols) << 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                  0, 1, 0, 0, 0, 0);
    DigitalRegister R_SOUTH =
        (cv::Mat)(cv::Mat_<uint8_t>(rows, cols) << 0, 0, 0, 0, 1, 0, 1, 0, 1, 0,
                  1, 0, 1, 0, 1, 0);
    DigitalRegister R_EAST = (cv::Mat)(cv::Mat_<uint8_t>(rows, cols) << 0, 0, 1,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1);
    DigitalRegister R_WEST = (cv::Mat)(cv::Mat_<uint8_t>(rows, cols) << 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    get_east(east, src, 1, 0, origin);
    get_north(north, src, 1, 0, origin);
    get_west(west, src, 1, 0, origin);
    get_south(south, src, 1, 0, origin);

    AND(east, east, R_EAST);
    AND(north, north, R_NORTH);
    AND(west, west, R_WEST);
    AND(south, south, R_SOUTH);

    OR(dst, east, north, south, west);
}

void DigitalBus::positions_from_bitorder(
    std::vector<std::vector<std::vector<int>>> bitorder, int banks, int height,
    int width, std::unordered_map<int, cv::Point> &locations) {
    // For now we only care about bank 1.
    for(int b = 0; b < banks; b++) {
        for(int h = 0; h < height; h++) {
            for(int w = 0; w < width; w++) {
                int index = bitorder[0][h][w];
                // todo double check w,h
                locations[index] = cv::Point(w, h);
            }
        }
    }
}
