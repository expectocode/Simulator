//
// Created by jm1417 on 31/03/2021.
//

#include <catch2/catch.hpp>
#include <opencv2/core.hpp>
#include "../../include/simulator/registers/analogue_register.h"
#include "../../include/simulator/buses/analogue_bus.h"
#include "../utility.h"
#include "../../include/simulator/util/utility.h"

SCENARIO("analogue bus correctly manipulates analogue registers when mask=all") {
    int rows = 3;
    int cols = 3;
    AnalogueBus bus;

    GIVEN("some number of analogue registers and a mask") {

        DigitalRegister mask = (cv::Mat)(cv::Mat_<uint8_t>(rows, cols) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
        AnalogueRegister a = (cv::Mat)(cv::Mat_<int16_t>(rows, cols) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
        AnalogueRegister dst(rows, cols);

        WHEN("just one register") {
            THEN("it is set to 0") {
                bus.bus(a, mask);
                REQUIRE(cv::sum(a.value())[0] == 0);
            }
        }

        WHEN("two registers dst and a") {
            THEN("dst = -a") {
                bus.bus(dst, a, mask);
                cv::Mat expected = (cv::Mat)(cv::Mat_<int16_t>(rows, cols) << -1, -2, -3, -4, -5, -6, -7, -8, -9);
                REQUIRE(utility::are_mats_equal(dst.value(), expected));
            }
        }

        WHEN("3 registers, dst, a and b") {
            AnalogueRegister b = (cv::Mat)(cv::Mat_<int16_t>(rows, cols) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
            THEN("dst = -(a + b)") {
                bus.bus(dst, a, b, mask);
                cv::Mat expected = (cv::Mat)(cv::Mat_<int16_t>(rows, cols) << -2, -4, -6, -8, -10, -12, -14, -16, -18);
                REQUIRE(utility::are_mats_equal(dst.value(), expected));
            }
        }
    }
}

SCENARIO("digital registers can be correctly set based on simple conditions with analogue registers") {
    int rows = 3;
    int cols = 3;
    AnalogueBus bus;

    GIVEN("some number of analogue registers and a digital register") {
        int16_t a_data[3][3] = {{-1,2,3},{-1,5,6},{-1,8,9}};
        AnalogueRegister a = cv::Mat(3, 3, CV_16S, a_data);

        WHEN("1 analogue register") {
            DigitalRegister b(rows, cols);
            bus.conditional_positive_set(b, a);
            cv::Mat expected = (cv::Mat)(cv::Mat_<uint8_t>(rows, cols) << 0, 1, 1, 0, 1, 1, 0, 1, 1);
            REQUIRE(utility::are_mats_equal(b.value(), expected));
        }

    }

}