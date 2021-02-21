//
// Created by Jamal on 20/02/2021.
//

#include <iostream>
#include "scamp5.h"
#include <simulator/util/utility.h>
#include "instruction_factory.h"

#define START_TIMER() auto TIME_START = std::chrono::high_resolution_clock::now()
#define END_TIMER() auto TIME_END = std::chrono::high_resolution_clock::now(); std::cout << "Elapsed time: " \
<< std::chrono::duration_cast<std::chrono::milliseconds>(TIME_END-TIME_START).count() << " ms\n"

int main() {
    SCAMP5 s;

    s.scamp5_in(s.E, 10);
    int i = 0;




    std::string test = R"(
        get_image(C, D)
    )";

    // Assume we've parsed this and got the following
    std::string func_name = "get_image";
    std::string arg1 = "C";
    std::string arg2 =  "D";

    // Need to get the function using the type arguments. So need type arguments first


    // Get function
    
    auto get_image = InstructionFactory<SCAMP5>::get_instruction<AREG, AREG>("get_image");
    while(i < 250) {
        START_TIMER();
        (s.*get_image)(s.C, s.D);
        END_TIMER();
        s.get_image(s.C, s.D);

        s.sub(s.A, s.C, s.E);
        s.where(s.A);
        s.MOV(s.R5, s.FLAG);
        s.all();

        utility::display_register("PIX", s.PIX);
        utility::display_register("A", s.A);
        utility::display_register("B", s.B);
        utility::display_register("C", s.C);
        utility::display_register("D", s.D);
        utility::display_register("E", s.E);
        utility::display_register("FLAG", s.FLAG);
        utility::display_register("R5", s.R5);
        utility::display_register("NEWS", s.NEWS);
        cv::waitKey(1);
        i++;
        std::cout << "\rIteration " << i << std::endl;

    }

//    std::unordered_map<std::string, void (SCAMP5::*)()> instructions;
//    std::unordered_map<std::string, AnyCallable<void>()> instructions;


    return 0;
}
