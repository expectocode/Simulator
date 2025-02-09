//
// Created by jm1417 on 31/01/2021.
//

#ifndef SIMULATOR_REGISTER_H
#define SIMULATOR_REGISTER_H

#include <opencv4/opencv2/core/mat.hpp>

#include "simulator/base/component.h"
#include "simulator/metrics/cycle_counter.h"
#include "simulator/memory/memory.h"

#include <rttr/type>

// TODO need to be able to pass in some way of creating the underlying memory of registers. add as param
class Register : public Component {
    RTTR_ENABLE();
    // TODO internal mask
   protected:
    int rows_;
    int cols_;
    int row_stride_ = 1;
    int col_stride_ = 1;
    std::shared_ptr<Config> config_;
    std::shared_ptr<Memory> memory_;

   private:
    cv::Mat value_;

   public:
    std::string name_;
    int min_val = 0;
    int max_val = 0;

    Register(int rows, int columns, int row_stride, int col_stride, int type, const std::shared_ptr<Config>& config, MemoryType memoryType);

    Register(int rows, int columns, int row_stride, int col_stride, int type);

    void change_memory_type(MemoryType memory_type);

#ifdef TRACK_STATISTICS
    // Can only be used if memory_ is defined
    void update_static(double time) override;
    void inc_read(const cv::_InputOutputArray& mask);

    void inc_read();
    void inc_write(const cv::_InputOutputArray& mask);
    void inc_write();
    cv::Mat get_static_energy() override;
    cv::Mat get_dynamic_energy() override;
    cv::Mat get_transistor_count() override;
    int get_cycle_count() override;
    void print_stats(const CycleCounter& counter) override = 0;
//    void write_stats(const CycleCounter& counter, json& j) override = 0;
#endif

    cv::Mat& read();
    void write(cv::Mat& data);
    void write(const cv::Mat& data);
    void write(cv::Mat& data, cv::Mat& mask);
    void write(Register& data);
    void write(Register& data, Register& mask);
    void write(int data);
    void write(int data, cv::Mat& mask);
    void write(int data, Register& mask);
};

#endif  // SIMULATOR_REGISTER_H
