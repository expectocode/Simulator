//
// Created by jm1417 on 17/04/2021.
//

#include "simulator/memory/dram_array.h"

#include <rttr/registration>

void Dram::init() {
#ifdef TRACK_STATISTICS
    transistor_count_ = fun_transistor(config_);
    static_power_ = fun_static(config_);
    dynamic_power_ = fun_dynamic(config_);
    internal_mask = cv::Mat(rows_, cols_, CV_8U, cv::Scalar(0));
    array_transistor_count_ = cv::Mat(rows_, cols_, CV_32S, cv::Scalar(0));
    array_static_energy_ = cv::Mat(rows_, cols_, CV_64F, cv::Scalar(0));
    array_dynamic_energy_ = cv::Mat(rows_, cols_, CV_64F, cv::Scalar(0));
    this->fun_internal_mask();
#endif
    int sizes[] = {((rows_ * cols_) / row_stride_) / col_stride_, array_rows_, array_cols_};
    data = cv::Mat(3, sizes, CV_32S, cv::Scalar(0));
}

void Dram::set_rows(int rows) {
    this->rows_ = rows;
}

void Dram::set_cols(int cols) {
    this->cols_ = cols;
}

void Dram::set_row_stride(int row_stride) {
    this->row_stride_ = row_stride;
}

void Dram::set_col_stride(int col_stride) {
    this->col_stride_ = col_stride;
}

void Dram::set_array_rows(int array_rows) {
    this->array_rows_ = array_rows;
}

void Dram::set_array_cols(int array_cols) {
    this->array_cols_ = array_cols;
}

void Dram::set_word_length(int word_length) {
    this->word_length_ = word_length;
}

void Dram::set_config(const std::shared_ptr<Config>& config) {
    this->config_ = config;
}

int Dram::read(int array, int row, int col) {
    return data.at<int32_t>(array, row, col);
}

void Dram::write(int array, int row, int col, int value) {
    data.at<int32_t>(array, row, col) = value;
}

void Dram::reset() {
    data = 0;
}

#ifdef TRACK_STATISTICS
void Dram::fun_internal_mask() {
    for (int row = 0; row < rows_; row += row_stride_) {
        for (int col = 0; col < cols_; col += col_stride_) {
            this->internal_mask.at<uint8_t>(row, col) = 1;
        }
    }
    // todo move somewhere else
    array_transistor_count_.setTo(transistor_count_, this->internal_mask);
}

int Dram::fun_transistor(const std::shared_ptr<Config>& config) {
    // 1t1c DRAM cells (for now). So rows * cols * word_length for just DRAM cell transistor counts
    // Some number for sense amplifiers
    // Some number for row/col decoders
    return (array_rows_ * array_cols_ * word_length_);
}

double Dram::fun_static(const std::shared_ptr<Config>& config) {
    // 1t1c DRAM cells (for now)
    // Some amount for sense amplifiers
    // Some amount for row/col decoders
    // Refresh
    return 0;
}

double Dram::fun_dynamic(const std::shared_ptr<Config>& config) {
    // 1t1c DRAM cells (for now)
    // Some amount for sense amplifiers
    // Some amount for row/col decoders
    // read/write entire row. dependant on word length and cols
    return 0;
}

void Dram::update_static(double time) {
    cv::add(this->array_static_energy_, this->static_power_ * time, this->array_static_energy_, this->internal_mask);
}

int Dram::get_cycle_count() {
    return cycle_count_;
}

cv::Mat Dram::get_static_energy() {
    return array_static_energy_;
}

cv::Mat Dram::get_dynamic_energy() {
    return array_dynamic_energy_;
}

cv::Mat Dram::get_transistor_count() {
    return array_transistor_count_;
}

void Dram::print_stats(const CycleCounter& counter) {
}
#endif

RTTR_REGISTRATION {
    using namespace rttr;

    registration::class_<Dram>("Dram")
        .constructor()
        .method("init", &Dram::init)
        .method("set_rows", &Dram::set_rows)
        .method("set_cols", &Dram::set_cols)
        .method("set_row_stride", &Dram::set_row_stride)
        .method("set_col_stride", &Dram::set_col_stride)
        .method("set_array_rows", &Dram::set_array_rows)
        .method("set_array_cols", &Dram::set_array_cols)
        .method("set_word_length", &Dram::set_word_length)
        .method("set_config", &Dram::set_config);
};
