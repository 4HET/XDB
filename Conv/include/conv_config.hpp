//
//  conv_config.hpp
//  GEMM
//
//  Created by Chen lux on 2025/6/13.
//

#ifndef CONV_CONFIG_H
#define CONV_CONFIG_H

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <memory>

// 错误检查宏
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << hipGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// 针对16GB显存的配置
struct OptimalConfig {
    static constexpr int MAX_BATCH_SIZE = 32;
    static constexpr int MAX_CHANNELS = 1024;
    static constexpr int MAX_IMAGE_SIZE = 512;
    static constexpr int TILE_SIZE = 16;
    static constexpr int SHARED_MEM_BANKS = 32;
    static constexpr int WARP_SIZE = 64;  // DCU warp size
};

// 卷积参数结构
struct ConvParams {
    int batch_size;
    int input_channels;
    int output_channels;
    int input_height;
    int input_width;
    int kernel_height;
    int kernel_width;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    
    __host__ __device__
    int output_height() const {
        return (input_height + 2 * pad_h - kernel_height) / stride_h + 1;
    }
    
    __host__ __device__
    int output_width() const {
        return (input_width + 2 * pad_w - kernel_width) / stride_w + 1;
    }
    
    size_t input_size() const {
        return batch_size * input_channels * input_height * input_width;
    }
    
    size_t kernel_size() const {
        return output_channels * input_channels * kernel_height * kernel_width;
    }
    
    size_t output_size() const {
        return batch_size * output_channels * output_height() * output_width();
    }
    
    size_t memory_requirement_mb() const {
        return (input_size() + kernel_size() + output_size()) * sizeof(float) / (1024 * 1024);
    }
};

#endif
