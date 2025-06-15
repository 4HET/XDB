//
//  conv_config.cpp
//  GEMM
//
//  Created by Chen lux on 2025/6/13.
//

#include "include/conv_config.hpp"
#include <hip/hip_runtime.h>
#include <omp.h>
#include <random>
#include <fstream>
#include <iomanip>

// 基础卷积核函数
__global__ void conv_kernel_basic(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    ConvParams params
) {
    int batch = blockIdx.x;
    int out_c = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    int output_h = params.output_height();
    int output_w = params.output_width();
    int total_output = output_h * output_w;
    
    for (int idx = tid; idx < total_output; idx += block_size) {
        int out_h = idx / output_w;
        int out_w = idx % output_w;
        
        float sum = 0.0f;
        
        for (int in_c = 0; in_c < params.input_channels; in_c++) {
            for (int kh = 0; kh < params.kernel_height; kh++) {
                for (int kw = 0; kw < params.kernel_width; kw++) {
                    int in_h = out_h * params.stride_h - params.pad_h + kh;
                    int in_w = out_w * params.stride_w - params.pad_w + kw;
                    
                    if (in_h >= 0 && in_h < params.input_height &&
                        in_w >= 0 && in_w < params.input_width) {
                        
                        int input_idx = ((batch * params.input_channels + in_c) *
                                       params.input_height + in_h) * params.input_width + in_w;
                        int kernel_idx = ((out_c * params.input_channels + in_c) *
                                        params.kernel_height + kh) * params.kernel_width + kw;
                        
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
        }
        
        int output_idx = ((batch * params.output_channels + out_c) *
                         output_h + out_h) * output_w + out_w;
        output[output_idx] = sum;
    }
    return;
}

// 优化的卷积核函数 - 使用共享内存和向量化
__global__ void conv_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    ConvParams params
) {
    extern __shared__ float shared_data[];
    
    int batch = blockIdx.x;
    int out_c = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // 分配共享内存
    float* shared_kernel = shared_data;
    float* shared_input = shared_kernel + params.input_channels * params.kernel_height * params.kernel_width;
    
    int kernel_size = params.input_channels * params.kernel_height * params.kernel_width;
    
    // 协作加载kernel到共享内存
    for (int i = tid; i < kernel_size; i += block_size) {
        int kernel_idx = out_c * kernel_size + i;
        shared_kernel[i] = kernel[kernel_idx];
    }
    __syncthreads();
    
    int output_h = params.output_height();
    int output_w = params.output_width();
    int total_output = output_h * output_w;
    
    // 处理多个输出像素
    for (int idx = tid; idx < total_output; idx += block_size) {
        int out_h = idx / output_w;
        int out_w = idx % output_w;
        
        float sum = 0.0f;
        
        // 使用共享内存中的kernel
        for (int in_c = 0; in_c < params.input_channels; in_c++) {
            for (int kh = 0; kh < params.kernel_height; kh++) {
                for (int kw = 0; kw < params.kernel_width; kw++) {
                    int in_h = out_h * params.stride_h - params.pad_h + kh;
                    int in_w = out_w * params.stride_w - params.pad_w + kw;
                    
                    if (in_h >= 0 && in_h < params.input_height &&
                        in_w >= 0 && in_w < params.input_width) {
                        
                        int input_idx = ((batch * params.input_channels + in_c) *
                                       params.input_height + in_h) * params.input_width + in_w;
                        int shared_kernel_idx = (in_c * params.kernel_height + kh) *
                                              params.kernel_width + kw;
                        
                        sum += input[input_idx] * shared_kernel[shared_kernel_idx];
                    }
                }
            }
        }
        
        int output_idx = ((batch * params.output_channels + out_c) *
                         output_h + out_h) * output_w + out_w;
        output[output_idx] = sum;
    }
    return;
}

// 分块卷积核函数 - 针对大型输入优化
__global__ void conv_kernel_tiled(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    ConvParams params
) {
    extern __shared__ float shared_data[];
    
    const int TILE_H = OptimalConfig::TILE_SIZE;
    const int TILE_W = OptimalConfig::TILE_SIZE;
    
    int batch = blockIdx.x;
    int out_c = blockIdx.y;
    int tile_h = blockIdx.z / ((params.output_width() + TILE_W - 1) / TILE_W);
    int tile_w = blockIdx.z % ((params.output_width() + TILE_W - 1) / TILE_W);
    
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid = tid_y * blockDim.x + tid_x;
    
    float* shared_input = shared_data;
    float* shared_kernel = shared_input + (TILE_H + params.kernel_height - 1) *
                          (TILE_W + params.kernel_width - 1) * params.input_channels;
    
    // 加载kernel到共享内存
    int kernel_size = params.input_channels * params.kernel_height * params.kernel_width;
    for (int i = tid; i < kernel_size; i += blockDim.x * blockDim.y) {
        int kernel_idx = out_c * kernel_size + i;
        shared_kernel[i] = kernel[kernel_idx];
    }
    
    // 加载输入tile到共享内存
    int input_tile_h = TILE_H + params.kernel_height - 1;
    int input_tile_w = TILE_W + params.kernel_width - 1;
    
    for (int in_c = 0; in_c < params.input_channels; in_c++) {
        for (int h = tid_y; h < input_tile_h; h += blockDim.y) {
            for (int w = tid_x; w < input_tile_w; w += blockDim.x) {
                int global_h = tile_h * TILE_H + h - params.pad_h;
                int global_w = tile_w * TILE_W + w - params.pad_w;
                
                float value = 0.0f;
                if (global_h >= 0 && global_h < params.input_height &&
                    global_w >= 0 && global_w < params.input_width) {
                    int input_idx = ((batch * params.input_channels + in_c) *
                                   params.input_height + global_h) * params.input_width + global_w;
                    value = input[input_idx];
                }
                
                int shared_idx = (in_c * input_tile_h + h) * input_tile_w + w;
                shared_input[shared_idx] = value;
            }
        }
    }
    
    __syncthreads();
    
    // 计算输出
    int out_h = tile_h * TILE_H + tid_y;
    int out_w = tile_w * TILE_W + tid_x;
    
    if (out_h < params.output_height() && out_w < params.output_width()) {
        float sum = 0.0f;
        
        for (int in_c = 0; in_c < params.input_channels; in_c++) {
            for (int kh = 0; kh < params.kernel_height; kh++) {
                for (int kw = 0; kw < params.kernel_width; kw++) {
                    int shared_h = tid_y + kh;
                    int shared_w = tid_x + kw;
                    int shared_input_idx = (in_c * input_tile_h + shared_h) * input_tile_w + shared_w;
                    int shared_kernel_idx = (in_c * params.kernel_height + kh) * params.kernel_width + kw;
                    
                    sum += shared_input[shared_input_idx] * shared_kernel[shared_kernel_idx];
                }
            }
        }
        
        int output_idx = ((batch * params.output_channels + out_c) *
                         params.output_height() + out_h) * params.output_width() + out_w;
        output[output_idx] = sum;
    }
    return;
}

class DCUConvolutionEngine {
private:
    float *d_input, *d_kernel, *d_output;
    ConvParams params;
    bool memory_allocated;
    hipStream_t stream;
    
public:
    DCUConvolutionEngine(const ConvParams& p) : params(p), memory_allocated(false) {
        HIP_CHECK(hipStreamCreate(&stream));
        allocateMemory();
    }
    
    ~DCUConvolutionEngine() {
        freeMemory();
        hipStreamDestroy(stream);
    }
    
    void allocateMemory() {
        if (memory_allocated) return;
        
        size_t input_bytes = params.input_size() * sizeof(float);
        size_t kernel_bytes = params.kernel_size() * sizeof(float);
        size_t output_bytes = params.output_size() * sizeof(float);
        size_t total_mb = (input_bytes + kernel_bytes + output_bytes) / (1024 * 1024);
        
        if (total_mb > 15 * 1024) {  // 留1GB给系统
            throw std::runtime_error("Memory requirement exceeds 15GB limit");
        }
        
        HIP_CHECK(hipMalloc(&d_input, input_bytes));
        HIP_CHECK(hipMalloc(&d_kernel, kernel_bytes));
        HIP_CHECK(hipMalloc(&d_output, output_bytes));
        
        memory_allocated = true;
        
        std::cout << "GPU Memory allocated: " << total_mb << " MB" << std::endl;
        std::cout << "  Input: " << input_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Kernel: " << kernel_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Output: " << output_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    }
    
    void freeMemory() {
        if (!memory_allocated) return;
        hipFree(d_input);
        hipFree(d_kernel);
        hipFree(d_output);
        memory_allocated = false;
    }
    
    double runBasicConvolution(const std::vector<float>& input,
                              const std::vector<float>& kernel,
                              std::vector<float>& output) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // 异步数据传输
        HIP_CHECK(hipMemcpyAsync(d_input, input.data(), input.size() * sizeof(float),
                                hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_kernel, kernel.data(), kernel.size() * sizeof(float),
                                hipMemcpyHostToDevice, stream));
        
        // 配置kernel参数
        dim3 blockSize(256);
        dim3 gridSize(params.batch_size, params.output_channels);
        
        // 启动kernel
        hipLaunchKernelGGL(conv_kernel_basic, gridSize, blockSize, 0, stream,
                          d_input, d_kernel, d_output, params);
        
        // 异步拷贝结果
        HIP_CHECK(hipMemcpyAsync(output.data(), d_output, output.size() * sizeof(float),
                                hipMemcpyDeviceToHost, stream));
        
        HIP_CHECK(hipStreamSynchronize(stream));
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }
    
    double runOptimizedConvolution(const std::vector<float>& input,
                                  const std::vector<float>& kernel,
                                  std::vector<float>& output) {
        auto start = std::chrono::high_resolution_clock::now();
        
        HIP_CHECK(hipMemcpyAsync(d_input, input.data(), input.size() * sizeof(float),
                                hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_kernel, kernel.data(), kernel.size() * sizeof(float),
                                hipMemcpyHostToDevice, stream));
        
        // 计算共享内存需求
        size_t shared_mem_size = (params.input_channels * params.kernel_height * params.kernel_width +
                                 256 * params.input_channels) * sizeof(float);
        
        dim3 blockSize(256);
        dim3 gridSize(params.batch_size, params.output_channels);
        
        hipLaunchKernelGGL(conv_kernel_optimized, gridSize, blockSize, shared_mem_size, stream,
                          d_input, d_kernel, d_output, params);
        
        HIP_CHECK(hipMemcpyAsync(output.data(), d_output, output.size() * sizeof(float),
                                hipMemcpyDeviceToHost, stream));
        
        HIP_CHECK(hipStreamSynchronize(stream));
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }
    
    double runTiledConvolution(const std::vector<float>& input,
                              const std::vector<float>& kernel,
                              std::vector<float>& output) {
        auto start = std::chrono::high_resolution_clock::now();
        
        HIP_CHECK(hipMemcpyAsync(d_input, input.data(), input.size() * sizeof(float),
                                hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_kernel, kernel.data(), kernel.size() * sizeof(float),
                                hipMemcpyHostToDevice, stream));
        
        const int TILE_SIZE = OptimalConfig::TILE_SIZE;
        int tiles_h = (params.output_height() + TILE_SIZE - 1) / TILE_SIZE;
        int tiles_w = (params.output_width() + TILE_SIZE - 1) / TILE_SIZE;
        
        size_t shared_mem_size = ((TILE_SIZE + params.kernel_height - 1) *
                                 (TILE_SIZE + params.kernel_width - 1) * params.input_channels +
                                 params.input_channels * params.kernel_height * params.kernel_width) * sizeof(float);
        
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize(params.batch_size, params.output_channels, tiles_h * tiles_w);
        
        hipLaunchKernelGGL(conv_kernel_tiled, gridSize, blockSize, shared_mem_size, stream,
                          d_input, d_kernel, d_output, params);
        
        HIP_CHECK(hipMemcpyAsync(output.data(), d_output, output.size() * sizeof(float),
                                hipMemcpyDeviceToHost, stream));
        
        HIP_CHECK(hipStreamSynchronize(stream));
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }
};

// CPU参考实现
class CPUConvolution {
private:
    ConvParams params;
    
public:
    CPUConvolution(const ConvParams& p) : params(p) {}
    
    double compute(const std::vector<float>& input,
                  const std::vector<float>& kernel,
                  std::vector<float>& output) {
        auto start = std::chrono::high_resolution_clock::now();
        
        int output_h = params.output_height();
        int output_w = params.output_width();
        
        #pragma omp parallel for collapse(4)
        for (int batch = 0; batch < params.batch_size; batch++) {
            for (int out_c = 0; out_c < params.output_channels; out_c++) {
                for (int out_h = 0; out_h < output_h; out_h++) {
                    for (int out_w = 0; out_w < output_w; out_w++) {
                        float sum = 0.0f;
                        
                        for (int in_c = 0; in_c < params.input_channels; in_c++) {
                            for (int kh = 0; kh < params.kernel_height; kh++) {
                                for (int kw = 0; kw < params.kernel_width; kw++) {
                                    int in_h = out_h * params.stride_h - params.pad_h + kh;
                                    int in_w = out_w * params.stride_w - params.pad_w + kw;
                                    
                                    if (in_h >= 0 && in_h < params.input_height &&
                                        in_w >= 0 && in_w < params.input_width) {
                                        
                                        int input_idx = ((batch * params.input_channels + in_c) *
                                                       params.input_height + in_h) * params.input_width + in_w;
                                        int kernel_idx = ((out_c * params.input_channels + in_c) *
                                                        params.kernel_height + kh) * params.kernel_width + kw;
                                        
                                        sum += input[input_idx] * kernel[kernel_idx];
                                    }
                                }
                            }
                        }
                        
                        int output_idx = ((batch * params.output_channels + out_c) *
                                         output_h + out_h) * output_w + out_w;
                        output[output_idx] = sum;
                    }
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }
};

// 性能测试和验证类
class ConvolutionBenchmark {
public:
    struct TestResult {
        double cpu_time;
        double gpu_basic_time;
        double gpu_optimized_time;
        double gpu_tiled_time;
        bool basic_correct;
        bool optimized_correct;
        bool tiled_correct;
        double speedup_basic;
        double speedup_optimized;
        double speedup_tiled;
    };
    
public:
    static std::vector<float> generateRandomData(size_t size, float min_val = -1.0f, float max_val = 1.0f) {
        std::vector<float> data(size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        
        for (auto& val : data) {
            val = dis(gen);
        }
        return data;
    }
    
    static bool verifyResults(const std::vector<float>& expected,
                             const std::vector<float>& actual,
                             float tolerance = 1e-4f) {
        if (expected.size() != actual.size()) {
            std::cout << "Size mismatch: expected " << expected.size()
                      << ", got " << actual.size() << std::endl;
            return false;
        }
        
        int error_count = 0;
        float max_error = 0.0f;
        
        for (size_t i = 0; i < expected.size(); i++) {
            float error = std::abs(expected[i] - actual[i]);
            max_error = std::max(max_error, error);
            
            if (error > tolerance) {
                error_count++;
                if (error_count <= 5) {  // 只打印前5个错误
                    std::cout << "Error at index " << i << ": expected "
                              << expected[i] << ", got " << actual[i]
                              << ", error = " << error << std::endl;
                }
            }
        }
        
        std::cout << "Max error: " << max_error << ", Error count: " << error_count
                  << "/" << expected.size() << std::endl;
        
        return error_count == 0;
    }
    
    static TestResult runFullBenchmark(const ConvParams& params) {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "Running Convolution Benchmark" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Batch size: " << params.batch_size << std::endl;
        std::cout << "  Input: " << params.input_channels << "x" << params.input_height
                  << "x" << params.input_width << std::endl;
        std::cout << "  Kernel: " << params.output_channels << "x" << params.input_channels
                  << "x" << params.kernel_height << "x" << params.kernel_width << std::endl;
        std::cout << "  Output: " << params.output_channels << "x" << params.output_height()
                  << "x" << params.output_width() << std::endl;
        std::cout << "  Memory requirement: " << params.memory_requirement_mb() << " MB" << std::endl;
        
        if (params.memory_requirement_mb() > 15 * 1024) {
            std::cout << "WARNING: Memory requirement exceeds 15GB!" << std::endl;
            TestResult result = {};
            return result;
        }
        
        // 生成测试数据
        std::cout << "\nGenerating test data..." << std::endl;
        auto input = generateRandomData(params.input_size());
        auto kernel = generateRandomData(params.kernel_size());
        
        std::vector<float> output_cpu(params.output_size());
        std::vector<float> output_gpu_basic(params.output_size());
        std::vector<float> output_gpu_optimized(params.output_size());
        std::vector<float> output_gpu_tiled(params.output_size());
        
        TestResult result = {};
        
        try {
            // CPU测试
            std::cout << "\nRunning CPU convolution..." << std::endl;
            CPUConvolution cpu_conv(params);
            result.cpu_time = cpu_conv.compute(input, kernel, output_cpu);
            std::cout << "CPU time: " << result.cpu_time << " ms" << std::endl;
            
            // GPU测试
            std::cout << "\nRunning GPU convolutions..." << std::endl;
            DCUConvolutionEngine gpu_conv(params);
            
            // 基础版本
            std::cout << "  Basic kernel..." << std::endl;
            result.gpu_basic_time = gpu_conv.runBasicConvolution(input, kernel, output_gpu_basic);
            std::cout << "  Basic time: " << result.gpu_basic_time << " ms" << std::endl;
            
            // 优化版本
            std::cout << "  Optimized kernel..." << std::endl;
            result.gpu_optimized_time = gpu_conv.runOptimizedConvolution(input, kernel, output_gpu_optimized);
            std::cout << "  Optimized time: " << result.gpu_optimized_time << " ms" << std::endl;
            
            // 分块版本
            std::cout << "  Tiled kernel..." << std::endl;
            result.gpu_tiled_time = gpu_conv.runTiledConvolution(input, kernel, output_gpu_tiled);
            std::cout << "  Tiled time: " << result.gpu_tiled_time << " ms" << std::endl;
            
            // 验证结果
            std::cout << "\nVerifying results..." << std::endl;
            std::cout << "Basic kernel verification: ";
            result.basic_correct = verifyResults(output_cpu, output_gpu_basic);
            std::cout << (result.basic_correct ? "PASSED" : "FAILED") << std::endl;
            
            std::cout << "Optimized kernel verification: ";
            result.optimized_correct = verifyResults(output_cpu, output_gpu_optimized);
            std::cout << (result.optimized_correct ? "PASSED" : "FAILED") << std::endl;
            
            std::cout << "Tiled kernel verification: ";
            result.tiled_correct = verifyResults(output_cpu, output_gpu_tiled);
            std::cout << (result.tiled_correct ? "PASSED" : "FAILED") << std::endl;
            
            // 计算加速比
            result.speedup_basic = result.cpu_time / result.gpu_basic_time;
            result.speedup_optimized = result.cpu_time / result.gpu_optimized_time;
            result.speedup_tiled = result.cpu_time / result.gpu_tiled_time;
            
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
        
        return result;
    }
    
    static void printSummary(const TestResult& result) {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "PERFORMANCE SUMMARY" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Execution Times:" << std::endl;
        std::cout << "  CPU:           " << std::setw(10) << result.cpu_time << " ms" << std::endl;
        std::cout << "  GPU Basic:     " << std::setw(10) << result.gpu_basic_time << " ms  "
                  << (result.basic_correct ? "✓" : "✗") << std::endl;
        std::cout << "  GPU Optimized: " << std::setw(10) << result.gpu_optimized_time << " ms  "
                  << (result.optimized_correct ? "✓" : "✗") << std::endl;
        std::cout << "  GPU Tiled:     " << std::setw(10) << result.gpu_tiled_time << " ms  "
                  << (result.tiled_correct ? "✓" : "✗") << std::endl;
        
        std::cout << "\nSpeedup vs CPU:" << std::endl;
        std::cout << "  Basic:         " << std::setw(10) << result.speedup_basic << "x" << std::endl;
        std::cout << "  Optimized:     " << std::setw(10) << result.speedup_optimized << "x" << std::endl;
        std::cout << "  Tiled:         " << std::setw(10) << result.speedup_tiled << "x" << std::endl;
        
        std::cout << "\nGPU Optimizations:" << std::endl;
        if (result.gpu_basic_time > 0) {
            std::cout << "  Optimized vs Basic: " << std::setw(6)
                      << result.gpu_basic_time / result.gpu_optimized_time << "x" << std::endl;
            std::cout << "  Tiled vs Basic:     " << std::setw(6)
                      << result.gpu_basic_time / result.gpu_tiled_time << "x" << std::endl;
        }
        return;
    }
    
    static void saveResults(const TestResult& result, const ConvParams& params, const std::string& filename) {
        std::ofstream file(filename, std::ios::app);
        if (file.is_open()) {
            file << params.batch_size << "," << params.input_channels << ","
                 << params.output_channels << "," << params.input_height << ","
                 << params.input_width << "," << params.kernel_height << ","
                 << params.kernel_width << "," << result.cpu_time << ","
                 << result.gpu_basic_time << "," << result.gpu_optimized_time << ","
                 << result.gpu_tiled_time << "," << result.speedup_basic << ","
                 << result.speedup_optimized << "," << result.speedup_tiled << std::endl;
        }
        return;
    }
};

int main() {
    std::cout << "DCU Convolution Performance Benchmark" << std::endl;
    std::cout << "Target Platform: 异构加速卡1 (16GB)" << std::endl;
    
    // 初始化DCU
    HIP_CHECK(hipInit(0));
    
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    std::cout << "Found " << deviceCount << " DCU device(s)" << std::endl;
    
    if (deviceCount == 0) {
        std::cout << "No DCU devices found!" << std::endl;
        return -1;
    }
    
    HIP_CHECK(hipSetDevice(0));
    
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    
    // 设置CPU线程数
    int cpu_threads = std::min(36, (int)std::thread::hardware_concurrency());  // 基于提供的36个空闲节点
    omp_set_num_threads(cpu_threads);
    std::cout << "Using " << cpu_threads << " CPU threads" << std::endl;
    
    // 创建结果文件
    std::string results_file = "results/convolution_results.csv";
    std::ofstream file(results_file);
    if (file.is_open()) {
        file << "batch_size,input_channels,output_channels,input_height,input_width,"
             << "kernel_height,kernel_width,cpu_time,gpu_basic_time,gpu_optimized_time,"
             << "gpu_tiled_time,speedup_basic,speedup_optimized,speedup_tiled" << std::endl;
        file.close();
    }
    
    // 测试用例：从小到大，充分利用16GB显存
    std::vector<ConvParams> test_cases = {
        // 小规模测试
        {1, 3, 16, 32, 32, 3, 3, 1, 1, 1, 1},
        {4, 16, 32, 64, 64, 3, 3, 1, 1, 1, 1},
        
        // 中等规模测试
        {8, 32, 64, 128, 128, 3, 3, 1, 1, 1, 1},
        {16, 64, 128, 256, 256, 3, 3, 1, 1, 1, 1},
        
        // 大规模测试 - 接近16GB限制
        {4, 128, 256, 512, 512, 3, 3, 1, 1, 1, 1},
        {2, 256, 512, 512, 512, 3, 3, 1, 1, 1, 1},
        
        // 不同kernel大小测试
        {8, 64, 128, 224, 224, 5, 5, 1, 1, 2, 2},
        {4, 128, 256, 224, 224, 7, 7, 2, 2, 3, 3},
        
        // ResNet风格测试
        {32, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1},
        {32, 128, 128, 28, 28, 3, 3, 1, 1, 1, 1},
        {32, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1},
        {32, 512, 512, 7, 7, 3, 3, 1, 1, 1, 1}
    };
    
    std::vector<ConvolutionBenchmark::TestResult> all_results;
    
    for (size_t i = 0; i < test_cases.size(); i++) {
        std::cout << "\n" << std::string(80, '*') << std::endl;
        std::cout << "TEST CASE " << (i + 1) << " / " << test_cases.size() << std::endl;
        std::cout << std::string(80, '*') << std::endl;
        
        auto result = ConvolutionBenchmark::runFullBenchmark(test_cases[i]);
        all_results.push_back(result);
        
        ConvolutionBenchmark::printSummary(result);
        ConvolutionBenchmark::saveResults(result, test_cases[i], results_file);
        
        // 简单的垃圾回收
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // 总体统计
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "OVERALL STATISTICS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    double avg_speedup_basic = 0, avg_speedup_optimized = 0, avg_speedup_tiled = 0;
    int valid_results = 0;
    
    for (const auto& result : all_results) {
        if (result.cpu_time > 0) {
            avg_speedup_basic += result.speedup_basic;
            avg_speedup_optimized += result.speedup_optimized;
            avg_speedup_tiled += result.speedup_tiled;
            valid_results++;
        }
    }
    
    if (valid_results > 0) {
        avg_speedup_basic /= valid_results;
        avg_speedup_optimized /= valid_results;
        avg_speedup_tiled /= valid_results;
        
        std::cout << "Average Speedups:" << std::endl;
        std::cout << "  Basic GPU:     " << std::fixed << std::setprecision(2)
                  << avg_speedup_basic << "x" << std::endl;
        std::cout << "  Optimized GPU: " << avg_speedup_optimized << "x" << std::endl;
        std::cout << "  Tiled GPU:     " << avg_speedup_tiled << "x" << std::endl;
    }
    
    std::cout << "\nResults saved to: " << results_file << std::endl;
    std::cout << "Benchmark completed successfully!" << std::endl;
    
    return 0;
}
