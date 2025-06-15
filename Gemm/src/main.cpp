#include "gemm.h"
#include "gemm_utils.h"
#include <functional>

void test_basic_gemm() {
    std::cout << "\n=== 任务1: 基础矩阵乘法测试 ===" << std::endl;
    
    std::vector<size_t> sizes = {128, 256, 512, 1024};
    
    for (size_t size : sizes) {
        Matrix<float> A(size, size), B(size, size), C(size, size);
        generate_random_matrix(A, -1.0f, 1.0f);
        generate_random_matrix(B, -1.0f, 1.0f);
        
        Timer timer;
        timer.start();
        gemm_basic(A, B, C);
        double time_ms = timer.stop();
        
        print_performance_stats("基础实现", time_ms, size, size, size);
    }
}

void test_optimized_gemm() {
    std::cout << "\n=== 任务2: 优化矩阵乘法测试 ===" << std::endl;
    
    std::vector<size_t> sizes = {128, 256, 512, 1024, 2048};
    
    for (size_t size : sizes) {
        Matrix<float> A(size, size), B(size, size);
        generate_random_matrix(A, -1.0f, 1.0f);
        generate_random_matrix(B, -1.0f, 1.0f);
        
        // 测试不同优化方法
        std::vector<std::pair<std::string, std::function<void()>>> methods = {
            {"分块优化", [&]() {
                Matrix<float> C(size, size);
                Timer timer;
                timer.start();
                gemm_blocked(A, B, C, 64);
                double time_ms = timer.stop();
                print_performance_stats("分块优化(64)", time_ms, size, size, size);
            }},
            {"SIMD优化", [&]() {
                Matrix<float> C(size, size);
                Timer timer;
                timer.start();
                gemm_simd(A, B, C);
                double time_ms = timer.stop();
                print_performance_stats("SIMD优化", time_ms, size, size, size);
            }},
            {"综合优化", [&]() {
                Matrix<float> C(size, size);
                Timer timer;
                timer.start();
                gemm_optimized(A, B, C);
                double time_ms = timer.stop();
                print_performance_stats("综合优化", time_ms, size, size, size);
            }}
        };
        
        std::cout << "\n矩阵规模: " << size << "x" << size << std::endl;
        for (auto& method : methods) {
            method.second();
        }
    }
}

void test_sparse_gemm() {
    std::cout << "\n=== 任务3: 稀疏矩阵乘法测试 ===" << std::endl;
    
    std::vector<size_t> sizes = {512, 1024, 2048};
    std::vector<double> sparsities = {0.7, 0.9, 0.95, 0.99};
    
    for (size_t size : sizes) {
        for (double sparsity : sparsities) {
            Matrix<float> A_dense(size, size), B(size, size), C_dense(size, size), C_sparse(size, size);
            
            generate_sparse_matrix(A_dense, sparsity);
            generate_random_matrix(B, -1.0f, 1.0f);
            
            // 转换为CSR格式
            CSRMatrix<float> A_csr = dense_to_csr(A_dense, 1e-6f);
            
            Timer timer;
            
            // 稠密矩阵乘法
            timer.start();
            gemm_basic(A_dense, B, C_dense);
            double dense_time = timer.stop();
            
            // 稀疏矩阵乘法
            timer.start();
            gemm_sparse_csr(A_csr, B, C_sparse);
            double sparse_time = timer.stop();
            
            // SIMD稀疏矩阵乘法
            Matrix<float> C_sparse_simd(size, size);
            timer.start();
            gemm_sparse_csr_simd(A_csr, B, C_sparse_simd);
            double sparse_simd_time = timer.stop();
            
            double actual_sparsity = (double)A_csr.values.size() / (size * size);
            double speedup = dense_time / sparse_time;
            double simd_speedup = dense_time / sparse_simd_time;
            
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "规模: " << size << "x" << size 
                      << " | 稀疏率: " << actual_sparsity * 100 << "%"
                      << " | 稠密: " << dense_time << "ms"
                      << " | 稀疏: " << sparse_time << "ms (加速比: " << speedup << "x)"
                      << " | 稀疏SIMD: " << sparse_simd_time << "ms (加速比: " << simd_speedup << "x)"
                      << std::endl;
            
            // 验证结果正确性
            if (!verify_result(C_dense, C_sparse, 1e-3f)) {
                std::cout << "警告: 稀疏矩阵结果验证失败!" << std::endl;
            }
        }
        std::cout << std::endl;
    }
}

void benchmark_all() {
    std::cout << "\n=== 综合性能基准测试 ===" << std::endl;
    
    // 设置OpenMP线程数
    int num_threads = omp_get_max_threads();
    std::cout << "使用OpenMP线程数: " << num_threads << std::endl;
    
    size_t test_size = 1024;
    Matrix<float> A(test_size, test_size), B(test_size, test_size);
    generate_random_matrix(A, -1.0f, 1.0f);
    generate_random_matrix(B, -1.0f, 1.0f);
    
    std::vector<std::pair<std::string, std::function<double()>>> benchmarks = {
        {"基础实现", [&]() {
            Matrix<float> C(test_size, test_size);
            Timer timer;
            timer.start();
            gemm_basic(A, B, C);
            return timer.stop();
        }},
        {"分块32", [&]() {
            Matrix<float> C(test_size, test_size);
            Timer timer;
            timer.start();
            gemm_blocked(A, B, C, 32);
            return timer.stop();
        }},
        {"分块64", [&]() {
            Matrix<float> C(test_size, test_size);
            Timer timer;
            timer.start();
            gemm_blocked(A, B, C, 64);
            return timer.stop();
        }},
        {"分块128", [&]() {
            Matrix<float> C(test_size, test_size);
            Timer timer;
            timer.start();
            gemm_blocked(A, B, C, 128);
            return timer.stop();
        }},
        {"SIMD优化", [&]() {
            Matrix<float> C(test_size, test_size);
            Timer timer;
            timer.start();
            gemm_simd(A, B, C);
            return timer.stop();
        }},
        {"综合优化", [&]() {
            Matrix<float> C(test_size, test_size);
            Timer timer;
            timer.start();
            gemm_optimized(A, B, C);
            return timer.stop();
        }}
    };
    
    // 多次运行取平均值
    const int runs = 3;
    for (auto& benchmark : benchmarks) {
        double total_time = 0;
        for (int i = 0; i < runs; ++i) {
            total_time += benchmark.second();
        }
        double avg_time = total_time / runs;
        print_performance_stats(benchmark.first, avg_time, test_size, test_size, test_size);
    }
}

int main() {
    std::cout << "=== GEMM 通用矩阵乘法优化测试程序 ===" << std::endl;
    std::cout << "编译选项: OpenMP + AVX2 SIMD" << std::endl;
    
    test_basic_gemm();
    test_optimized_gemm();
    test_sparse_gemm();
    benchmark_all();
    
    return 0;
}