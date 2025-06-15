#ifndef GEMM_H
#define GEMM_H

#include <vector>
#include <chrono>
#include <memory>
#include <immintrin.h>
#include <omp.h>
#include <random>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <algorithm>

// 矩阵类定义
template<typename T>
class Matrix {
public:
    std::vector<T> data;
    size_t rows, cols;
    
    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c) {}
    
    T& operator()(size_t i, size_t j) {
        return data[i * cols + j];
    }
    
    const T& operator()(size_t i, size_t j) const {
        return data[i * cols + j];
    }
    
    T* ptr() { return data.data(); }
    const T* ptr() const { return data.data(); }
};

// 稀疏矩阵CSR格式
template<typename T>
struct CSRMatrix {
    std::vector<T> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    size_t rows, cols;
    
    CSRMatrix(size_t r, size_t c) : rows(r), cols(c) {
        row_ptr.resize(r + 1, 0);
    }
};

// 性能计时器
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        return duration.count() / 1000.0; // 返回毫秒
    }
};

// 基础矩阵乘法
template<typename T>
void gemm_basic(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);

// 优化矩阵乘法
template<typename T>
void gemm_optimized(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);

// 分块矩阵乘法
template<typename T>
void gemm_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, 
                  size_t block_size = 64);

// SIMD优化矩阵乘法
void gemm_simd(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);

// 稀疏矩阵乘法
template<typename T>
void gemm_sparse_csr(const CSRMatrix<T>& A, const Matrix<T>& B, Matrix<T>& C);

// 工具函数
template<typename T>
void generate_random_matrix(Matrix<T>& matrix, T min_val = 0, T max_val = 1);

template<typename T>
CSRMatrix<T> dense_to_csr(const Matrix<T>& dense, T threshold = 1e-6);

template<typename T>
bool verify_result(const Matrix<T>& A, const Matrix<T>& B, T tolerance = 1e-5);

void print_performance_stats(const std::string& method, double time_ms, 
                           size_t m, size_t n, size_t k);

#endif // GEMM_H