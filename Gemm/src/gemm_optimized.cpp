#include "gemm.h"

// 分块矩阵乘法
template<typename T>
void gemm_blocked(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, size_t block_size) {
    size_t m = A.rows;
    size_t n = B.cols;
    size_t k = A.cols;
    
    std::fill(C.data.begin(), C.data.end(), T(0));
    
    #pragma omp parallel for collapse(2)
    for (size_t ii = 0; ii < m; ii += block_size) {
        for (size_t jj = 0; jj < n; jj += block_size) {
            for (size_t kk = 0; kk < k; kk += block_size) {
                size_t i_end = std::min(ii + block_size, m);
                size_t j_end = std::min(jj + block_size, n);
                size_t k_end = std::min(kk + block_size, k);
                
                // 内层分块计算
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t l = kk; l < k_end; ++l) {
                        T a_il = A(i, l);
                        for (size_t j = jj; j < j_end; ++j) {
                            C(i, j) += a_il * B(l, j);
                        }
                    }
                }
            }
        }
    }
}

// SIMD优化的浮点矩阵乘法
void gemm_simd(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    size_t m = A.rows;
    size_t n = B.cols;
    size_t k = A.cols;
    
    std::fill(C.data.begin(), C.data.end(), 0.0f);
    
    const size_t simd_width = 8; // AVX 256位可处理8个float
    
    #pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        for (size_t l = 0; l < k; ++l) {
            __m256 a_il = _mm256_broadcast_ss(&A(i, l));
            
            size_t j = 0;
            // SIMD处理
            for (; j + simd_width <= n; j += simd_width) {
                __m256 b_vec = _mm256_loadu_ps(&B(l, j));
                __m256 c_vec = _mm256_loadu_ps(&C(i, j));
                __m256 result = _mm256_fmadd_ps(a_il, b_vec, c_vec);
                _mm256_storeu_ps(&C(i, j), result);
            }
            
            // 处理剩余元素
            for (; j < n; ++j) {
                C(i, j) += A(i, l) * B(l, j);
            }
        }
    }
}

// 循环展开优化
template<typename T>
void gemm_unrolled(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    size_t m = A.rows;
    size_t n = B.cols;
    size_t k = A.cols;
    
    std::fill(C.data.begin(), C.data.end(), T(0));
    
    #pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        for (size_t l = 0; l < k; ++l) {
            T a_il = A(i, l);
            size_t j = 0;
            
            // 4路展开
            for (; j + 4 <= n; j += 4) {
                C(i, j) += a_il * B(l, j);
                C(i, j + 1) += a_il * B(l, j + 1);
                C(i, j + 2) += a_il * B(l, j + 2);
                C(i, j + 3) += a_il * B(l, j + 3);
            }
            
            // 处理剩余
            for (; j < n; ++j) {
                C(i, j) += a_il * B(l, j);
            }
        }
    }
}

// 综合优化版本
template<typename T>
void gemm_optimized(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    // 根据矩阵大小选择最优策略
    size_t total_ops = A.rows * B.cols * A.cols;
    
    if (total_ops < 1000000) {
        // 小矩阵使用简单优化
        gemm_unrolled(A, B, C);
    } else {
        // 大矩阵使用分块优化
        size_t block_size = 64;
        if (total_ops > 100000000) block_size = 128;
        gemm_blocked(A, B, C, block_size);
    }
}

// 显式实例化
template void gemm_blocked<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&, size_t);
template void gemm_blocked<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&, size_t);
template void gemm_unrolled<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_unrolled<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void gemm_optimized<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_optimized<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);