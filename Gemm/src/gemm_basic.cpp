#include "gemm.h"

template<typename T>
void gemm_basic(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    size_t m = A.rows;
    size_t n = B.cols;
    size_t k = A.cols;
    
    // 初始化结果矩阵
    std::fill(C.data.begin(), C.data.end(), T(0));
    
    // 基础三重循环
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t l = 0; l < k; ++l) {
                C(i, j) += A(i, l) * B(l, j);
            }
        }
    }
}

// 循环重排序优化（ikj顺序）
template<typename T>
void gemm_ikj(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    size_t m = A.rows;
    size_t n = B.cols;
    size_t k = A.cols;
    
    std::fill(C.data.begin(), C.data.end(), T(0));
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t l = 0; l < k; ++l) {
            T a_il = A(i, l);
            for (size_t j = 0; j < n; ++j) {
                C(i, j) += a_il * B(l, j);
            }
        }
    }
}

// 显式实例化
template void gemm_basic<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_basic<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);
template void gemm_ikj<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_ikj<double>(const Matrix<double>&, const Matrix<double>&, Matrix<double>&);