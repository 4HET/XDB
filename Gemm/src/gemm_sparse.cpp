#include "gemm.h"

// 稀疏矩阵CSR格式乘法
template<typename T>
void gemm_sparse_csr(const CSRMatrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    size_t m = A.rows;
    size_t n = B.cols;
    
    std::fill(C.data.begin(), C.data.end(), T(0));
    
    #pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; ++idx) {
            int col = A.col_indices[idx];
            T val = A.values[idx];
            
            for (size_t j = 0; j < n; ++j) {
                C(i, j) += val * B(col, j);
            }
        }
    }
}

// 稀疏矩阵CSR格式乘法（SIMD优化）
void gemm_sparse_csr_simd(const CSRMatrix<float>& A, const Matrix<float>& B, Matrix<float>& C) {
    size_t m = A.rows;
    size_t n = B.cols;
    
    std::fill(C.data.begin(), C.data.end(), 0.0f);
    
    const size_t simd_width = 8;
    
    #pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; ++idx) {
            int col = A.col_indices[idx];
            __m256 val_vec = _mm256_broadcast_ss(&A.values[idx]);
            
            size_t j = 0;
            for (; j + simd_width <= n; j += simd_width) {
                __m256 b_vec = _mm256_loadu_ps(&B(col, j));
                __m256 c_vec = _mm256_loadu_ps(&C(i, j));
                __m256 result = _mm256_fmadd_ps(val_vec, b_vec, c_vec);
                _mm256_storeu_ps(&C(i, j), result);
            }
            
            for (; j < n; ++j) {
                C(i, j) += A.values[idx] * B(col, j);
            }
        }
    }
}

// 稀疏矩阵分块CSR格式
template<typename T>
struct BCSR_Matrix {
    std::vector<T> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    size_t rows, cols;
    size_t block_rows, block_cols;
    size_t block_size;
    
    BCSR_Matrix(size_t r, size_t c, size_t bs) 
        : rows(r), cols(c), block_size(bs) {
        block_rows = (r + bs - 1) / bs;
        block_cols = (c + bs - 1) / bs;
        row_ptr.resize(block_rows + 1, 0);
    }
};

// 稠密矩阵转CSR格式
template<typename T>
CSRMatrix<T> dense_to_csr(const Matrix<T>& dense, T threshold) {
    CSRMatrix<T> csr(dense.rows, dense.cols);
    
    for (size_t i = 0; i < dense.rows; ++i) {
        for (size_t j = 0; j < dense.cols; ++j) {
            if (std::abs(dense(i, j)) > threshold) {
                csr.values.push_back(dense(i, j));
                csr.col_indices.push_back(j);
            }
        }
        csr.row_ptr[i + 1] = csr.values.size();
    }
    
    return csr;
}

// 生成稀疏矩阵
template<typename T>
void generate_sparse_matrix(Matrix<T>& matrix, double sparsity) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::normal_distribution<T> val_dis(0.0, 1.0);
    
    for (size_t i = 0; i < matrix.rows; ++i) {
        for (size_t j = 0; j < matrix.cols; ++j) {
            if (dis(gen) > sparsity) {
                matrix(i, j) = val_dis(gen);
            } else {
                matrix(i, j) = T(0);
            }
        }
    }
}

// 显式实例化
template void gemm_sparse_csr<float>(const CSRMatrix<float>&, const Matrix<float>&, Matrix<float>&);
template void gemm_sparse_csr<double>(const CSRMatrix<double>&, const Matrix<double>&, Matrix<double>&);
template CSRMatrix<float> dense_to_csr<float>(const Matrix<float>&, float);
template CSRMatrix<double> dense_to_csr<double>(const Matrix<double>&, double);
template void generate_sparse_matrix<float>(Matrix<float>&, double);
template void generate_sparse_matrix<double>(Matrix<double>&, double);