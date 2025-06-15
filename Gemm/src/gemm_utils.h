#pragma once
#include "gemm.h"  
#include <vector>
#include <immintrin.h>  // for __m256, _mm256_* intrinsics

template<typename T>
void generate_sparse_matrix(Matrix<T>& matrix, double sparsity);
#pragma once

// 稀疏矩阵CSR格式乘法（SIMD优化）
void gemm_sparse_csr_simd(const CSRMatrix<float>& A, const Matrix<float>& B, Matrix<float>& C);

// 稀疏矩阵分块CSR格式模板结构体
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
