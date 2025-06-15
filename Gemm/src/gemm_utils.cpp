#include "gemm.h"

template<typename T>
void generate_random_matrix(Matrix<T>& matrix, T min_val, T max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min_val, max_val);
    
    for (auto& element : matrix.data) {
        element = dis(gen);
    }
}

template<typename T>
bool verify_result(const Matrix<T>& A, const Matrix<T>& B, T tolerance) {
    if (A.rows != B.rows || A.cols != B.cols) {
        return false;
    }
    
    for (size_t i = 0; i < A.data.size(); ++i) {
        if (std::abs(A.data[i] - B.data[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

void print_performance_stats(const std::string& method, double time_ms, 
                           size_t m, size_t n, size_t k) {
    double gflops = (2.0 * m * n * k) / (time_ms * 1e6);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "方法: " << std::setw(20) << method 
              << " | 时间: " << std::setw(8) << time_ms << " ms"
              << " | 性能: " << std::setw(8) << gflops << " GFLOPS"
              << " | 矩阵规模: " << m << "x" << k << " * " << k << "x" << n << std::endl;
}

// 显式实例化
template void generate_random_matrix<float>(Matrix<float>&, float, float);
template void generate_random_matrix<double>(Matrix<double>&, double, double);
template bool verify_result<float>(const Matrix<float>&, const Matrix<float>&, float);
template bool verify_result<double>(const Matrix<double>&, const Matrix<double>&, double);