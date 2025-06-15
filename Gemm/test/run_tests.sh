#!/bin/bash

echo "=== GEMM 性能测试脚本 ==="

# 检查编译环境
check_environment() {
    echo "检查编译环境..."
    
    if ! command -v g++ &> /dev/null; then
        echo "错误: 未找到 g++ 编译器"
        exit 1
    fi
    
    # 检查OpenMP支持
    if ! echo '#include <omp.h>' | g++ -fopenmp -x c++ - -o /tmp/test_omp 2>/dev/null; then
        echo "警告: OpenMP支持可能不完整"
    else
        rm -f /tmp/test_omp
        echo "OpenMP支持正常"
    fi
    
    # 检查AVX2支持
    if grep -q avx2 /proc/cpuinfo; then
        echo "检测到AVX2支持"
    else
        echo "警告: 未检测到AVX2支持，SIMD优化可能无效"
    fi
    
    echo "环境检查完成"
}

# 编译程序
compile_program() {
    echo "编译程序..."
    make clean
    make -j$(nproc)
    
    if [ $? -ne 0 ]; then
        echo "编译失败"
        exit 1
    fi
    echo "编译成功"
}

# 运行基础测试
run_basic_tests() {
    echo "运行基础功能测试..."
    export OMP_NUM_THREADS=1
    timeout 300 ./gemm_test > test_results_single.txt 2>&1
    
    if [ $? -eq 124 ]; then
        echo "基础测试超时"
        return 1
    fi
    
    echo "基础测试完成，结果保存到 test_results_single.txt"
}

# 运行多线程测试
run_parallel_tests() {
    echo "运行多线程性能测试..."
    
    for threads in 1 4 8 16 32; do
        echo "测试 $threads 线程..."
        export OMP_NUM_THREADS=$threads
        
        echo "=== 使用 $threads 线程 ===" >> test_results_parallel.txt
        timeout 600 ./gemm_test >> test_results_parallel.txt 2>&1
        
        if [ $? -eq 124 ]; then
            echo "  $threads 线程测试超时"
            continue
        fi
        
        echo "  $threads 线程测试完成"
    done
    
    echo "多线程测试完成，结果保存到 test_results_parallel.txt"
}

# 运行内存使用测试
run_memory_tests() {
    echo "运行内存使用测试..."
    
    export OMP_NUM_THREADS=32
    
    # 使用valgrind检查内存泄漏（如果可用）
    if command -v valgrind &> /dev/null; then
        echo "使用valgrind检查内存..."
        valgrind --tool=memcheck --leak-check=full --track-origins=yes \
                 ./gemm_test > memory_test.txt 2>&1
        echo "内存检查结果保存到 memory_test.txt"
    else
        echo "valgrind未安装，跳过内存检查"
    fi
    
    # 使用time命令监控资源使用
    echo "监控资源使用..."
    /usr/bin/time -v ./gemm_test > resource_usage.txt 2>&1
    echo "资源使用情况保存到 resource_usage.txt"
}

# 生成性能报告
generate_report() {
    echo "生成性能报告..."
    
    cat > performance_report.md << EOF
# GEMM 性能测试报告

## 测试环境
- 处理器: $(cat /proc/cpuinfo | grep "model name" | head -1 | cut -d: -f2)
- 核心数: $(nproc)
- 内存: $(free -h | grep Mem | awk '{print $2}')
- 编译器: $(g++ --version | head -1)
- 测试时间: $(date)

## 测试结果

### 单线程性能
EOF
    
    if [ -f test_results_single.txt ]; then
        echo '```' >> performance_report.md
        cat test_results_single.txt >> performance_report.md
        echo '```' >> performance_report.md
    fi
    
    cat >> performance_report.md << EOF

### 多线程性能
EOF
    
    if [ -f test_results_parallel.txt ]; then
        echo '```' >> performance_report.md
        cat test_results_parallel.txt >> performance_report.md
        echo '```' >> performance_report.md
    fi
    
    cat >> performance_report.md << EOF

### 资源使用情况
EOF
    
    if [ -f resource_usage.txt ]; then
        echo '```' >> performance_report.md
        tail -20 resource_usage.txt >> performance_report.md
        echo '```' >> performance_report.md
    fi
    
    echo "性能报告生成完成: performance_report.md"
}

# 主测试流程
main() {
    cd "$(dirname "$0")/.."
    
    check_environment
    compile_program
    
    echo "开始性能测试..."
    run_basic_tests
    run_parallel_tests
    run_memory_tests
    generate_report
    
    echo "所有测试完成！"
    echo "查看以下文件获取详细结果:"
    echo "  - test_results_single.txt: 单线程测试结果"
    echo "  - test_results_parallel.txt: 多线程测试结果"
    echo "  - resource_usage.txt: 资源使用情况"
    echo "  - performance_report.md: 综合性能报告"
}

# 运行主程序
main "$@"