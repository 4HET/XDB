#!/bin/bash

# 确保构建目录存在
mkdir -p results

# 运行测试
echo "Starting DCU convolution benchmark..."
echo "Results will be saved to results/"

# 基础测试
echo "Running basic functionality test..."
timeout 1800 ./build/dcu_convolution > results/test_output.log 2>&1

if [ $? -eq 0 ]; then
    echo "Basic test completed successfully!"
else
    echo "Basic test failed or timed out!"
    exit 1
fi

# 性能分析（如果有hiprof）
if command -v hiprof &> /dev/null; then
    echo "Running performance analysis with hiprof..."
    hiprof --trace-hip --trace-kernel ./build/dcu_convolution > results/hiprof_output.log 2>&1
    echo "Performance analysis completed!"
fi

# 生成报告
python3 scripts/generate_report.py results/convolution_results.csv results/performance_report.html

echo "All tests completed!"
echo "Check results/ directory for output files."