#!/bin/bash

# 完整的自动化测试流程
set -e  # 遇到错误立即退出

echo "========================================="
echo "DCU 卷积计算自动化测试"
echo "目标平台: 异构加速卡1 (16GB显存)"
echo "========================================="

# 1. 环境检查
echo "1. 检查环境..."
if ! command -v hipcc &> /dev/null; then
    echo "错误: hipcc 编译器未找到"
    exit 1
fi

echo "✓ HIP 环境检查通过"

# 2. 编译项目
echo "2. 编译项目..."
chmod +x scripts/build.sh
./scripts/build.sh

if [ ! -f "build/dcu_convolution" ]; then
    echo "错误: 编译失败"
    exit 1
fi
echo "✓ 编译成功"

# 3. 运行测试
echo "3. 运行性能测试..."
chmod +x scripts/run_tests.sh
./scripts/run_tests.sh

# 4. 生成报告
echo "4. 生成性能报告..."
if [ -f "results/convolution_results.csv" ]; then
    python3 scripts/generate_report.py results/convolution_results.csv results/performance_report.html
    echo "✓ 性能报告已生成: results/performance_report.html"
else
    echo "警告: 未找到测试结果文件"
fi

# 5. 总结
echo "========================================="
echo "测试完成!"
echo "结果文件位置:"
echo "  - 详细日志: results/test_output.log"
echo "  - CSV数据: results/convolution_results.csv"
echo "  - HTML报告: results/performance_report.html"
echo "========================================="