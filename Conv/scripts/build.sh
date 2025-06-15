#!/bin/bash

# 设置环境变量
export HIP_PATH=/opt/dtk/hip
export HIP_PLATFORM=amd  # 注意：如果你使用 ROCm 4.x+，不应使用 hcc
export PATH=$HIP_PATH/bin:$PATH
export LD_LIBRARY_PATH=$HIP_PATH/lib:$LD_LIBRARY_PATH

# 编译选项
COMPILER=hipcc
CXX_FLAGS="-std=c++14 -O3 -fopenmp"
INCLUDE_DIRS="-I./include -I$HIP_PATH/include"
LIBRARY_DIRS="-L$HIP_PATH/lib"
LIBRARIES="-lamdhip64 -lrocblas"

# 编译
echo "Building DCU convolution benchmark..."
$COMPILER $CXX_FLAGS $INCLUDE_DIRS $LIBRARY_DIRS src/dcu_convolution.cpp $LIBRARIES -o build/dcu_convolution

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Executable: build/dcu_convolution"
else
    echo "Build failed!"
    exit 1
fi
