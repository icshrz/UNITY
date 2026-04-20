#!/bin/bash

# 设置工程的根目录（根据你的实际情况修改）
PROJECT_DIR=$(pwd)

# 删除旧的构建文件
echo "Cleaning previous builds..."
rm -rf $PROJECT_DIR/build

# 创建新的构建目录
echo "Creating a new build directory..."
mkdir $PROJECT_DIR/build
cd $PROJECT_DIR/build

# 运行 cmake 生成构建文件
echo "Running cmake..."
cmake ..

# 编译项目
echo "Running make to build the shared library..."
make

# 如果需要安装到 Python 路径，可以运行
# make install

echo "Build complete. The .so file should now be located in the build directory."

