cmake_minimum_required(VERSION 3.12)
project(feature_manager)

# Set the path to pybind11
set(CMAKE_PREFIX_PATH "/home/zhr/miniconda3/envs/ptdgl2/lib/python3.9/site-packages/pybind11/share/cmake/pybind11/")
find_package(pybind11 REQUIRED)

# Set the path to PyTorch (LibTorch)
set(Torch_DIR "/home/zhr/pytorch/pytorch1/pytorch/torch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Set the path to yaml-cpp (ensure it's installed or modify the path accordingly)
find_package(yaml-cpp REQUIRED)

# Specify the path to the cpp file
add_library(feature_manager MODULE lib/mapping_core.cpp)

# Link pybind11, PyTorch, and yaml-cpp to your project
target_link_libraries(feature_manager PRIVATE pybind11::module "${TORCH_LIBRARIES}" yaml-cpp)

# Set the C++ standard to C++11 (or higher)
set(CMAKE_CXX_STANDARD 11)

# Specify the output directory for the shared library (the current directory)
set_target_properties(feature_manager PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR} PREFIX "")

# Link Python libraries if necessary
target_link_libraries(feature_manager PRIVATE ${PYTHON_LIBRARIES})
