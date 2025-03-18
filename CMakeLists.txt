cmake_minimum_required(VERSION 3.10)
project(MNIST_Project)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set OpenMP paths for macOS (assuming OpenMP is installed via Homebrew)
if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    include_directories("/opt/homebrew/opt/libomp/include")
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Enable optimization flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Set Eigen include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/eigen/3.4.0_1/include/eigen3)

# Add executable
add_executable(read_dataset_images src/read_dataset_images.cpp src/mnist_loader.cpp)
add_executable(read_dataset_labels src/read_dataset_label.cpp src/mnist_loader.cpp)
add_executable(mnist src/mnist.cpp src/mnist_loader.cpp src/Neural_Network.cpp)
add_executable(test src/test.cpp)

# Link OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(mnist OpenMP::OpenMP_CXX)
    target_link_libraries(read_dataset_images OpenMP::OpenMP_CXX)
    target_link_libraries(read_dataset_labels OpenMP::OpenMP_CXX)
    target_link_libraries(test OpenMP::OpenMP_CXX)
endif()