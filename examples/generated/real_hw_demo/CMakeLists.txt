cmake_minimum_required(VERSION 3.22)
project(quad_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QNN SDK path
set(QNN_SDK_PATH "")

# Include directories
include_directories(
    ${QNN_SDK_PATH}/include
    ${QNN_SDK_PATH}/include/QNN
)

# Link directories
link_directories(
    ${QNN_SDK_PATH}/lib/aarch64-windows
)

# Source files
add_executable(inference
    inference.cpp
)

# Link QNN libraries
target_link_libraries(inference
    QnnHtp
    QnnSystem
)

# Install
install(TARGETS inference DESTINATION bin)
