project(cudp)
cmake_minimum_required(VERSION 2.8.11)

set(${PROJECT_NAME}_VERSION_MAJOR 0)
set(${PROJECT_NAME}_VERSION_MINOR 1)

find_package(CUDA REQUIRED)
find_package(Qt5Widgets REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# CUDA SESSION
list(APPEND CUDA_NVCC_FLAGS -std=c++11)
cuda_add_library(cudadevice cudadevice.cu cudadevice.h STATIC)
list(APPEND CUDA_LIBRARIES cudadevice)

# QT SESSION
list(APPEND QT_LIBRARIES Qt5::Widgets)
set(CMAKE_AUTOMOC ON)
add_definitions(-std=c++11)
aux_source_directory(. SRC_LIST)
qt5_wrap_ui(SRC_LIST deviceinfowidget.ui mainwindow.ui)
add_executable(${PROJECT_NAME} ${SRC_LIST})

# LINKING SESSION
target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES} ${CUDA_LIBRARIES})
