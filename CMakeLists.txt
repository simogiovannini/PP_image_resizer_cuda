cmake_minimum_required(VERSION 3.26)

set(CMAKE_CUDA_STANDARD 17)
project(PP_image_resizer_cuda C CXX CUDA)

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(PP_image_resizer_cuda main.cu)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

set_target_properties(PP_image_resizer_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)