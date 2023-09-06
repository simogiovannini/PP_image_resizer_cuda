cmake_minimum_required(VERSION 3.26)
project(PP_image_resizer_cuda CUDA)

find_package( OpenCV REQUIRED )


set(CMAKE_CUDA_STANDARD 17)

add_executable(PP_image_resizer_cuda main.cu)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

set_target_properties(PP_image_resizer_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)