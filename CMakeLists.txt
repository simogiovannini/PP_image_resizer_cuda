cmake_minimum_required(VERSION 3.26)
project(PP_image_resizer_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(PP_image_resizer_cuda main.cu)

set_target_properties(PP_image_resizer_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
