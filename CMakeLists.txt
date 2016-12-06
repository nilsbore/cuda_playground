cmake_minimum_required(VERSION 2.6.0)
project(cuda_playground)

add_definitions(-std=c++11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenCV REQUIRED)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

include_directories(include)

file(GLOB cuda src/cuda/*.cu)

# this BS is needed to get VTK stuff to compile together with CUDA
get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
set(vtk_flags)
foreach(it ${dir_defs})
    if(it MATCHES "vtk*")
    list(APPEND vtk_flags ${it})
    endif()
endforeach()

foreach(d ${vtk_flags})
    remove_definitions(-D${d})
endforeach()

#set(CUDA_ARCH_BIN "30 35 50 52 61" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_BIN "20 30 35 50 52 61" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")

include(CudaComputeTargetFlags.cmake)
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")

CUDA_COMPILE(cuda_objs ${cuda})

set(CMAKE_CXX_FLAGS ${ADDITIONAL_CMAKE_CXX_FLAGS} "-O3 -msse2 -msse3 -Wall -std=c++11 -DSHADER_DIR=${efusion_SHADER_DIR}")

add_library(kernel SHARED src/cuda/kernel.cuh ${cuda} ${cuda_objs})
set_target_properties(kernel PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(kernel ${CUDA_LIBRARIES})

add_library(project_kernel SHARED src/cuda/project_kernel.cuh ${cuda} ${cuda_objs})
set_target_properties(project_kernel PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(project_kernel ${CUDA_LIBRARIES})

add_executable(hello src/hello.cpp)
target_link_libraries(hello kernel ${CUDA_LIBRARIES})

add_executable(cpu_project src/cpu_project.cpp)
target_link_libraries(cpu_project ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_executable(gpu_project src/gpu_project.cpp)
target_link_libraries(gpu_project project_kernel ${OpenCV_LIBS} ${PCL_LIBRARIES})
