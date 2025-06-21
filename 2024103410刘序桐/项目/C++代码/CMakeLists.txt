cmake_minimum_required(VERSION 3.10)

project(hvns
    VERSION 1.0.0
    DESCRIPTION "HVNS 实现"
    LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

file(GLOB SRC_C "*.c")
file(GLOB SRC_CPP "*.cpp")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(SRC_C)
    add_library(lib_c_static STATIC ${SRC_C})
    add_library(lib_c_shared SHARED ${SRC_C})
    add_executable(hvns_c_static main.c)
    target_link_libraries(hvns_c_static PRIVATE lib_c_static)
    add_executable(hvns_c_shared main.c)
    target_link_libraries(hvns_c_shared PRIVATE lib_c_shared)
endif()

if(SRC_CPP)
    add_library(lib_cpp_static STATIC ${SRC_CPP})
    add_library(lib_cpp_shared SHARED ${SRC_CPP})
    add_executable(hvns_cpp_static main.cpp)
    target_link_libraries(hvns_cpp_static PRIVATE lib_cpp_static)
    add_executable(hvns_cpp_shared main.cpp)
    target_link_libraries(hvns_cpp_shared PRIVATE lib_cpp_shared)
endif()