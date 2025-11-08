cmake_minimum_required(VERSION 3.20)
project(llamagents C CXX)

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)

add_subdirectory(external/llama.cpp)
add_subdirectory(shims)