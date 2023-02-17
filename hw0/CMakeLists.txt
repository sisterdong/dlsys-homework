cmake_minimum_required(VERSION 3.4)
project(hw0)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(simple_ml_ext src/simple_ml_ext.cpp)
set_target_properties(simple_ml_ext PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src")
