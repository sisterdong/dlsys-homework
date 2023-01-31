cmake_minimum_required(VERSION 2.8.12)
project(hw0)
set(CMAKE_CXX_FLAGS "-O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup $$(python3 -m pybind11 --includes)")

add_subdirectory(src)
pybind11_add_module(hw0 src/simple_ml_ext.cpp)
