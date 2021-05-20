cmake_minimum_required(VERSION 3.19)
project(life_mpi)

include_directories(/usr/include/mpi)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_COMPILER /usr/bin/mpic++)

add_executable(life_mpi main.cpp)
