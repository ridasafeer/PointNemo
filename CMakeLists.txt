cmake_minimum_required(VERSION 3.16)
project(PointNemoRT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# sources
file(GLOB PN_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# target
add_executable(pointnemo_rt ${PN_SOURCES})

# headers
target_include_directories(pointnemo_rt PUBLIC ${CMAKE_SOURCE_DIR}/include)
