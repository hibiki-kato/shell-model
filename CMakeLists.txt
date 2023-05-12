# CMakeのバージョンを設定
cmake_minimum_required(VERSION 3.14)
# プロジェクト名と使用する言語を設定
project(trajectory_observation CXX)
add_executable(trajectory_observation.out Runge_Kutta.cpp trajectory_observation.cpp)

#バージョン指定
set(CMAKE_CXX_FLAGS "-std=c++20 -O3 -mtune=native -march=native")