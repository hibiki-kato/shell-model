# CMakeのバージョンを設定
cmake_minimum_required(VERSION 3.14)
# プロジェクト名と使用する言語を設定
project(ShellModel CXX)
add_executable(energy energy_spectrum.cpp ShellModel.cpp)

#includeパス
target_include_directories(energy PUBLIC $/usr/include/python3.10)

#libディレクトリを指定
link_directories($/usr/lib/python3.10)

#ライブラリをリンク
target_link_libraries(energy PUBLIC python3.10)
#バージョン指定
set(CMAKE_CXX_FLAGS "-std=c++20 -O3 -mtune=native -march=native -I/usr/include/python3.10 -L/usr/lib/python3.10 -lpython3.10")