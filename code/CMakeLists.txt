# CMakeのバージョンを設定
cmake_minimum_required(VERSION 3.14)
# プロジェクト名と使用する言語を設定
project(ShellModel CXX)
add_executable(laminar jacobian_observation.cpp LongLaminar.cpp ShellModel.cpp)

#includeパス
target_include_directories(laminar PUBLIC $/usr/include/python3.10)

#libディレクトリを指定
link_directories($/usr/lib/python3.10)

#ライブラリを作成
add_library(cnpy SHARED "cnpy/cnpy.cpp")

#ライブラリをリンク
target_link_libraries(laminar PUBLIC python3.10 cnpy z)

#キャッシュなし
set(CMAKE_NO_CACHE TRUE)

#バージョン指定
set(CMAKE_CXX_FLAGS "-std=c++20 -O3 -fopenmp -mtune=native -march=native -I/usr/include/python3.10 -L/usr/lib/python3.10 -lpython3.10")
