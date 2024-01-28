# CMakeのバージョンを設定
cmake_minimum_required(VERSION 3.11)
project(ShellModel)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(EXECUTABLE exe)
add_executable(${EXECUTABLE} 
    error_growth_mpi.cpp
    shared/myFunc.cpp
    shared/Flow.cpp
    # shared/Map.cpp
)

# OSでソースが異なるので分岐
if($ENV{OS_NAME} MATCHES "Darwin")
    message("Build in Mac")
    find_package (Python 3.10 REQUIRED COMPONENTS Interpreter Development NumPy)
    message(STATUS "Python_INCLUDE_DIRS:${Python_INCLUDE_DIRS}" )
    message(STATUS "Numpy include dirs:${Python_NumPy_INCLUDE_DIRS}" )
    link_directories(/opt/homebrew/lib)
    include_directories(/opt/homebrew/include ${Python_INCLUDE_DIRS})
elseif($ENV{OS_NAME} MATCHES "Wisteria")
    message("Build in Wisteria")
    link_directories(/usr/lib, /work/gt00/t00006/shell-model)
    include_directories(/usr/include, /work/gt00/t00006/shell-model)
else()
    message("Build in Linux")
    find_package (Python 3 REQUIRED COMPONENTS Interpreter Development NumPy)
    find_package(MPI REQUIRED)
    target_link_libraries(${EXECUTABLE} PUBLIC MPI::MPI_CXX)
    link_directories(/usr/lib)
    include_directories(/usr/include, ${MPI_iNCLUDE_PATH})
    target_link_libraries(${EXECUTABLE} PUBLIC z cnpy)
endif()

# Wisteriaでない場合はPythonをリンク
if(NOT $ENV{OS_NAME} STREQUAL "Wisteria")
    add_library(cnpy "shared/cnpy/cnpy.cpp")
    target_link_libraries(${EXECUTABLE} PUBLIC z cnpy Python::Python Python::NumPy)
endif()
#キャッシュなし
set(CMAKE_NO_CACHE TRUE)
set(CMAKE_CXX_FLAGS "-O3 -fopenmp -mtune=native -march=native -std=c++23")
if($ENV{OS_NAME} STREQUAL "Wisteria")
    set(CMAKE_CXX_COMPILER mpiFCCpx)
    set(CMAKE_CXX_FLAGS "-std=gnu++11 -Nfjomplib  -Ksimd=auto -Kopenmp -Kfast -O3 -w -Kocl -Koptmsg=2 -Krestp=all")
endif()

# message(STATUS "CMAKE_VERSION:${CMAKE_VERSION}" )
# message(STATUS "include_directories(PYTHON_INCLUDE_DIRS):${PYTHON_INCLUDE_DIRS}" )
# message(STATUS "include_directories(Python_INCLUDE_DIRS):${Python_INCLUDE_DIRS}" )
# message(STATUS "include_directories(PYTHON_LIB    RARIES):${PYTHON_LIBRARIES}" )
# message(STATUS "include_directories(Python_LIBRARIES):${Python_LIBRARIES}" )
# message(STATUS "include_directories(Python_LIBRARY_DIRS):${Python_LIBRARY_DIRS}" )