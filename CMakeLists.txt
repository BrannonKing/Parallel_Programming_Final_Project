execute_process(
	COMMAND bash "-c" "gcc -march=native -Q --help=target | grep -- '-march=' | cut -f3"
    OUTPUT_VARIABLE FOO
)
message(STATUS "GCC native architecture detected: ${FOO}")
cmake_minimum_required(VERSION 3.10)
project(CS5234_Final_Project LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "-O3 -mprefer-vector-width=256 -fvect-cost-model=cheap  -march=skylake-avx512 -Wall -Wextra -march=native")

find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#add_executable(kmeans_new kmeans2.cpp kmeans_util.cpp)
add_executable(kmeans_exec kmeans_new.cpp kmeans_util.cpp kmeans_par.cpp kmeans_parallel.cpp)
add_executable(nqueens_branch nqueens_branch.cpp)
add_executable(nqueens_backtrack nqueens_backtrack.c)
