cmake_minimum_required(VERSION 3.10)
project(CS5234_Final_Project LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# NOTES for this to work (assuming nVidia CUDA target):
# it requires nvptx to be installed: sudo apt install gcc-offload-nvptx
# or if using gcc-10: sudo apt install gcc-10-offload-nvptx

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopt-info-all=vectorizations.txt") # print out optimized loops in txt file
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fcf-protection=none -fno-stack-protector -foffload=nvptx-none=\"-misa=sm_35\"" )

set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math" )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math" )

add_executable(kmeans kmeans.c)
#add_executable(nqueens_branch nqueens_branch.cpp)
#add_executable(nqueens_backtrack nqueens_backtrack.c)