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

set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math" )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math" )

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopt-info-all=vectorizations.txt") # print out optimized loops in txt file
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-all=vectorizations.txt")

#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

add_executable(kmeans kmeans.c)
target_link_libraries(kmeans PRIVATE m)
add_executable(nqueens_branch nqueens_branch.cpp)
add_executable(nqueens_backtrack nqueens_backtrack.c)
add_executable(nqueens_somers nqueens_somers.c)