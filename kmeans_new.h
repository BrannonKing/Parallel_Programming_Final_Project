//
// Created by ayush on 4/29/21.
//
#pragma once
#ifndef CS5234_FINAL_PROJECT_KMEANS_NEW_H
#define CS5234_FINAL_PROJECT_KMEANS_NEW_H

#include <vector>
#include <iostream>

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>

#include <iostream>
#include <ctime>
#include <ratio>
#include <immintrin.h>
#include <chrono>
#include <iomanip>
#include <queue>




struct FeatureDefinition {
    int32_t npoints, nfeatures;
    float **features; // [npoints][nfeatures]
};


typedef std::vector<float> v_float;

extern    long N;
extern    long M;
using DataFrame = std::vector<std::vector<float>>;

#define AOCL_ALIGNMENT 64
#endif //CS5234_FINAL_PROJECT_KMEANS_NEW_H
