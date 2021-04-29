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
// high_resolution_clock example
#include <iostream>
#include <ctime>
#include <ratio>
#include <immintrin.h>
#include <chrono>
#include <iomanip>
#include <queue>

struct  Membership {
    int centroid;
    float distance;
}__attribute__((packed));

struct custom_cmpr {
    bool operator()(float p1,float p2) {
        // return "true" if "p1" is ordered before "p2", for example:
        if (p1 > p2)
            return true;
        return false;
    }
};

struct FeatureDefinition {
    int32_t npoints, nfeatures;
    float **features; // [npoints][nfeatures]
};

typedef struct Membership mem_point;
typedef std::vector<float> v_float;

#define AOCL_ALIGNMENT 64
#endif //CS5234_FINAL_PROJECT_KMEANS_NEW_H
