//
// Created by ayush on 4/29/21.
//

#ifndef CS5234_FINAL_PROJECT_KMEANS_PARALLEL_H
#define CS5234_FINAL_PROJECT_KMEANS_PARALLEL_H

#pragma once
#include "kmeans_new.h"
#include "kmeans_util.h"
namespace kmeansparallel {
    int32_t *calculateMeans_omp(int k, vector<v_float> data_array, long iteration, vector<v_float> &means_array);
}
#endif //CS5234_FINAL_PROJECT_KMEANS_PARALLEL_H
