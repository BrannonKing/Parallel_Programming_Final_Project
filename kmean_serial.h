//
// Created by ayush on 4/29/21.
//

#ifndef CS5234_FINAL_PROJECT_KMEAN_SERIAL_H
#define CS5234_FINAL_PROJECT_KMEAN_SERIAL_H
#pragma once
#include "kmeans_new.h"
#include "kmeans_util.h"

using namespace std;
using DataFrame = std::vector<vector<float>>;
namespace kmeans_serial {
    int32_t *calculateMeans_serial(int k, vector<v_float> data_array, long iteration, vector<v_float> &means_array);
}
#endif //CS5234_FINAL_PROJECT_KMEAN_SERIAL_H
