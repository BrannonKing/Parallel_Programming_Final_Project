//
// Created by ayush on 4/29/21.
//

#ifndef CS5234_FINAL_PROJECT_KMEAN_SERIAL_H
#define CS5234_FINAL_PROJECT_KMEAN_SERIAL_H
#pragma once
#include "kmeans_new.h"
#include "kmeans_util.h"

using namespace std;

namespace kmeans_serial {
      void calculateMeans_serial(
        int k,
        vector<v_float> const &data_array,
        long iteration,
        vector<v_float>& means_array,
        vector<int32_t>& membership,
        vector<int32_t>& cluster_size, int num_threads);
}
#endif //CS5234_FINAL_PROJECT_KMEAN_SERIAL_H
