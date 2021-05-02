//
// Created by ayush on 4/29/21.
//

#pragma once
#ifndef CS5234_FINAL_PROJECT_KMEANS_UTIL_H
#define CS5234_FINAL_PROJECT_KMEANS_UTIL_H
#include "kmeans_new.h"
using namespace std;

struct FeatureDefinition load_file(char* filename);
float calc_distance(v_float p1, vector<float> const p2, int M);
float float_rand( float min, float max );
char print_vect(vector<int32_t> const ar);

#endif //CS5234_FINAL_PROJECT_KMEANS_UTIL_H
