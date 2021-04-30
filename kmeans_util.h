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
int32_t* find_cluster(float** data_array,int k, int csize);
void init_max(vector<v_float> data_array,v_float& max_array, int N ,int M);
void init_min(vector<v_float> data_array,v_float& min_array, int N, int M);

float float_rand( float min, float max );
void print_vect(vector<float> const ar);
#endif //CS5234_FINAL_PROJECT_KMEANS_UTIL_H
