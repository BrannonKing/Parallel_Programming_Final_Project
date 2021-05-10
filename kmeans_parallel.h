//
// Created by ayush on 4/29/21.
//

#ifndef CS5234_FINAL_PROJECT_KMEANS_PARALLEL_H
#define CS5234_FINAL_PROJECT_KMEANS_PARALLEL_H

#pragma once
#include "kmeans_new.h"
#include "kmeans_util.h"
struct  Membership {
    int centroid;
    float distance;
}__attribute__((packed));
typedef struct Membership mem_point;
namespace kmeansparallel {
    struct centroid_data
    {
        float* membership = NULL;

        unsigned long count =0;
        centroid_data()
        {



        }

        centroid_data(v_float membership)
        {
            this->membership = (float *) memalign(AOCL_ALIGNMENT, M * sizeof(M));
//            memset(membership,1,M);
           for(int i=0;i<M;i++)
           {
               this->membership[i]  = membership[i];
           }
        }
        void push(v_float point)
        {
#pragma omp parallel for schedule(static)
           for(int i=0;i<M;i++)
           {

               float m = membership[i];
               m *= count;
               m += point[i];
               m = m / (count+1);
               membership[i] = m;
           }

           count++;
        }

        void update_mean(int N, v_float &means_row, v_float data_row) {
            if (N < 0) {
                cout << "N is less than zero error";
                exit(5);
            }
            // update mean for each feature
            for (int i = 0; i < M; i++) {
                float m = means_row[i];
                m *= (N - 1);
                m += data_row[i];
                m = m / N;
                means_row[i] = m;
            }
        }
    };
    int32_t*  calculateMeans_omp(int k, vector<v_float> data_array, long iteration, vector<v_float> &means_array);
}
#endif //CS5234_FINAL_PROJECT_KMEANS_PARALLEL_H
