#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define AOCL_ALIGNMENT 64

typedef struct {
    int32_t npoints, nfeatures;
    float *features; // [npoints x nfeatures]
} FeatureDefinition;

FeatureDefinition load_file(char* filename) {
    FILE *infile;
    char line[2048];
    int32_t i, j;
    FeatureDefinition ret;
    ret.nfeatures = ret.npoints = 0;

    if ((infile = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        exit(1);
    }
    while (fgets(line, 2048, infile) != NULL)
        if (strtok(line, " \t\n") != 0)
            ret.npoints++;
    rewind(infile);
    while (fgets(line, 2048, infile) != NULL) {
        if (strtok(line, " \t\n") != 0) {
            /* ignore the id (first attribute): nfeatures = 1; */
            while (strtok(NULL, " ,\t\n") != NULL) ret.nfeatures++;
            break;
        }
    }

    /* allocate space for features[] and read attributes of all objects */
    ret.features = (float*) memalign(AOCL_ALIGNMENT,ret.npoints*ret.nfeatures*sizeof(float));
    rewind(infile);
    i = 0;
    while (fgets(line, 2048, infile) != NULL) {
        if (strtok(line, " \t\n") == NULL) continue;
        for (j=0; j<ret.nfeatures; ++j) {
            float value = atof(strtok(NULL, " ,\t\n"));
            ret.features[i * ret.nfeatures + j] = value;
        }
        ++i;
    }
    fclose(infile);

    printf("\nI/O completed\n");
    printf("Number of objects: %d\n", ret.npoints);
    printf("Number of features: %d\n\n", ret.nfeatures);

    return ret;
}

int isClose(float *restrict c1, float *restrict c2, int n, int p, float tolerance) {
    for (int i = 0; i < n*p; ++i)
        if (fabsf(c1[i]-c2[i]) > tolerance) return 0;

    return 1;
}

void update_centroids_0(const int *restrict labels, const float *restrict features, int N, int F, int *labelCounts, float *centroids, int K){
    memset(centroids, 0, K * F * sizeof(float));
    memset(labelCounts, 0, sizeof(int) * K);

// MUST BE SERIAL!
    for (int i = 0; i < N; ++i) {
        int k = labels[i];
        labelCounts[k] += 1;
#pragma omp simd
        for (int j = 0; j < F; ++j) {
            centroids[k * F + j] += features[i * F + j];
        }
    }
    for (int k = 0; k < K; ++k) {
        float count = (float) (labelCounts[k]);
        if (count <= 0) continue;
#pragma omp simd
        for (int j = 0; j < F; ++j) {
            centroids[k * F + j] /= count;
        }
    }
}


void update_centroids_1(const int *restrict labels, const float *restrict features, int N, int F, int *labelCounts, float *centroids, int K){
    memset(centroids, 0, K * F * sizeof(float));
    memset(labelCounts, 0, sizeof(int) * K);

#pragma omp parallel for
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < N; ++i) {
            if (k != labels[i]) continue;
            labelCounts[k] += 1;
#pragma omp simd
            for (int j = 0; j < F; ++j) {
                centroids[k * F + j] += features[i * F + j];
            }
        }
    }
    for (int k = 0; k < K; ++k) {
        float count = (float) (labelCounts[k]);
        if (count <= 0) continue;
#pragma omp simd
        for (int j = 0; j < F; ++j) {
            centroids[k * F + j] /= count;
        }
    }
}

void update_centroids_2_3(const int *restrict labels, const float *restrict features,
                             int N, int F, int *labelCounts, float *centroids, int K, omp_lock_t* locks){
    memset(centroids, 0, K * F * sizeof(float));
    memset(labelCounts, 0, sizeof(int) * N);

#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        int k = labels[i];
//#pragma omp atomic
        omp_set_lock(&locks[k]);
        labelCounts[k] += 1;
#pragma omp simd
        for (int j = 0; j < F; ++j) {
            float value = features[i * F + j];
            int index = k * F + j;
//#pragma omp atomic
            centroids[index] += value;
        }
        omp_unset_lock(&locks[k]);
    }
    for (int k = 0; k < K; ++k) {
        float count = (float)(labelCounts[k]);
        if (count <= 0) continue;
#pragma omp simd
        for (int j = 0; j < F; ++j) {
            centroids[k * F + j] /= count;
        }
    }
}

void update_centroids_4(const int *restrict labels, const float *restrict features, int N, int F, int *labelCounts, float *centroids, int K){
    memset(centroids, 0, K * F * sizeof(float));
    memset(labelCounts, 0, sizeof(int) * K);

#pragma omp parallel
    {
        int *localCounts = (int *) calloc(K, sizeof(int));  // TODO: reuse this memory between runs
        float *localCentroids = (float *) calloc(K * F, sizeof(float));
#pragma omp for
        for (int i = 0; i < N; ++i) {
            int c = labels[i];
            localCounts[c] += 1;
#pragma omp simd
            for (int j = 0; j < F; ++j) {
                localCentroids[c * F + j] += features[i * F + j];
            }
        }
#pragma omp critical
        {
            for (int k = 0; k < K; ++k) {
                labelCounts[k] += localCounts[k];
#pragma omp simd
                for (int j = 0; j < F; ++j) {
                    centroids[k * F + j] += localCentroids[k * F + j];
                }
            }
        }
        free(localCounts);
        free(localCentroids);
    }
    for (int k = 0; k < K; ++k) {
        float count = (float) (labelCounts[k]);
        if (count <= 0) continue;
#pragma omp simd
        for (int j = 0; j < F; ++j) {
            centroids[k * F + j] /= count;
        }
    }
}

void update_labels(int *labels, const float *restrict features, int N, int F, const float *restrict centroids, int K) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) { // each point
        float best_distance = INFINITY;
        int best_centroid = -1;
        for (int k = 0; k < K; ++k) { // each cluster
            float distance = 0.0f;
            for (int j = 0; j < F; ++j) { // each feature
                float d = centroids[k * F + j] - features[i * F + j];
                distance += d * d;
            }
            if (distance < best_distance) {
                best_distance = distance;
                best_centroid = k;
            }
        }
        labels[i] = best_centroid;
    }
}

#define USAGE "Usage: kmeans <cluster count k> <max iterations> <input file>"
int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Invalid parameters. " USAGE);
        return 2;
    }
    int K = atoi(argv[1]);
    if (K <= 0) {
        fprintf(stderr, "Invalid cluster count. " USAGE);
        return 3;
    }
    int max_iterations = atoi(argv[2]);
    if (max_iterations <= 0) {
        fprintf(stderr, "Invalid maximum iterations. " USAGE);
        return 3;
    }

    FeatureDefinition fd = load_file(argv[3]);
    int32_t N = fd.npoints;
    int32_t F = fd.nfeatures;

    // do something with fd:

    // algorithm:
    // centroids = random(K, points)
    // while centroids != old centroids:
    //    old centroids = centroids, iterations++
    //    labels = getlabels(points, centroids)
    //    centroids = labels.groupby().mean()

    srand(43);
    //srand(time(0));

    float *old_centroids = (float*)malloc(K * F * sizeof(float));
    memset(old_centroids, 0, K * F * sizeof(float));
    float* centroids = (float*)malloc(K * F * sizeof(float));
    for (int i = 0; i < K; ++i)
        memcpy(&centroids[i*F], &fd.features[(rand() % N) * F], F * sizeof(float));
    int* labels = (int*)malloc(sizeof(int) * N);
    int* labelCounts = (int*)malloc(sizeof(int) * K);

    printf("Starting computation...\n");
    double start = omp_get_wtime(), centroid_only = 0.0;
    int iterations = 0;
    while (++iterations < max_iterations && !isClose(centroids, old_centroids, K, F, 0.001f)) {
        memcpy(old_centroids, centroids, sizeof(float) * K * F);
        update_labels(labels, fd.features, N, F, centroids, K);
        double start_centroid = omp_get_wtime();
        update_centroids_4(labels, fd.features, N, F, labelCounts, centroids, K);
        centroid_only += omp_get_wtime() - start_centroid;
    }
    printf("Done %u iterations in %f seconds, in centroid update: %f seconds.\n",
           iterations, omp_get_wtime() - start, centroid_only);
    int features_to_show = F < 4 ? F : 4;
    for (int i=0; i < K; ++i) {
        printf("Center %d with %d objects:", i, labelCounts[i]);
        for (int j=0; j < features_to_show; ++j) {
            printf(" %f,", centroids[i*F + j]);
        }
        printf("\n");
    }

    return 0;
}
