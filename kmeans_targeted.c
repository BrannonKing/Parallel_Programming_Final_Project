#include <float.h>
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
    float *features; // [npoints*nfeatures + f]
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

//#pragma omp declare target // for doing the whole thing on the GPU
int isClose(float *restrict centroids, float *restrict old_centroids, int K, int N, float tolerance) {
    for (int i = 0; i < K * N; ++i)
        if (fabsf(centroids[i]-old_centroids[i]) > tolerance) return 0;

    return 1;
}

void update_centroids(const int *restrict labels, const float *restrict features, int N, int F, int *labelCounts, float *centroids, int K){
    memset(centroids, 0, K * F * sizeof(float));
    memset(labelCounts, 0, sizeof(int) * K);

// EXPECTING SERIAL!
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

void update_labels(int *labels, const float *restrict features, int N, int F, const float *restrict centroids, int K) {
#pragma omp target teams distribute parallel for simd \
        map(to: centroids[0:K*F]), map(from: labels[0:N])
    for (int i = 0; i < N; ++i) { // each point
        float best_distance = FLT_MAX; //   INFINITY;
        int best_centroid = -1;
        for (int k = 0; k < K; ++k) { // each cluster
            float distance = 0.0f;
            for (int j = 0; j < F; ++j) { // each feature
                float d = centroids[k * F + j] - features[i * F + j];
                distance += d*d;
            }
            if (distance < best_distance) {
                best_distance = distance;
                best_centroid = k;
            }
        }
        labels[i] = best_centroid;
    }
}
//#pragma omp end declare target // for doing the whole thing on the GPU

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
    const float* features = fd.features;
    const int32_t N = fd.npoints;
    const int32_t F = fd.nfeatures;
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
    for (int k = 0; k < K; ++k)
        memcpy(&centroids[k * F], &features[(rand() % N) * F], F * sizeof(float));
    int* labels = (int*)malloc(sizeof(int) * N);
    int* labelCounts = (int*)malloc(sizeof(int) * K);

    printf("Starting computation...\n");
    double start = omp_get_wtime();
    int iterations = 0;

#pragma omp target data map(to: features[0:F*N])
    while (++iterations < max_iterations) {
        memcpy(old_centroids, centroids, sizeof(float) * K * F);
        update_labels(labels, features, N, F, centroids, K);
        update_centroids(labels, features, N, F, labelCounts, centroids, K);
        if (isClose(centroids, old_centroids, K, F, 0.001f)) break;
    }
    printf("Done %u iterations in %f seconds.\n", iterations, omp_get_wtime() - start);
    for (int i=0; i < K; ++i) {
        printf("Center %d with %d:", i, labelCounts[i]);
        for (int j=0; j < fd.nfeatures; ++j) {
            printf(" %f,", centroids[i*fd.nfeatures + j]);
        }
        printf("\n");
    }

    return 0;
}
