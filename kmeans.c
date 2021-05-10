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
    float **features; // [npoints][nfeatures]
} FeatureDefinition;

FeatureDefinition load_file(char* filename) {
    FILE *infile;
    char line[2048];
    float *buf;
    int32_t i, j;
    FeatureDefinition ret;
    ret.nfeatures = ret.npoints = 0;

    if ((infile = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        exit(1);
    }
    while (fgets(line, 1024, infile) != NULL)
        if (strtok(line, " \t\n") != 0)
            ret.npoints++;
    rewind(infile);
    while (fgets(line, 1024, infile) != NULL) {
        if (strtok(line, " \t\n") != 0) {
            /* ignore the id (first attribute): nfeatures = 1; */
            while (strtok(NULL, " ,\t\n") != NULL) ret.nfeatures++;
            break;
        }
    }

    /* allocate space for features[] and read attributes of all objects */
    buf             = (float*) memalign(AOCL_ALIGNMENT,ret.npoints*ret.nfeatures*sizeof(float));
    ret.features    = (float**)memalign(AOCL_ALIGNMENT,ret.npoints*sizeof(float*));
    ret.features[0] = (float*) memalign(AOCL_ALIGNMENT,ret.npoints*ret.nfeatures*sizeof(float));
    for (i=1; i<ret.npoints; i++)
        ret.features[i] = ret.features[i-1] + ret.nfeatures;
    rewind(infile);
    i = 0;
    while (fgets(line, 1024, infile) != NULL) {
        if (strtok(line, " \t\n") == NULL) continue;
        for (j=0; j<ret.nfeatures; j++) {
            buf[i] = atof(strtok(NULL, " ,\t\n"));
            i++;
        }
    }
    fclose(infile);

    printf("\nI/O completed\n");
    printf("\nNumber of objects: %d\n", ret.npoints);
    printf("Number of features: %d\n\n", ret.nfeatures);

    memcpy(ret.features[0], buf, ret.npoints*ret.nfeatures*sizeof(float)); /* now features holds 2-dimensional array of features */
    free(buf);

    return ret;
}

int isClose(float *restrict c1, float *restrict c2, int n, int p, float tolerance) {
    for (int i = 0; i < n*p; ++i)
        if (fabsf(c1[i]-c2[i]) > tolerance) return 0;

    return 1;
}

void update_centroids(const int *restrict labels, FeatureDefinition* fd, int *labelCounts, float *centroids, int n){
    // memset(centroids, 0, n * fd->nfeatures * sizeof(float));
    // dont' want to have to do locks for multiple threads updating the same centroid:
    #pragma omp parallel for
    for (int c = 0; c < n; ++c) {
        int counter = 0;
        for (int i = 0; i < fd->npoints; ++i) {
            if (labels[i] != c) continue;
            if (counter++ == 0) {
                #pragma omp simd
                for (int j = 0; j < fd->nfeatures; ++j) {
                    centroids[c * fd->nfeatures + j] = fd->features[i][j];
                }
                continue;
            }
            #pragma omp simd
            for (int j = 0; j < fd->nfeatures; ++j) {
                centroids[c * fd->nfeatures + j] += fd->features[i][j];
            }
        }
        labelCounts[c] = counter;
        if (counter <= 0) continue;
        #pragma omp simd
        for (int j = 0; j < fd->nfeatures; ++j) {
            centroids[c * fd->nfeatures + j] /= (float)counter;
        }
    }
}

void update_centroids_locked(const int *restrict labels, FeatureDefinition* fd, int *labelCounts, float *centroids, int k, omp_lock_t* locks){
    memset(centroids, 0, k * fd->nfeatures * sizeof(float));
    memset(labelCounts, 0, sizeof(int) * k);

    #pragma omp parallel
    {
        int *localCounts = (int *) calloc(k, sizeof(int));
        float *localCentroids = (float *) calloc(k * fd->nfeatures, sizeof(float));
        #pragma omp for
        for (int i = 0; i < fd->npoints; ++i) {
            int c = labels[i];
            localCounts[c] += 1;
            #pragma omp simd
            for (int j = 0; j < fd->nfeatures; ++j) {
                localCentroids[c * fd->nfeatures + j] += fd->features[i][j];
            }
        }
        #pragma omp critical
        {
            for (int c = 0; c < k; ++c) {
                labelCounts[c] += localCounts[c];
                #pragma omp simd
                for (int j = 0; j < fd->nfeatures; ++j) {
                    centroids[c * fd->nfeatures + j] += localCentroids[c * fd->nfeatures + j];
                }
            }
        }
        free(localCounts);
        free(localCentroids);
    }
    for (int c = 0; c < k; ++c) {
        float count = (float) (labelCounts[c]);
        if (count <= 0) continue;
#pragma omp simd
        for (int j = 0; j < fd->nfeatures; ++j) {
            centroids[c * fd->nfeatures + j] /= count;
        }
    }

}

void update_labels(int *labels, FeatureDefinition* fd, const float *restrict centroids, int n) {
    #pragma omp parallel for
    for (int i = 0; i < fd->npoints; ++i) { // each point
        float best_distance = INFINITY;
        int best_centroid = -1;
        #pragma omp simd
        for (int c = 0; c < n; ++c) { // each cluster
            float distance = 0.0f;
            for (int j = 0; j < fd->nfeatures; ++j) { // each feature
                float d = centroids[c*fd->nfeatures + j] - fd->features[i][j];
                distance += d*d;
            }
            if (distance < best_distance) {
                best_distance = distance;
                best_centroid = c;
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
    int k = atoi(argv[1]);
    if (k <= 0) {
        fprintf(stderr, "Invalid cluster count. " USAGE);
        return 3;
    }
    int max_iterations = atoi(argv[2]);
    if (max_iterations <= 0) {
        fprintf(stderr, "Invalid maximum iterations. " USAGE);
        return 3;
    }


    FeatureDefinition fd = load_file(argv[3]);
    // do something with fd:

    // algorithm:
    // centroids = random(k, points)
    // while centroids != old centroids:
    //    old centroids = centroids, iterations++
    //    labels = getlabels(points, centroids)
    //    centroids = labels.groupby().mean()

    srand(43);
    //srand(time(0));

    float *old_centroids = (float*)malloc(k * fd.nfeatures * sizeof(float));
    memset(old_centroids, 0, k * fd.nfeatures * sizeof(float));
    float* centroids = (float*)malloc(k * fd.nfeatures * sizeof(float));
    for (int i = 0; i < k; ++i)
        memcpy(&centroids[i*fd.nfeatures], fd.features[rand() % fd.npoints], fd.nfeatures * sizeof(float));
    int* labels = (int*)malloc(sizeof(int) * fd.npoints);
    int* labelCounts = (int*)malloc(sizeof(int) * k);

    omp_lock_t* locks = (omp_lock_t*)calloc(k, sizeof(omp_lock_t));
//    for (int i = 0; i < k; ++i) {
//        omp_init_lock_with_hint(&locks[i], omp_lock_hint_speculative);
//    }

    printf("Starting computation...\n");
    double start = omp_get_wtime();
    int iterations = 0;
    while (++iterations < max_iterations && !isClose(centroids, old_centroids, k, fd.nfeatures, 0.001f)) {
        memcpy(old_centroids, centroids, sizeof(float) * k * fd.nfeatures);
        update_labels(labels, &fd, centroids, k);
        update_centroids_locked(labels, &fd, labelCounts, centroids, k, locks);
        //update_centroids(labels, &fd, labelCounts, centroids, k);
    }
    printf("Done %u iterations in %f seconds.\n", iterations, omp_get_wtime() - start);
    for (int i=0; i < k; ++i) {
        printf("Center %d with %d:", i, labelCounts[i]);
        for (int j=0; j < fd.nfeatures; ++j) {
            printf(" %f,", centroids[i*fd.nfeatures + j]);
        }
        printf("\n");
    }

//    for (int i = 0; i < k; ++i) {
//        omp_destroy_lock(&locks[i]);
//    }

    return 0;
}
