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
        if (abs(c1[i]-c2[i]) > tolerance) return 0;

    return 1;
}

void update_centroids(int *restrict labels, FeatureDefinition* fd, int *labelCounts, float *centroids, int n){
    memset(centroids, 0, n * fd->nfeatures * sizeof(float));
    memset(labelCounts, 0, n * sizeof(int));
    // dont' want to have to do locks for multiple threads updating the same centroid:
    #pragma omp parallel for
    for (int c = 0; c < n; ++c) {
        for (int i = 0; i < fd->npoints; ++i) {
            if (labels[i] != c) continue;
            ++labelCounts[c];
            #pragma omp simd
            for (int j = 0; j < fd->nfeatures; ++j) {
                centroids[c * fd->nfeatures + j] += fd->features[i][j];
            }
        }
    }
    #pragma omp simd
    for (int c = 0; c < n; ++c){
        if (labelCounts[c] <= 0) continue;
        for (int j = 0; j < fd->nfeatures; ++j) {
            centroids[c * fd->nfeatures + j] /= labelCounts[c];
        }
    }
}

void update_labels(int *labels, FeatureDefinition* fd, float *restrict centroids, int n) {
    #pragma omp parallel for
    for (int i = 0; i < fd->npoints; ++i) {
        float best_distance = INFINITY;
        int best_centroid = -1;
        #pragma omp simd
        for (int c = 0; c < n; ++c) {
            float distance = 0.0;
            for (int j = 0; j < fd->nfeatures; ++j) {
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

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Invalid parameters. Usage: kmeans <k> <input file>");
        return 2;
    }
    int k = atoi(argv[1]);
    if (k <= 0) {
        fprintf(stderr, "Invalid cluster count. Usage: k-means <cluster count k> <input file>");
        return 3;
    }
    FeatureDefinition fd = load_file(argv[2]);
    // do something with fd:

    // algorithm:
    // centroids = random(k, points)
    // while centroids != old centroids:
    //    old centroids = centroids, iterations++
    //    labels = getlabels(points, centroids)
    //    centroids = labels.groupby().mean()

    //srand(43);
    srand(time(0));

    float *old_centroids = (float*)malloc(k * fd.nfeatures * sizeof(float));
    memset(old_centroids, 0, k * fd.nfeatures * sizeof(float));
    float* centroids = (float*)malloc(k * fd.nfeatures * sizeof(float));
    for (int i = 0; i < k; ++i)
        memcpy(&centroids[i*fd.nfeatures], fd.features[rand() % fd.npoints], fd.nfeatures * sizeof(float));
    int* labels = (int*)malloc(sizeof(int) * fd.npoints);
    int* labelCounts = (int*)malloc(sizeof(int) * k);

    printf("Starting computation...\n");
    double start = omp_get_wtime();
    while (!isClose(centroids, old_centroids, k, fd.nfeatures, 0.001)) {
        memcpy(old_centroids, centroids, sizeof(float) * k * fd.nfeatures);
        update_labels(labels, &fd, centroids, k);
        update_centroids(labels, &fd, labelCounts, centroids, k);
    }
    printf("Done in %f seconds.\n", omp_get_wtime() - start);
    for (int i=0; i < k; ++i) {
        printf("Center %d:", i);
        for (int j=0; j < fd.nfeatures; ++j) {
            printf(" %f,", centroids[i*fd.nfeatures + j]);
        }
        printf("\n");
    }

    return 0;
}
