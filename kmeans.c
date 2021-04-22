#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define AOCL_ALIGNMENT 64

struct FeatureDefinition {
    int32_t npoints, nfeatures;
    float **features; // [npoints][nfeatures]
};

struct FeatureDefinition load_file(char* filename) {
    FILE *infile;
    char line[2048];
    float *buf;
    int32_t i, j;
    struct FeatureDefinition ret;
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
    printf("Number of features: %d\n", ret.nfeatures);

    memcpy(ret.features[0], buf, ret.npoints*ret.nfeatures*sizeof(float)); /* now features holds 2-dimensional array of features */
    free(buf);

    return ret;
}
int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Invalid parameters. Usage: kmeans <clusters> <input file>");
        return 2;
    }
    int clusters = atoi(argv[1]);
    if (clusters <= 0) {
        fprintf(stderr, "Invalid cluster count. Usage: kmeans <clusters> <input file>");
        return 3;
    }
    struct FeatureDefinition fd = load_file(argv[2]);
    // do something with fd:
    return 0;
}
