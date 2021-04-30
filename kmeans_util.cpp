//
// Created by ayush on 4/29/21.
#include "kmeans_util.h"
using namespace  std;
long N;
long M;

/***
 * Utility function to read a file
 * @param filename
 * @return
 */
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

/***
 * Calculate the distance of 2 points in M dimension plane, Euclidean distance
 * @param p1
 * @param p2
 * @return a distance
 */
float calc_distance(v_float p1, vector<float> const p2, int M)
{
    float distance_sq_sum = 0;


    for (int ii = 0; ii < M; ii++)
        distance_sq_sum += ((p1[ii] - p2[ii])* (p1[ii] - p2[ii]));

    return sqrtf(distance_sq_sum);

}

/***
 *
 * @param min
 * @param max
 * @return
 */
float float_rand( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}


void print_vect(vector<float> const ar)
{
    for(int i =0;i<ar.size();i++)
        cout<<ar[i]<<" ";

    cout<<endl;
}
//

