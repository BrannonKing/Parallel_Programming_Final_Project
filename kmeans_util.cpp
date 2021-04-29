//
// Created by ayush on 4/29/21.
#include "kmeans_util.h"
using namespace  std;
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
float calc_distance(float* p1, vector<float> const p2, int M)
{
    float distance_sq_sum = 0;


    for (int ii = 0; ii < M; ii++)
        distance_sq_sum += ((p1[ii] - p2[ii])* (p1[ii] - p2[ii]));

    return sqrtf(distance_sq_sum);

}

/***
 *
 * @param data_array
 * @param k
 * @return
 */


/***
 *
 * @param data_array
 * @param max_array
 */
void init_max(float ** data_array,
              v_float& max_array, int N, int M)
{
    for(long i =0;i<N;i++)
    {
        for(long j=0;j<M;j++)
        {
            max_array[j]= data_array[i][j] > max_array[j] ? data_array[i][j] : max_array[j];
        }
    }
}

/***
 *
 * @param data_array
 * @param min_array
 */
void init_min(float ** data_array,v_float& min_array, int N, int M)
{
    for(long i =0;i<N;i++)
    {
        for(long j=0;j<M;j++)
        {
            min_array[j]= data_array[i][j] < min_array[j] ? data_array[i][j] : min_array[j];
        }
    }

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

