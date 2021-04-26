#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

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
    buf             = (float*) malloc(ret.npoints*ret.nfeatures*sizeof(float));
    ret.features    = (float**)malloc(ret.npoints*sizeof(float*));
    ret.features[0] = (float*) malloc(ret.npoints*ret.nfeatures*sizeof(float));
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
//    for(int i=0;i<30;i++)
//    {
//        for(int j=0;j<ret.nfeatures;j++)
//        {
//            printf("M[i][j]= %f ",ret.features[i* ret.nfeatures +j]);
//        }
//        printf(("\n"));
//
//    }
    memcpy(ret.features[0], buf, ret.npoints*ret.nfeatures*sizeof(float)); /* now features holds 2-dimensional array of features */
    free(buf);

    return ret;
}
float calc_distance(int dim, float *p1, float *p2)
{
    float distance_sq_sum = 0;

    for (int ii = 0; ii < dim; ii++)
        distance_sq_sum += ((p1[ii] - p2[ii])* (p1[ii] - p2[ii]));

    return distance_sq_sum;

}

float* find_cluster(float** data_array , float** means_array,long N,long dim,int k);

float* init_max(float ** data_array,long dim,long N)
{
    float* max_array = (float *) malloc(sizeof (float )*dim);


    memset(max_array,INT32_MIN,dim* sizeof(float));


    for(long i =0;i<dim;i++)
    {
        for(long j=0;j<N;j++)
        {
            max_array[j]= data_array[i][j] > max_array[i] ? data_array[i][j] : max_array[j];
        }
    }
    return  max_array;

}

float* init_min(float ** data_array,long dim,long N)
{
    float* min_array = (float *) malloc(sizeof(float)*dim);


    memset(min_array,INT32_MAX,dim* sizeof(float));


    for(long i =0;i<dim;i++)
    {
        for(long j=0;j<N;j++)
        {
            min_array[j]= data_array[i][j] < min_array[i] ? data_array[i][j] : min_array[j];
        }
    }
    return  min_array;

}
float float_rand( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}


float** init_means(float **data_array, long dim,short k,float* min_array,float* max_array)
{

    float** means_array = (float **) malloc(sizeof (float )*dim*k);
//    int num = (rand() %
//                   (upper - lower + 1)) + lower;

    for(short i=0;i<k;i++)
    {
        for(long j=0;j<dim;j++)
        {
            // means_array[i][j] = (rand() % (max_array[j] - min_array[j] + 1)) + min_array[j];
            means_array[i][j] = float_rand(min_array[j],max_array[j]);
        }
    }

    return  means_array;
}

//
float* update_mean(long n,int dim,float * means_row,float* data_row)
{
    // update mean for each feature 
    for(int i =0;i<dim;i++)
    {
        float m= means_row[i];
        m = (m*(n-1)+data_row[i]) / (float) n;
        means_row[i] = m;
    }
    return means_row;
}  



long classify(float** means_array,float* data_row,long dim,int k)
{
    float min= INT32_MIN;
    long index = -1;

    // for each centroid
    for( int i=0;i<k ;i++)
    {

        // for each feature
        float dist_temp = calc_distance(dim,data_row,means_array[i]);
        if(dist_temp < min)
        {
            min = dist_temp;
            index =i;
        }
    }

    return  index;
}

// dim = no of features
float** calculate_means(long k,float** data_array,long iteration,int dim,long N)
{
    //Find the minima and maxima for columns
    float * max_array = init_max(data_array,dim,N);
    float * min_array = init_min(data_array,dim,N);

    // initialize means to random points , means array = dim x K
    float** means_array = init_means(data_array,dim,k,min_array,max_array);

    // no of items per cluster.
    int cluster_size[k];
    memset(cluster_size,0,k*sizeof (int));
    // membership , point i belong to membership[i] cluster
    int membership[N];
    memset(cluster_size,0,N*sizeof (int));


    for(int jj =0;jj<iteration;jj++) {
        int nochange = 1;
        for (int ii = 0; ii < N; ii++) {


            float *item_row = data_array[ii];
            // classify return the value from 0->k

            int index = classify(means_array, item_row, dim, k);
            cluster_size[index] += 1;
            int csize = cluster_size[index];
            // update centroids/means array, for this cluster only ( coz new value added)
            means_array[ii] = update_mean(csize, dim,means_array[index], item_row);
            if (index != membership[ii])
                nochange = 0;

            //point i belongs to cluster index/membership[i]
            membership[ii] = index;

        }

        if(nochange)
            break;
    }

    return means_array;
}


int main(int argc, char **argv) {
    srand(time(0));
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
    
    // int32_t* membership = (int*) memalign(AOCL_ALIGNMENT,fd.npoints*sizeof (int32_t));

    float **means_array = calculate_means(3,fd.features,3,fd.nfeatures,fd.npoints);
    float* membership = find_cluster(fd.features,means_array,fd.npoints,fd.nfeatures,3);

    printf("done");
    return 0;
}

float* find_cluster(float** data_array , float** means_array,long N,long dim,int k)
{
    float* clusters = memalign(AOCL_ALIGNMENT,sizeof(float)*N);
    int index;
    for(int i =0;i<N;i++)
    {
        index = classify(means_array,data_array[i],dim,k);
        clusters[i] = index ;

    }

    return clusters;

}