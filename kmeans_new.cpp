#include "kmeans_new.h"
#include "kmeans_util.h"

using namespace std;
using DataFrame = std::vector<vector<float>>;
vector<vector<float>> means_array;
long N;
long M;
/***
 *
 * @param k - no of centroids
 * @param min_array : array having minimum of each feature
 * @param max_array : array having maximum of each feature
 * @param means_array: kxM array, basically there are k centroid of M dimension/features
 */
void init_means(short k,v_float const min_array,v_float const max_array,vector<vector<float>>& means_array)
{
    for(short i=0;i<k;i++)
    {
        v_float vec(M,0);
        for(long j=0;j<M;j++)
        {

            vec[j] = float_rand(min_array[j],max_array[j]);
//            vec.push_back((min_array[j]+max_array[j])/8);
        }

        means_array.push_back(vec);
    }

}

void init_means2(short k,vector<vector<float>>& means_array, float** data_array)
{
//    srand(43);
    for(short i=0;i<k;i++)
    {
        v_float vec(M,0);
        for(long j=0;j<M;j++)
        {

            float* data_row = data_array[rand() % N];
//            memcpy(&centroids[i*fd.nfeatures], fd.features[rand() % fd.npoints], fd.nfeatures * sizeof(float));
            vec[i] = data_row[i];
        }

        means_array.push_back(vec);
    }

}

void init_means3(short k,vector<vector<float>>& means_array, vector<v_float> data_array)
{
//    srand(43);
    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());
    std::uniform_int_distribution<size_t> indices(0, N - 1);
    for (auto& cluster : means_array) {
        cluster = data_array[indices(random_number_generator)];
    }

}

/***
 * Update the k centroids of M feature
 * @param N : no of elements in this cluster
 * @param means_row: the centroid ?
 * @param data_row: the single point ?
 */
void update_mean(int N, vector<float>& means_row, v_float data_row)
{
    if(N < 0) {
        cout<<"N is less than zero error";
        exit(5);
    }
    // update mean for each feature
    for(int i =0;i<M;i++)
    {
        float m= means_row[i];
        m *= (N-1);
        m += data_row[i];
        m = m/N;
        means_row[i] = m;
    }
}

void update_old_cluster_mean(int old_size, vector<float>& means_row, v_float data_row)
{
    if(N < 0) {
        cout<<"N is less than zero error";
        exit(5);
    }
    // update mean for each feature
    for(int i =0;i<M;i++)
    {
        float m= means_row[i];
        m *= old_size;
        m -= data_row[i];
        m = m/(old_size-1);
        means_row[i] = m;
    }
}


/***
 * For each point having M dimensions/features
 * Calculate
 * @param means_array - Mxk array of centroids
 * @param data_row - distance vector rows
 * @param k
 * @return
 */
int classify(v_float distances,int k)
{
    float min= INT32_MAX;
    long index = -1;

    // for each centroid
    for( int i=0;i<distances.size() ;i++)
    {

        // calculate distance
        float dist_temp = distances[i];
        if(dist_temp < min)
        {
            min = dist_temp;
            index =i;
        }
    }

    return  index;
}



priority_queue <mem_point,std::vector<mem_point>,  custom_cmpr> priority_q;
int isClose(vector<v_float>&  c1, vector<v_float>& c2, float tolerance, int k) {
    for (int i = 0; i < k; ++i) {
        for(int j =i;j<M;j++)
    if (abs(c1[i][j] - c2[i][j]) > tolerance) return 0;
}
    return 1;
}

// dim = no of features
int32_t* calculateMeans_serial(int k, vector<v_float> data_array, long iteration, vector<v_float>& means_array)
{
    cout<<"k: "<<k<<endl;
    cout<<"iterations "<<iteration<<endl;

    // initialize means to random points , means array = dim x K
    init_means3(k,means_array,data_array);

    int32_t* membership = (int32_t*) memalign(AOCL_ALIGNMENT,N*sizeof (int32_t));
    int32_t* cluster_size = (int32_t*) memalign(AOCL_ALIGNMENT,k*sizeof (int32_t));
    memset(membership,-1,N*sizeof(int32_t));
    memset(cluster_size,0,k*sizeof(int32_t));
    vector<v_float> oldmean(k);

        // for each point
        for(int j=0;j<iteration;j++)
        {
        for (int ii = 0; ii < N; ii++) {
            v_float dist(k, 0);

            auto item_row = data_array[ii];

            // classify this point return the value from 0->k
            for (int i = 0; i < k; i++) {
                // calculate distance between each k centroid
                float d = calc_distance(item_row, means_array[i], M);
                dist[i] = d;
            }
            int index = classify(dist, k);
            if (index != membership[ii])
            {
                oldmean = means_array;
                if(cluster_size[membership[ii]] > 0) {
                    int old_cluster = membership[ii];
                    int old_cluster_size = cluster_size[old_cluster];
                    update_old_cluster_mean(old_cluster_size,means_array[old_cluster],item_row);
                    cluster_size[membership[ii]] -= 1; // decrement count from previous
                }
                membership[ii] = index; // update membership mapping
                cluster_size[index] += 1; // increament count in new cluster
                int csize = cluster_size[index];
                update_mean(csize, means_array[index], item_row);
            }
        }
    }
//    while( !isClose(oldmean,means_array,k,0.001));

    return cluster_size;
}


int32_t* calculateMeans_omp(int k, vector<v_float> data_array, long iteration, vector<v_float>& means_array)
{
    cout<<"k: "<<k<<endl;
    cout<<"iterations "<<iteration<<endl;

    // initialize means to random points , means array = dim x K
    init_means3(k,means_array,data_array);

    int32_t* membership = (int32_t*) memalign(AOCL_ALIGNMENT,N*sizeof (int32_t));
    int32_t* cluster_size = (int32_t*) memalign(AOCL_ALIGNMENT,k*sizeof (int32_t));
    memset(membership,-1,N*sizeof(int32_t));
    memset(cluster_size,0,k*sizeof(int32_t));
//    vector<v_float> oldmean(k);

    // for each point
    #pragma omp parallel shared(k,data_array,iteration,means_array)
    {
        for (int j = 0; j < iteration; j++) {
            for (int ii = 0; ii < N; ii++) {
                v_float dist(k, 0);

                auto item_row = data_array[ii];

                // classify this point return the value from 0->k
                for (int i = 0; i < k; i++) {
                    // calculate distance between each k centroid
                    float d = calc_distance(item_row, means_array[i], M);
                    dist[i] = d;
                }
                int index = classify(dist, k);
                if (index != membership[ii]) {
//                    oldmean = means_array;
                    if (cluster_size[membership[ii]] > 0) {
                        int old_cluster = membership[ii];
                        int old_cluster_size = cluster_size[old_cluster];
                        update_old_cluster_mean(old_cluster_size, means_array[old_cluster], item_row);
                        cluster_size[membership[ii]] -= 1; // decrement count from previous
                    }
                    membership[ii] = index; // update membership mapping
                    cluster_size[index] += 1; // increament count in new cluster
                    int csize = cluster_size[index];
                    update_mean(csize, means_array[index], item_row);
                }
            }
        }
    }
//    while( !isClose(oldmean,means_array,k,0.001));

    return cluster_size;
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
    N = fd.npoints;
    M = fd.nfeatures;
    int32_t* membership = (int*) memalign(AOCL_ALIGNMENT,fd.npoints*sizeof (int32_t));
    ios_base::sync_with_stdio(false);


    DataFrame  data(N);
    for(long i =0;i<N;i++)
    {
        vector<float> vec(M,0);
        for(int j=0;j<M;j++)
        {
            vec[j] = fd.features[i][j];
        }
        data[i] = vec;
    }

    ios_base::sync_with_stdio(false);
    auto start = std::chrono::high_resolution_clock::now();
    int k =9;
    int iterations = 10;
    vector<v_float> means_arr(k, v_float(M,0));
    int32_t * cluster_map = calculateMeans_serial(k, data, iterations, means_arr);
//    membership = find_cluster(fd.features,k);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << "It took " << time_span.count()*1e03<<setprecision(9) << " milli seconds.\n";

    for(int i=0;i<k;i++)
        cout<<"no of elements in cluster "<<i<<": "<<cluster_map[i]<<endl;
    long sun =0;
    for(int i=0 ;i < k;i++)
        sun+=cluster_map[i];
    sun == N ? cout<<"correctness pass" : cout<<"correctness fail, total sum: "<<sun<<endl;
    return 0;
}


int32_t* find_cluster(float** data_array,int k, int csize)
{
    int32_t* clusters = (int32_t*)  memalign(AOCL_ALIGNMENT,sizeof(int)*csize);
    int index;
//    for(int i =0;i<csize;i++)
//    {
//        index = classify(data_array[i],k);
//        clusters[i] = index ;
//
//    }

    return clusters;

}