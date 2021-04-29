#include "kmeans_new.h"
#include "kmeans_util.h"

using namespace std;
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

/***
 * Update the k centroids of M feature
 * @param N : no of elements in this cluster
 * @param means_row: the centroid ?
 * @param data_row: the single point ?
 */
void update_mean(int N, vector<float>& means_row, float *__restrict__  data_row)
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

// dim = no of features
int32_t* calculate_means(int k,float** data_array,long iteration)
{
    vector<int> abcd(M);
    vector<float> min_array(M, INT32_MAX);
    vector<float> max_array(M, INT32_MIN);
    cout<<"k: "<<k<<endl;
    cout<<"iterations "<<iteration<<endl;

    init_max(data_array, max_array, N, M );
    init_min(data_array,min_array, N , M);

    // initialize means to random points , means array = dim x K

    init_means(k,min_array,max_array,means_array);

    int32_t* membership = (int32_t*) memalign(AOCL_ALIGNMENT,N*sizeof (int32_t));
    int32_t* cluster_size = (int32_t*) memalign(AOCL_ALIGNMENT,k*sizeof (int32_t));
    memset(membership,-1,N*sizeof(int32_t));
    memset(cluster_size,0,k*sizeof(int32_t));

    for(int jj =0;jj<iteration;jj++) {
        int nochange = 1;
//        memset(membership,-1,N*sizeof(int32_t));
//        memset(cluster_size,0,k*sizeof(int32_t));
        // for each point
        for (int ii = 0; ii < N; ii++) {
            v_float dist(k,0);
            priority_queue<float,v_float,custom_cmpr> que;
            float *item_row = data_array[ii];

            // classify this point return the value from 0->k
            for(int i =0;i<k;i++) {
                // calculate distance between each k centroid
                float d = calc_distance(data_array[ii], means_array[i],M);
                dist[i] = d;
                que.push(d);
//                mem_point p;
//                p.distance = d;
//                p.centroid = i;
//                        priority_q.push(p);
            }
            int index = classify(dist, k);
            if(index == membership[ii])
            {
                // this implies no change in membership

            }
            else {
                if(cluster_size[membership[ii]] > 0)
                    cluster_size[membership[ii]] -= 1; // decrement count from previous

                membership[ii] = index; // update membership mapping
                cluster_size[membership[ii]] += 1; // incerement count in new cluster
                int csize = cluster_size[index];
                update_mean(csize, means_array[index], item_row);

                nochange = 0;
            }
            //point i belongs to cluster index/membership[i]

            // update this centroid/means array, for this cluster only ( coz new value added)

        }



        if(nochange)
            break;
    }


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
    auto start = std::chrono::high_resolution_clock::now();
    int k =9;
    int iterations = 50;
    int32_t * cluster_map = calculate_means(k,fd.features,iterations);
//    membership = find_cluster(fd.features,k);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << "It took " << time_span.count()*1e03<<setprecision(9) << " milli seconds.\n";

    for(int i=0;i<k;i++)
        cout<<"no of elements in cluster "<<i<<": "<<cluster_map[i]<<endl;


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