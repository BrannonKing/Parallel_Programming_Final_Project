#include "kmeans_parallel.h"
#include "omp.h"
//long N;
//long M;

/****
 *
 * @param k
 * @param means_array
 * @param data_array
 */
namespace kmeansparallel {

typedef struct centroid_data centroid_data;
    float calc_distance(v_float p1, float* p2, int M)
    {
        float distance_sq_sum = INFINITY;
        int ii;
        // #pragma omp parallel for schedule(static) simd
        for (ii = 0; ii < M; ii++)
        {
            float s = p1[ii] - p2[ii];
            distance_sq_sum += s*s;
        }

        return sqrtf(distance_sq_sum);

    }

    void init_means(short k, vector<v_float> &means_array, vector<v_float> data_array) {
//    srand(43);
        static std::random_device seed;
        static std::mt19937 random_number_generator(seed());
        std::uniform_int_distribution<size_t> indices(0, N - 1);
        int i;
//        #pragma omp for simd private(i)
        for (i=0;i<means_array.size();i++) {
            means_array[i] = data_array[indices(random_number_generator)];
        }

    }
    void init_means3(int k,vector<v_float> data_array,vector<v_float>& centroids) {
//    srand(43);
        static std::random_device seed;
        static std::mt19937 random_number_generator(seed());
        std::uniform_int_distribution<size_t> indices(0, N - 1);
//        #pragma omp for simd private(i)
        for (int i=0;i<k;i++) { // means arraay.size() = k
//            centroid_data  temp();
            centroids[i] = data_array[indices(random_number_generator)];
        }
//        return data_array[indices(random_number_generator)];
    }
    
    v_float init_means2(vector<v_float> data_array) {
//    srand(43);
        static std::random_device seed;
        static std::mt19937 random_number_generator(seed());
        std::uniform_int_distribution<size_t> indices(0, N - 1);
//        #pragma omp for simd private(i)
//        for (auto& item : means_array) { // means arraay.size() = k
//            item.push(data_array[indices(random_number_generator)]);
//        }
    return data_array[indices(random_number_generator)];
    }

/***
 * Update the k centroids of M feature
 * @param N : no of elements in this cluster
 * @param means_row: the centroid ?
 * @param data_row: the single point ?
 */
    void update_mean(int N, v_float &means_row, v_float data_row) {
        if (N < 0) {
            cout << "N is less than zero error";
            exit(5);
        }
        // update mean for each feature
        for (int i = 0; i < M; i++) {
            float m = means_row[i];
            m *= (N - 1);
            m += data_row[i];
            m = m / N;
            means_row[i] = m;
        }
    }

    void update_old_cluster_mean(int old_size, v_float &means_row, v_float data_row) {
        if (N < 0) {
            cout << "N is less than zero error";
            exit(5);
        }
        // update mean for each feature
        for (int i = 0; i < M; i++) {
            float m = means_row[i];
            m *= old_size;
            m -= data_row[i];
            m = m / (old_size - 1);
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
    int classify(v_float distances, int k) {
        float min = INT32_MAX;
        long index = -1;

        // for each centroid
        for (int i = 0; i < distances.size(); i++) {

            // calculate distance
            float dist_temp = distances[i];
            if (dist_temp < min) {
                min = dist_temp;
                index = i;
            }
        }

        return index;
    }


    

    int isClose(vector<v_float> &c1, vector<v_float> &c2, float tolerance, int k) {
        for (int i = 0; i < k; ++i) {
            for (int j = i; j < M; j++)
                if (abs(c1[i][j] - c2[i][j]) > tolerance) return 0;
        }
        return 1;
    }
    struct custom_cmpr {
    bool operator()(mem_point p1,mem_point p2) {
        // return "true" if "p1" is ordered before "p2", for example:
        if (p1.distance > p2.distance)
            return true;
        return false;
    }
};
    char print_arr(auto cluster_map, int k)
    {
        int i=0;
        for (auto item : cluster_map)
            cout <<"no of elements in cluster " << i++ << ": " << item.count << endl;

        return '\n';
    }

// dim = no of features

    vector<int32_t>  calculateMeans_omp(int k, vector<v_float> data_array, long iteration, vector<v_float> &means_array) {
        cout << endl<<"k: " << k << endl;
        cout << "iterations " << iteration << endl;
        // initialize means to random points , means array = dim x K
//        vector<centroid_data> centroids(k);

        vector<v_float > centroids(k,v_float(M,0));
        init_means3(k,data_array,centroids);

//       int32_t *membership = (int32_t *) memalign(AOCL_ALIGNMENT, N * sizeof(int32_t));
//       int32_t *cluster_size = (int32_t *) memalign(AOCL_ALIGNMENT, k * sizeof(int32_t));
//       memset(membership, -1, N *sizeof(int32_t));
//       memset(cluster_size, 0, k * sizeof(int32_t));
       vector<v_float> membership(N, vector<float>(8,-1) );
       vector<int32_t> cluster_size(k,0);


//    vector<v_float> oldmean(k);
           int j,ii,i;
        // for each point
            if(omp_get_thread_num() == 0)
                cout<<"no of threads: "<<omp_get_num_threads()<<endl;

        for (j = 0; j < iteration; j++) {
            std::fill(membership.begin(),membership.end(),v_float(M,-1));
            std::fill(cluster_size.begin(),cluster_size.end(),0);

             #pragma omp parallel for schedule(static) \
             shared(data_array,membership,cluster_size,centroids)\
            private(ii,j,i)
            //for each point
            for (ii = 0; ii < N; ii++) {

                priority_queue<mem_point, std::vector<mem_point>, custom_cmpr> priority_q;
                // v_float dist(k, 0.0);

                auto item_row = data_array[ii];
                // for this point -> calculate distance between each k cmembershipentroid
                for (i = 0; i < k; i++) {
                    float distance_sq_sum = 0.0;
//                    int ii;
                     #pragma omp simd
                    for (int jj = 0; jj < M; jj++)
                    {
                        float s = centroids[i][jj] - data_array[i][jj];
                        distance_sq_sum += s*s;
                        // dist[i] = distance_sq_sum;
                        
                    }  
                        mem_point p;
                        p.distance = distance_sq_sum;
                        p.centroid = i;
                        priority_q.push(p);
                }

                // this point (ii) belongs to p[i].top() centroid
                    int index = priority_q.top().centroid;
                    membership[ii][0] = index;
            }

            // int sum = 0;
#pragma omp parallel for shared(membership,centroids,cluster_size,data_array,M,N,k) default(none) schedule(static)
            for (int jj = 0; jj < k ; ++jj) {
                for (int io = 0; io < N; ++io) {
                    if (membership[io][0] != jj) continue;
                    ++cluster_size[jj];
                    //update centroi

                    for(int yy=0;yy<M;yy++) {
                        centroids[jj][yy] += data_array[io][yy];
                        if(cluster_size[jj] >0)
                            centroids[jj][yy] /= cluster_size[jj];
                    }
                }
            }

//            for (int c = 0; c < k; ++c){
//                if (cluster_size[c] <= 0) continue;
//                for (int j = 0; j < M ;++j) {
//                    centroids[c][j] /= cluster_size[c];
//                }
//            }


            // for(int i=0;i<M;i++)
            // {
            //     for(int j=0;i<k;j++)
            //     {
            //         centroids[k].membership[i] = 
            //     }
            // }
//            sum == N ? cout<<endl<<"iteration ok" : cout<<endl<<"iteration: "<<j<<" "<<endl<<print_arr(centroids,k);
        }
            
//    while( !isClose(oldmean,means_array,k,0.001));

        return cluster_size;
    }
}
