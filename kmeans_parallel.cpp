#include "kmeans_parallel.h"
#include "omp.h"
#include <climits>
//long N;
//long M;

/****
 *
 * @param k
 * @param means_array
 * @param data_array
 */
namespace kmeansparallel
{

    typedef struct centroid_data centroid_data;
    float calc_distance(v_float p1, float *p2, int M)
    {
        float distance_sq_sum = INFINITY;
        int ii;
        // #pragma omp parallel for schedule(static) simd
        for (ii = 0; ii < M; ii++)
        {
            float s = p1[ii] - p2[ii];
            distance_sq_sum += s * s;
        }

        return sqrtf(distance_sq_sum);
    }

    void init_means3(int k, vector<v_float> data_array, vector<v_float> &centroids)
    {
        //    srand(43);
         static std::random_device seed;
        static std::mt19937 random_number_generator(seed());
        auto start = std::chrono::steady_clock::now();
        std::uniform_int_distribution<size_t> indices(0, N - 1);
        for (auto &cluster : centroids)
        {
            cluster = data_array[indices(random_number_generator)];
        }
        auto end = std::chrono::steady_clock::now();
        chrono::duration<double> time_span = std::chrono::duration_cast<chrono::duration<double>>(end - start);
        cout << "Time of initializing centroid: \t: " << time_span.count() * 1e03 << setprecision(9) << " milli seconds.\n";
        //        return data_array[indices(random_number_generator)];
    }


    struct custom_cmpr
    {
        bool operator()(mem_point p1, mem_point p2)
        {
            // return "true" if "p1" is ordered before "p2", for example:
            if (p1.distance > p2.distance)
                return true;
            return false;
        }
    };
    char print_arr(auto cluster_map)
    {
        int i = 0;
        for (auto item : cluster_map)
            cout << "no of elements in cluster " << i++ << ": " << item << endl;

        return '\n';
    }


   void calculateMeans_omp(
        int k, 
        vector<v_float> const &data_array, 
        long iteration, 
        int num_threads, 
        vector<int32_t> &cluster_size,
        vector<int32_t> &membership,
        vector<v_float> &centroids)
    {
       
        omp_set_num_threads(num_threads);
        #pragma omp parallel
        {}

        init_means3(k, data_array, centroids);

        int j, ii, kk;
        int printed =0;
        for (j = 0; j < iteration; j++)
        {

            std::fill(membership.begin(),membership.end(),-1);
            std::fill(cluster_size.begin(), cluster_size.end(), 0);
            // cout<<"debug1"<<endl;

#pragma omp parallel for schedule(static) \
    shared(data_array, membership, cluster_size, centroids) private(ii, j, kk)
            //for each point
            for (ii = 0; ii < N; ii++)
            {
                if (omp_get_thread_num() == 0 && printed)
                    {printed=1;cout << "no of threads: " << omp_get_num_threads() << endl;}

                // cout<<"debug12";
                int index=-1;
                priority_queue<mem_point, std::vector<mem_point>, custom_cmpr> priority_q;
                // v_float dist(k, 0.0);
                // cout<<"debug2";
                auto item_row = data_array[ii];
                // for this point -> calculate distance between each k cmembershipentroid
                for (kk = 0; kk < k; kk++)
                {
                    float distance_sq_sum = INT_MAX;
                    float s=0.0;
                    //                    int ii;
#pragma omp simd
                    for (int jj = 0; jj < M; jj++)
                    {
                        // cout<<"debug3";
                        s = centroids[kk][jj] - data_array[kk][jj];
                        distance_sq_sum += s * s;
                        // dist[i] = distance_sq_sum;
                    }

                    if(s< distance_sq_sum)
                    {distance_sq_sum = s; index = kk;}

                    // cout<<"debug4";
                    // mem_point p;
                    // p.distance = distance_sq_sum;
                    // p.centroid = kk;
                    // priority_q.push(p);
                }

                // this point (ii) belongs to p[i].top() centroid
                // int index = priority_q.top().centroid;
                // cout<<"debug5";
                membership[ii] = index;
            }

            int yy;
            // int sum = 0;
#pragma omp parallel for private(yy) shared(membership, centroids, cluster_size, data_array, M, N, k) default(none) schedule(static)
            for (int jj = 0; jj < k; ++jj)
            {
                for (int io = 0; io < N; ++io)
                {
                    if (membership[io] != jj)
                        continue;
                    ++cluster_size[jj];
                    //update centroi
                    // #pragma omp simd
                    for (yy = 0; yy < M; yy++)
                    {
                        centroids[jj][yy] += data_array[io][yy];
                    }
                }

                // #pragma  omp simd
                for (yy = 0; yy < M; yy++)
                {

                    if (cluster_size[jj] > 0)
                        centroids[jj][yy] /= cluster_size[jj];
                }
            }
            // print_arr(cluster_size);
        }

        
    }
}
