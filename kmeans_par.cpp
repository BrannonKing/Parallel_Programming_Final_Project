#include "kmean_serial.h"
#include "omp.h"
//long N;
//long M;

namespace kmeans_serial
{
    /****
 *
 * @param k
 * @param means_array
 * @param data_array
 */
    void init_means(short k, vector<vector<float>> &means_array, vector<v_float> data_array)
    {
        //    srand(43);
        static std::random_device seed;
        static std::mt19937 random_number_generator(seed());
        auto start = std::chrono::steady_clock::now();
        std::uniform_int_distribution<size_t> indices(0, N - 1);
        for (auto &cluster : means_array)
        {
            cluster = data_array[indices(random_number_generator)];
        }
        auto end = std::chrono::steady_clock::now();
        chrono::duration<double> time_span = std::chrono::duration_cast<chrono::duration<double>>(end - start);
        cout << "Time of initializing centroid: \t: " << time_span.count() * 1e03 << setprecision(9) << " milli seconds.\n";
    }
    int u = 0;
    /***
 * Update the k centroids of M feature
 * @param N : no of elements in this cluster
 * @param means_row: the centroid ?
 * @param data_row: the single point ?
 */
    void update_mean(long N, vector<float> &means_row, v_float data_row)
    {
        auto start = std::chrono::steady_clock::now();
        if (N < 0)
        {
            cout << "N is less than zero error";
            exit(5);
        }
        // update mean for each feature
        #pragma omp simd
        for (int i = 0; i < M; i++)
        {
            float m = means_row[i];
            m *= (N - 1);
            m += data_row[i];
            m = m / N;
            means_row[i] = m;
        }

        auto end = std::chrono::steady_clock::now();
        chrono::duration<double> time_span = std::chrono::duration_cast<chrono::duration<double>>(end - start);
        if (u == 0)
            cout << "Time of updating centroid: \t: " << time_span.count() * 1e03 << setprecision(9) << " milli seconds.\n";
        u = 1;
    }

    void update_old_cluster_mean(int old_size, vector<float> &means_row, v_float data_row)
    {
        if (N < 0)
        {
            cout << "N is less than zero error";
            exit(5);
        }
        // update mean for each feature
        for (int i = 0; i < M; i++)
        {
            float m = means_row[i];
            m *= old_size;
            m -= data_row[i];
            m = m / (old_size - 1);
            means_row[i] = m;
        }
    }

    int s = 0;
    /***
 * For each point having M dimensions/features
 * Calculate
 * @param means_array - Mxk array of centroids
 * @param data_row - distance vector rows
 * @param k
 * @return
 */
    int classify(v_float distances, int k)
    {
        auto start = std::chrono::steady_clock::now();

        float min = INFINITY;
        long index = -1;

        // for each centroid
        for (int i = 0; i < distances.size(); i++)
        {

            // calculate distance
            float dist_temp = distances[i];
            if (dist_temp < min)
            {
                min = dist_temp;
                index = i;
            }
        }

        auto end = std::chrono::steady_clock::now();
        chrono::duration<double> time_span = std::chrono::duration_cast<chrono::duration<double>>(end - start);
        if (s == 0)
            cout << "Time of classifying single point: \t: " << time_span.count() * 1e03 << setprecision(9) << " milli seconds.\n";
        s = 1;
        return index;
    }

    int isClose(vector<v_float> &c1, vector<v_float> &c2, float tolerance, int k)
    {
        for (int i = 0; i < k; ++i)
        {
            for (int j = i; j < M; j++)
                if (abs(c1[i][j] - c2[i][j]) > tolerance)
                    return 0;
        }
        return 1;
    }

    // dim = no of features
    void calculateMeans_serial(
        int k,
        vector<v_float> const &data_array,
        long iteration,
        vector<v_float> &means_array,
        vector<int32_t>& membership,
        vector<int32_t>& cluster_size, int num_threads)
    {

        init_means(k, means_array, data_array);
        // omp_set_num_threads(num_threads);
        int ii,i;
        // for each point
        for (int j = 0; j < iteration; j++)
        {
            std::fill(membership.begin(), membership.end(), -1);
            std::fill(cluster_size.begin(), cluster_size.end(), 0);
            int ii,i,jj;
            #pragma omp parallel for schedule(static)\
            shared(data_array, membership, cluster_size, means_array) private(ii, i,jj)
            for ( ii = 0; ii < N; ii++)
            {
                v_float dist(k, 0);

                auto item_row = data_array[ii];

                // classify this point return the value from 0->k
                for ( i = 0; i < k; i++)
                {
                    // calculate distance between each k centroid
                    float distance_sq_sum=0.0;
                    #pragma omp simd
                     for ( jj = 0; jj < M; jj++)
                         distance_sq_sum += ((item_row[jj] - means_array[i][jj])* (item_row[jj] - means_array[i][jj]));
                    //  d = calc_distance(item_row, means_array[i], M);
                    dist[i] = distance_sq_sum;
                }

                int index = classify(dist, k);
                membership[ii] = index;
                #pragma omp critical
                {
                long csize = ++cluster_size[index];
update_mean(csize, means_array[index], item_row);}
            }
        }
        //    while( !isClose(oldmean,means_array,k,0.001));

      
    }
}