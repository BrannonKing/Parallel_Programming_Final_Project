#include "kmean_serial.h"
#include "kmeans_parallel.h"
#include "rdtsc.h"

using namespace std;

char print_arr(int32_t *cluster_map, int k)
{
    for (int i = 0; i < k; i++)
        cout << "no of elements in cluster " << i << ": " << cluster_map[i] << endl;

    return '\n';
}

char print_arr(auto cluster_map, int k)
{
    int i = 0;
    for (auto item : cluster_map)
        cout << "no of elements in cluster " << i++ << ": " << item << endl;

    return '\n';
}

double run_parallel(int k, int iterations, vector<v_float> &data, int num_threads)
{
     cout << endl
             << "Parallel run k: " << k << endl
        << "iterations " << iterations  << endl;
        std::vector<int32_t> cluster_map(k,0);
       std::vector<v_float> centroids(k, v_float(M, 0));
        std::vector<int32_t> membership(N,-1);
    
    auto start = std::chrono::steady_clock::now();       
    
    kmeans_serial::calculateMeans_serial(k, data, iterations,centroids,membership,cluster_map,num_threads);
    
    auto end = std::chrono::steady_clock::now();

    chrono::duration<double> time_span = std::chrono::duration_cast<chrono::duration<double>>(end - start);
    cout << "parallel time: " << time_span.count() * 1e03 << setprecision(9) << " milli seconds.\n";
    auto timespan = time_span.count() * 1e03;

    unsigned long long sun = 0;
    for (int i = 0; i < k; i++)
        sun += cluster_map[i];
    sun == N ? cout << "correctness pass" : cout << "correctness fail, details: " << print_vect(cluster_map);
    print_vect(cluster_map);
    return timespan;
}
int main(int argc, char **argv)
{

    cout<<"argc: "<<argc ;
    //   printf("%llu\n", b-a);

    srand(time(0));
    if (argc != 5)
    {
        fprintf(stderr, "Invalid parameters. Usage: kmeans <clusters> <iterations> <input file>");
        return 2;
    }
    int clusters = atoi(argv[1]);
    if (clusters <= 0)
    {
        fprintf(stderr, "Invalid cluster count. Usage: kmeans <clusters> <iterations> <input file>");
        return 3;
    }

    int iterations = atoi(argv[2]);
    if (iterations <= 0)
    {
        fprintf(stderr, "Invalid iterations count. Usage: kmeans <clusters> <iterations> <input file>");
        return 3;
    }
    int num_threads = atoi(argv[3]);
    if (num_threads <= 0)
    {
        fprintf(stderr, "Invalid iterations count. Usage: kmeans <clusters> <iterations>  <threads> <input file>");
        return 3;
    }

    struct FeatureDefinition fd = load_file(argv[4]);
    N = fd.npoints;
    M = fd.nfeatures;
    // int32_t* membership = (int*) memalign(AOCL_ALIGNMENT,fd.npoints*sizeof (int32_t));
    ios_base::sync_with_stdio(false);

    DataFrame data(N);
    for (long i = 0; i < N; i++)
    {
        vector<float> vec(M, 0);
        for (int j = 0; j < M; j++)
        {
            vec[j] = fd.features[i][j];
        }
        data[i] = vec;
    }

    int k = clusters;
    //int iterations = 5000;

    //   auto serial_timespan = run_serial(k,iterations,data,num);
    auto parallel_timespan = run_parallel(k, iterations, data, num_threads);

    // harcoding serial time for other optimization runs
    // cout << endl
        //  << "Speed up: " << 55358.5 / parallel_timespan << endl;

    return 0;
}
