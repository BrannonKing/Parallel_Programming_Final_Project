#include "kmean_serial.h"
#include "kmeans_parallel.h"

using namespace std;

char print_arr(int32_t* cluster_map, int k)
{
    for (int i = 0; i < k; i++)
        cout << "no of elements in cluster " << i << ": " << cluster_map[i] << endl;

    return '\n';
}

char print_arr(auto cluster_map, int k)
{
    int i=0;
    for (auto item : cluster_map)
        cout << "no of elements in cluster " << i++ << ": " << item << endl;

    return '\n';
}
double run_serial(int k, int iterations,vector<v_float>& data, vector<v_float>& means_arr)
{
    auto start = std::chrono::high_resolution_clock::now();
    int32_t * cluster_map = kmeans_serial::calculateMeans_serial(k, data, iterations, means_arr);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << "serial time: " << time_span.count()*1e03<<setprecision(9) << " milli seconds.\n";
    auto serial_timespan = time_span.count()*1e03;
    long sun =0;
    for(int i=0 ;i < k;i++)
        sun+=cluster_map[i];
    sun == N ? cout << "correctness pass" : cout << "correctness fail, details: " << print_arr(cluster_map,k);

    return serial_timespan;
}


double run_parallel(int k, int iterations,vector<v_float>& data, vector<v_float>& means_arr)
{
    auto start = chrono::high_resolution_clock::now();
    auto cluster_map = kmeansparallel::calculateMeans_omp(k, data, iterations, means_arr);
    auto end =chrono::high_resolution_clock::now();
    chrono::duration<double> time_span = std::chrono::duration_cast<chrono::duration<double>>(end - start);
    cout << "parallel time: " << time_span.count() * 1e03 << setprecision(9) << " milli seconds.\n";
    auto serial_timespan = time_span.count() * 1e03;

    unsigned long long sun = 0;
    for (int i = 0; i < k; i++)
        sun += cluster_map[i];
    sun == N ? cout<< "correctness pass" : cout << "correctness fail, details: " << print_vect(cluster_map);
    return serial_timespan;
}
int main(int argc, char **argv) {
    srand(time(0));
    if (argc != 4) {
        fprintf(stderr, "Invalid parameters. Usage: kmeans <clusters> <iterations> <input file>");
        return 2;
    }
    int clusters = atoi(argv[1]);
    if (clusters <= 0) {
        fprintf(stderr, "Invalid cluster count. Usage: kmeans <clusters> <iterations> <input file>");
        return 3;
    }

    int iterations = atoi(argv[2]);
    if (iterations <= 0) {
        fprintf(stderr, "Invalid iterations count. Usage: kmeans <clusters> <iterations> <input file>");
        return 3;
    }

    struct FeatureDefinition fd = load_file(argv[3]);
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
    int k =clusters;
    //int iterations = 5000;
    vector<v_float> means_arr(k, v_float(M,0));
    auto serial_timespan = run_serial(k,iterations,data,means_arr);
    auto parallel_timespan = run_parallel(k,iterations,data,means_arr);


    cout<<endl<<"Speed up: "<<serial_timespan/parallel_timespan<<endl;

    return 0;
}
