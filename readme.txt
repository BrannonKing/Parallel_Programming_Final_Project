Compilation instructions:
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..

Comment out the bottom three lines of CMakeLists.txt if you cannot support NVPTX builds.
The random seed in kmeans is fixed. You may want to modify the srand call.

Execution instructions:

For k-means with 50 clusters and 150 iteration maximum on the OpenDwarfs test file using core-count threads:
./kmeans 50 150 ../../OpenDwarfs/test/dense-linear-algebra/kmeans/819200.txt

For n-queens with board size 16x16 using all available threads:
./nqueens 16

To limit the thread-count to 8:
OMP_NUM_THREADS=8 ./nqueens 16

To analyize the cache usage, use a command similar to this:
perf stat -e cache-misses,cache-references,l2_lines_in.all,l1d.replacement,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.512b_packed_double,fp_arith_inst_retired.512b_packed_single,l1d.replacement,branch-misses,branch ./kmeans 5 5 ~/project/OpenDwarfs-master/test/dense-linear-algebra/kmeans/819200.txt

Code in this repository is written by Brannon King with feedback on k-means from Ayush Chaturvedi.
Exceptions to this are Jeff Somers N-queens implementation and the file loading code taken from OpenDwarfs.