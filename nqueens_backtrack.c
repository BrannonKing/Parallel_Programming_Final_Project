#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int32_t queens;
uint64_t queens_mask;

uint64_t backtrack(int32_t row, uint64_t left, uint64_t down, uint64_t right) {
    if (row == queens) return 1;
    uint64_t hits = 0;
    uint64_t available = queens_mask & ~(left | down | right);
    while (available > 0) {
        uint64_t trailing_zeros = __builtin_ctzll(available);
        uint64_t flag = 1ULL << trailing_zeros;
        available ^= flag;
        hits += backtrack(row + 1, (left | flag) << 1U, down | flag, (right | flag) >> 1);
    }
    return hits;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid parameters. Usage: nqueens <queens>");
        return 1;
    }
    queens = atoi(argv[1]);
    if (queens <= 0 || queens > 32) {
        fprintf(stderr, "Invalid queens count. Number expected from 1 to 32.");
        return 2;
    }
    queens_mask = (1ULL << queens) - 1ULL;

//    given state P and partial candidate c:
//    procedure backtrack(c):
//      if reject(P, c) then return
//      if accept(P, c) then output(P, c)
//      s ← first(P, c)
//      while s ≠ NULL do
//          backtrack(s)
//          s ← next(P, s)

    double wtime = omp_get_wtime();
    uint64_t hits = 0;
    #pragma omp parallel for default(none) shared(queens, queens_mask) reduction(+:hits) schedule(dynamic) collapse(2)
    for (int32_t i = 0; i < queens; ++i) {
        for (int32_t j = 0; j < queens; ++j) {
            if (j >= i - 1 && j <= i + 1) continue;
            uint64_t down = (1ULL << i) | (1ULL << j);
            uint64_t left = queens_mask & ((1ULL << (i+2)) | (1ULL << (j+1)));
            uint64_t right = queens_mask & ((1ULL << (i-2)) | (1ULL << (j-1)));
            hits += backtrack(2, left, down, right);
        }
    }
    wtime = omp_get_wtime() - wtime;
    printf("Discovered %llu solutions in %f s.\n", (unsigned long long)hits, wtime);
    return 0;
}
