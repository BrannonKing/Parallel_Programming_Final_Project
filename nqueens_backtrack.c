#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_DEPTH 32
int32_t queens;
uint64_t queens_mask;

uint64_t build_starters_internal(uint64_t *starters, uint64_t position, int32_t rows, uint64_t left, uint64_t down, uint64_t right) {
    if (rows == 0) {
        starters[position] = left;
        starters[position+1] = down;
        starters[position+2] = right;
        return position + 3;
    }
    uint64_t available_slots = queens_mask & ~(left | down | right);
    while (available_slots > 0) {
        uint64_t trailing_zeros = __builtin_ctzll(available_slots);
        uint64_t slot = 1ULL << trailing_zeros;
        available_slots ^= slot; // filling that slot so it's no longer available
        position = build_starters_internal(starters, position, rows - 1, (left | slot) << 1U, down | slot, (right | slot) >> 1);
    }
    return position;
}

uint64_t build_starters(uint64_t* starters, int32_t starter_depth) {
    return build_starters_internal(starters, 0, starter_depth, 0, 0, 0);
}

// this function doesn't utilize a stack; it could be ran on a GPU
uint64_t run_to_end(uint64_t* starters, int32_t starter_depth, uint64_t starter_count) {

    const int32_t end = queens - starter_depth - 1;
    const uint64_t mask = queens_mask;

    #pragma omp target teams distribute parallel for simd default(none) \
        shared(starters, end, starter_count, mask) map(tofrom:starters[0:starter_count])
    for (uint64_t i = 0; i < starter_count; i += 3) {
        uint64_t stack_left[MAX_DEPTH];
        uint64_t stack_right[MAX_DEPTH];
        uint64_t stack_down[MAX_DEPTH];
        uint64_t stack_available[MAX_DEPTH];

        uint64_t left = starters[i];
        uint64_t down = starters[i+1];
        uint64_t right = starters[i+2];

        stack_left[0] = left;
        stack_down[0] = down;
        stack_right[0] = right;
        stack_available[0] = mask & ~(left | down | right);

        uint64_t hits = 0;
        int32_t depth = 0;

        while(depth >= 0) {
            // we look at what's on the stack for available options
            // if we're at the end we count all the items and pop it
            // if there aren't any we roll back one depth on the stack
            // if we can't go back we're done
            // if there is an option we take it and push a new item onto the stack

            uint64_t available_slots = stack_available[depth];
            if (depth == end) {
                hits += __builtin_popcountll(available_slots);
                available_slots = 0;
            }
            if (available_slots == 0) {
                --depth;
            }
            else {
                uint64_t trailing_zeros = __builtin_ctzll(available_slots);
                uint64_t slot = 1ULL << trailing_zeros;
                available_slots ^= slot; // filling that slot so it's no longer available
                stack_available[depth] = available_slots;
                left = (stack_left[depth] | slot) << 1U;
                right = (stack_right[depth] | slot) >> 1U;
                down = (stack_down[depth] | slot);
                ++depth;
                stack_left[depth] = left;
                stack_right[depth] = right;
                stack_down[depth] = down;
                stack_available[depth] = mask & ~(left | down | right);
            }
        }
        starters[i] = hits;
    }
    uint64_t result = 0;
    for (uint64_t i = 0; i < starter_count; i += 3) {
        result += starters[i];
    }
    return result;
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

    double wtime = omp_get_wtime();
    const int starter_depth = 4;
    uint64_t* starters = (uint64_t*) malloc(MAX_DEPTH*MAX_DEPTH*MAX_DEPTH*MAX_DEPTH*3*sizeof(uint64_t));
    uint64_t n = build_starters(starters, starter_depth);
    uint64_t hits = run_to_end(starters, starter_depth, n);
    wtime = omp_get_wtime() - wtime;
    printf("Discovered %llu solutions in %f s.\n", (unsigned long long)hits, wtime);
    return 0;
}