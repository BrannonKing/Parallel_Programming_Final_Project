#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int32_t queens;
uint64_t queens_mask;

typedef struct {
    uint64_t left, down, right;
} Starter;

// this function doesn't utilize a stack; it could be ran on a GPU
uint64_t run_to_end(int32_t end, const Starter* starter) {
    if (end-- > 32) {
        fprintf(stderr, "Invalid board dimension. Maximum size: 32\n");
        exit(10);
    }
    uint64_t stack_left[32];
    uint64_t stack_right[32];
    uint64_t stack_down[32];
    uint64_t stack_available[32];

    stack_left[0] = starter->left;
    stack_down[0] = starter->down;
    stack_right[0] = starter->right;
    stack_available[0] = queens_mask & ~(starter->left | starter->down | starter->right);

    uint64_t hits = 0;
    int32_t depth = 0;

    while(depth >= 0) {
        // we look at what's on the stack for available options
        // if we're at the end we count all the items and pop it
        // if there aren't any we roll back one depth on the stack
        // if we can't go back we're done
        // if there is an option we take it and push a new item onto the stack

        uint64_t available_slots = stack_available[depth];
        if (available_slots == 0) {
            --depth;
            continue;
        }
        if (depth == end) {
            hits += __builtin_popcountll(available_slots);
            --depth;
            continue;
        }

        uint64_t trailing_zeros = __builtin_ctzll(available_slots);
        uint64_t slot = 1ULL << trailing_zeros;
        available_slots ^= slot; // filling that slot so it's no longer available
        stack_available[depth] = available_slots;
        uint64_t left = (stack_left[depth] | slot) << 1U;
        uint64_t right = (stack_right[depth] | slot) >> 1U;
        uint64_t down = (stack_down[depth] | slot);
        ++depth;
        stack_left[depth] = left;
        stack_right[depth] = right;
        stack_down[depth] = down;
        stack_available[depth] = queens_mask & ~(left | down | right);
    }
    return hits;
}

uint64_t backtrack(int32_t row, uint64_t left, uint64_t down, uint64_t right) {
    if (row == queens) return 1;
    uint64_t hits = 0;
    uint64_t available_slots = queens_mask & ~(left | down | right);
    while (available_slots > 0) {
        uint64_t trailing_zeros = __builtin_ctzll(available_slots);
        uint64_t slot = 1ULL << trailing_zeros;
        available_slots ^= slot; // filling that slot so it's no longer available
        hits += backtrack(row + 1, (left | slot) << 1U, down | slot, (right | slot) >> 1);
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
            //hits += backtrack(2, left, down, right);
            Starter starter = { left, down, right };
            hits += run_to_end(queens - 2, &starter);
        }
    }

    // alternate approach in progress:
    // int starter_depth = 4;
    // Starter starters[32*32*32*32];
    // int n = build_starters(starters, starter_depth);
    // #pragma omp target teams distribute parallel for simd map(to:starters[0:n]) reduction(+:hits)
    // for (int i = 0; i < n; ++i)
    //     hits += run_to_end(queens - starter_depth, &starters[i]);

    wtime = omp_get_wtime() - wtime;
    printf("Discovered %llu solutions in %f s.\n", (unsigned long long)hits, wtime);
    return 0;
}
