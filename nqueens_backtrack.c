#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int32_t queens;

typedef struct {
    uint64_t diag_ur;
    uint64_t diag_ul;
    uint32_t rows;
} State;

uint32_t accept(const State* p, int32_t col) {
    return col == queens - 1;
}
uint64_t reject(const State* p, int32_t col, int32_t row) {
    int32_t diag_ur = row + col;
    int32_t diag_ul = row + queens - col;
    uint64_t ret = (p->rows & (1U << row));
    ret += (p->diag_ul & (1ULL << diag_ul));
    ret += (p->diag_ur & (1ULL << diag_ur));
    return ret;
}

void update_state(State* p, int32_t col, int32_t row) {
    p->rows |= (1U << row);
    p->diag_ur |= (1ULL << (row + col));
    p->diag_ul |= (1ULL << (row + queens - col));
}

void reverse_state(State* p, int32_t col, int32_t row) {
    p->rows &= ~(1U << row);
    p->diag_ur &= ~(1ULL << (row + col));
    p->diag_ul &= ~(1ULL << (row + queens - col));
}

uint64_t backtrack_unwind(State* p, int32_t col) {
    uint64_t hits = 0;
    for (int32_t i5 = 0; i5 < queens; ++i5) {
        if (reject(p, col, i5)) continue;
        update_state(p, col, i5);
        for (int32_t i4 = 0; i4 < queens; ++i4) {
            if (reject(p, col + 1, i4)) continue;
            update_state(p, col + 1, i4);
            for (int32_t i3 = 0; i3 < queens; ++i3) {
                if (reject(p, col + 2, i3)) continue;
                update_state(p, col + 2, i3);
                for (int32_t i2 = 0; i2 < queens; ++i2) {
                    if (reject(p, col + 3, i2)) continue;
                    update_state(p, col + 3, i2);
                    for (int32_t i1 = 0; i1 < queens; ++i1) {
                        if (reject(p, col + 4, i1)) continue;
                        update_state(p, col + 4, i1);
                        #pragma omp simd
                        for (int32_t i0 = 0; i0 < queens; ++i0) {
                            if (reject(p, col + 5, i0)) continue;
                            ++hits;
                        }
                        reverse_state(p, col + 4, i1);
                    }
                    reverse_state(p, col + 3, i2);
                }
                reverse_state(p, col + 2, i3);
            }
            reverse_state(p, col + 1, i4);
        }
        reverse_state(p, col, i5);
    }
    return hits;
}

uint64_t backtrack(const State* p, int32_t col, int32_t row) {
    if (reject(p, col, row)) return 0;
    if (accept(p, col)) return 1;
    State pi = *p;
    update_state(&pi, col, row);
    if (queens - col == 7) {
        return backtrack_unwind(&pi, col+1);
    }
    uint64_t hits = 0;
    for (int32_t i = 0; i < queens; ++i) {
        hits += backtrack(&pi, col + 1, i);
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
    #pragma omp parallel for default(none) shared(queens) reduction(+:hits) schedule(dynamic) collapse(3)
    for (int32_t i = 0; i < queens; ++i) {
        for (int32_t j = 0; j < queens; ++j) {
            for (int32_t k = 0; k < queens; ++k) {
                if (j >= i - 1 && j <= i + 1) continue;
                State p = {0};
                update_state(&p, 0, i);
                update_state(&p, 1, j);
                hits += backtrack(&p, 2, k);
            }
        }
    }
    wtime = omp_get_wtime() - wtime;
    printf("Discovered %llu solutions in %f s.\n", (unsigned long long)hits, wtime);
    return 0;
}
