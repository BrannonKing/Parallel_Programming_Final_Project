#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

uint32_t queens;

struct State {
    uint64_t diag_ur;
    uint64_t diag_ul;
    uint32_t rows;
    uint32_t cols;
};

struct Candidate {
    uint8_t row, col;
};

uint32_t accept(const struct State* p, const struct Candidate* c) {
    return c->col == queens - 1;
}
uint32_t reject(const struct State* p, const struct Candidate* c) {
    uint32_t diag_ur = c->row + c->col;
    uint32_t diag_ul = c->row + queens - c->col;
    uint32_t ret = (p->rows & (1 << c->row));
    ret += (p->cols & (1 << c->col));
    ret += (p->diag_ul & (1 << diag_ul));
    ret += (p->diag_ur & (1 << diag_ur));
    return ret;
}

uint64_t backtrack(const struct State* p, const struct Candidate* c) {
    if (reject(p, c)) return 0;
    if (accept(p, c)) return 1;
    struct State pi = *p;
    pi.rows |= (1U << c->row);
    pi.cols |= (1U << c->col);
    pi.diag_ur |= (1ULL << (c->row + c->col));
    pi.diag_ul |= (1ULL << (c->row + queens - c->col));
    uint64_t hits = 0;
    for (int32_t i = 0; i < queens; ++i) {
        struct Candidate ci = {i, c->col + 1};
        hits += backtrack(&pi, &ci);
    }
    return hits;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid parameters. Usage: nqueens <queens>");
        return 1;
    }
    queens = strtoul(argv[1], NULL, 10);
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

    struct State P = {0};
    uint64_t hits = 0;
    for (int32_t i = 0; i < queens; ++i) {
        struct Candidate c = {i, 0};
        hits += backtrack(&P, &c);
    }
    printf("Discovered %llu solutions.\n", (unsigned long long)hits);
    return 0;
}
