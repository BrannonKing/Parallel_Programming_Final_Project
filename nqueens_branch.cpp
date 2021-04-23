#include <omp.h>
//#include <boost/lockfree/queue.hpp>
#include <atomic>
#include <functional>
//#include <iostream>
#include <queue>
#include <vector>

template <typename TWork, bool acceptedHaveNoChildren=true>
class SearchSpaceBase {
protected:
    [[nodiscard]] virtual bool accept(const TWork&) const = 0;
    [[nodiscard]] virtual bool reject(const TWork&) const = 0;
    virtual bool dequeue_work(TWork&) = 0;
    virtual void update_work(TWork&) const = 0;
    virtual void produce_children(TWork) = 0;  // TODO: return a range instead
    virtual void initialize_queues(int threads) = 0;
public:
    virtual bool enqueue_work(TWork) = 0;
    bool run(std::function<void(const TWork&)> accepted) {
        TWork work;
        std::vector<bool> done_signals;
        #pragma omp parallel default(none) private(work) shared(accepted, done_signals)
        {
            #pragma omp single
            {
                initialize_queues(omp_get_num_threads());
                done_signals.resize(omp_get_num_threads());
            }
            while (true) {
                auto success = dequeue_work(work);
                if (!success) {
                    done_signals[omp_get_thread_num()] = true;
                    auto all_done = true;
                    for (auto signal: done_signals)
                        if (!signal) { all_done = false; break; }
                    if (all_done)
                        break;
                    continue;
                }
                done_signals[omp_get_thread_num()] = false;
                if (reject(work)) continue;
                if (accept(work)) {
                    accepted(work);
                    if constexpr(acceptedHaveNoChildren) continue;
                }
                update_work(work);
                produce_children(work);
            }
        }
        return true;
    }
};

template <typename TWork, bool acceptedHaveNoChildren=true>
class BacktrackerBase : public SearchSpaceBase<TWork, acceptedHaveNoChildren> {
    //std::queue<TWork> _queue;
    //boost::lockfree::queue<TWork> _queue;
protected:
    std::vector<std::queue<TWork>> _queues;
    std::vector<omp_lock_t> _locks;

    void initialize_queues(int threads) override {
        _queues.resize(threads);
        _locks.resize(threads);
    }

    bool dequeue_work(TWork& work) override {
        auto rank = omp_get_thread_num();
        omp_set_lock(&_locks[rank]);
        bool success = false;
//        #pragma omp critical
        if (!_queues[rank].empty()) {
            work = _queues[rank].front();
            _queues[rank].pop();
            success = true;
        }
        omp_unset_lock(&_locks[rank]);
        return success;
    }
public:
    BacktrackerBase(): _queues(1), _locks(1) {}

    bool enqueue_work(TWork work) override {
        //#pragma omp critical
        auto rank = omp_get_thread_num();
        omp_set_lock(&_locks[rank]);
        _queues[rank].push(work);
        omp_unset_lock(&_locks[rank]);
        return true;
    }
};

struct Work {
    uint64_t diag_ur;
    uint64_t diag_ul;
    uint32_t rows;
    uint32_t cols;
    uint32_t row, col;
};

uint32_t queens;

class NQueensBacktracker: public BacktrackerBase<Work> {
protected:
    void update_work(Work& work) const override {
        work.rows |= (1U << work.row);
        work.cols |= (1U << work.col);
        work.diag_ur |= (1ULL << (work.row + work.col));
        work.diag_ul |= (1ULL << (work.row + queens - work.col));
        ++work.col;
    }

    void produce_children(Work work) override {
        for (int32_t i = 0; i < queens; ++i) {
            work.row = i;
            enqueue_work(work);
        }
    }

    [[nodiscard]] bool accept(const Work& work) const override {
        return work.col == queens - 1;
    }
    [[nodiscard]] bool reject(const Work& work) const override {
        uint32_t diag_ur = work.row + work.col;
        uint32_t diag_ul = work.row + queens - work.col;
        uint32_t ret = (work.rows & (1U << work.row));
        ret += (work.cols & (1U << work.col));
        ret += (work.diag_ul & (1ULL << diag_ul));
        ret += (work.diag_ur & (1ULL << diag_ur));
        return bool(ret);
    }
public:
    bool enqueue_work(Work work) override {
        //#pragma omp critical
        auto idx = work.row % _locks.size();
        omp_set_lock(&_locks[idx]);
        _queues[idx].push(work);
        omp_unset_lock(&_locks[idx]);
        return true;
    }
};

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid parameters. Usage: nqueens <queens>");
        return 1;
    }
    queens = std::strtoul(argv[1], nullptr, 10);
    if (queens <= 0 || queens > 32) {
        fprintf(stderr, "Invalid queens count. Number expected from 1 to 32.");
        return 2;
    }

    double wtime = omp_get_wtime();
    std::atomic<uint64_t> hits(0);

    NQueensBacktracker tracker;
    Work work = {0};
    for (int32_t i = 0; i < queens; ++i) {
        work.row = i;
        tracker.enqueue_work(work);
    }

    tracker.run([&hits](const Work& work){
        ++hits;
    });

    wtime = omp_get_wtime() - wtime;
    printf("Discovered %llu solutions in %f s.\n", (unsigned long long)hits, wtime);
    return 0;
}
