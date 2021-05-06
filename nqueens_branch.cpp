#include <omp.h>
//#include <boost/lockfree/queue.hpp>
#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <queue>
#include <thread>

template <typename TWork, bool acceptedHaveNoChildren=true>
class SearchSpaceBase {
protected:
    [[nodiscard]] virtual bool accept(const TWork&) const = 0;
    [[nodiscard]] virtual bool reject(const TWork&) const = 0;
    [[nodiscard]] virtual bool accepted(const TWork&) = 0;
    virtual bool dequeue_work(TWork&) = 0;
    virtual void update_work(TWork&) const = 0;
    virtual void produce_children(TWork) = 0;  // TODO: return a range instead
public:
    virtual bool enqueue_work(TWork) = 0;
    bool run() {
        TWork work;
        auto still_working = true;
        #pragma omp parallel default(none) private(work) shared(still_working)
        {
            while (still_working) {
                auto success = dequeue_work(work);
                if (!success) {
                    break; // this isn't perfect; if you hit a bottleneck your threads will exit
                }
                if (reject(work)) continue;
                if (accept(work)) {
                    still_working = !accepted(work);
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
    std::queue<TWork> _queue;
protected:
    bool dequeue_work(TWork& work) override {
        bool success;
        #pragma omp critical
        if (!_queue.empty()) {
            work = _queue.front();
            _queue.pop();
            success = true;
        }
        else success = false;
        return success;
    }
public:
    bool enqueue_work(TWork work) override {
        #pragma omp critical
        _queue.push(work);
        return true;
    }
};

struct Work {
    uint64_t diag_ur;
    uint64_t diag_ul;
    uint32_t rows;
    uint32_t row, col;
};

uint32_t queens;

class NQueensBacktracker: public BacktrackerBase<Work> {
    uint64_t hits = 0;
protected:
    void update_work(Work& work) const override {
        work.rows |= (1U << work.row);
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
        ret += (work.diag_ul & (1ULL << diag_ul));
        ret += (work.diag_ur & (1ULL << diag_ur));
        return bool(ret);
    }

    [[nodiscard]] bool accepted(const Work& work) override {
        #pragma omp atomic
        hits++;
        return false;
    }

public:
    [[nodiscard]] uint64_t hit_count() const { return hits; }
};

template <typename T>
class StallingBacktracker: public BacktrackerBase<T> {
    int _stall, _depth;
protected:
    void update_work(T& work) const override {
        //std::this_thread::sleep_for (std::chrono::microseconds(_stall));
#pragma omp atomic
        ++_works;
        ++work;
    }

    void produce_children(T work) override {
        for (int32_t i = 0; i < queens; ++i) {
            BacktrackerBase<T>::enqueue_work(work);
        }
    }

    [[nodiscard]] bool accept(const T& work) const override {
        return work == _depth;
    }

    [[nodiscard]] bool reject(const T& work) const override {
        return false;
    }

    [[nodiscard]] bool accepted(const T& work) override {
        return false;
    }

public:
    StallingBacktracker(int stall, T depth) : _stall(stall), _depth(depth), _works(0) {}
    mutable int _works;
};

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Invalid parameters. Usage: nqueens <queens>" << std::endl;
        return 1;
    }
    queens = std::strtoul(argv[1], nullptr, 10);
    if (queens <= 0 || queens > 32) {
        std::cerr << "Invalid queens count. Number expected from 1 to 32." << std::endl;
        return 2;
    }

//    NQueensBacktracker tracker;
//    Work work = {0};
//    for (int32_t i = 0; i < queens; ++i) {
//        work.row = i;
//        tracker.enqueue_work(work);
//    }
//    tracker.run();

    for (int i = 1; i <= 64; i += 1) {
        double wtime = omp_get_wtime();
        omp_set_num_threads(i);
        StallingBacktracker<__int128> tracker(i, 6);
        tracker.enqueue_work(0);
        tracker.run();
        wtime = omp_get_wtime() - wtime;
        //printf("Discovered %llu solutions in %f s.\n", (unsigned long long)tracker.hit_count(), wtime);
        std::cout << i << " in " << wtime << " s. for " << tracker._works << " gives " << (wtime / tracker._works) << std::endl;
        //break;
    }
    return 0;
}
