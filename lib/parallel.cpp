// parallel.cpp
#include <vector>
#include <thread>
#include <future>
#include <functional>
#include <algorithm>
#include <cmath>

namespace unity {

struct ExecutionStats {
    double last_cpu_ms = 1.0;
    double last_gpu_ms = 1.0;
    double last_cpu_load = 0.0;
    double curr_cpu_load = 0.0;
};

class HeteroPartitioner {
public:
    HeteroPartitioner(double pc, double pg)
        : pc_(pc), pg_(pg), ratio_(pg / pc) {}

    void update_ratio(const ExecutionStats& s) {
        double new_ratio = ratio_
            * (s.last_cpu_ms / std::max(1e-6, s.last_gpu_ms))
            * ((1.0 + s.curr_cpu_load) / (1.0 + s.last_cpu_load));

        ratio_ = std::clamp(new_ratio, min_ratio_, max_ratio_);
    }

    template <typename T>
    std::pair<std::vector<T>, std::vector<T>> split_rows(const std::vector<T>& rows) const {
        size_t gpu_rows = static_cast<size_t>(
            std::round(rows.size() * (ratio_ / (1.0 + ratio_))));
        gpu_rows = std::min(gpu_rows, rows.size());

        std::vector<T> gpu_part(rows.begin(), rows.begin() + gpu_rows);
        std::vector<T> cpu_part(rows.begin() + gpu_rows, rows.end());
        return {gpu_part, cpu_part};
    }

    double ratio() const { return ratio_; }

private:
    double pc_;
    double pg_;
    double ratio_;
    double min_ratio_ = 0.1;
    double max_ratio_ = 32.0;
};

class ParallelExecutor {
public:
    template <typename RowT, typename CpuFn, typename GpuFn>
    void run_disjoint(
        const std::vector<RowT>& rows,
        HeteroPartitioner& partitioner,
        const ExecutionStats& stats,
        CpuFn cpu_fn,
        GpuFn gpu_fn)
    {
        partitioner.update_ratio(stats);
        auto [gpu_rows, cpu_rows] = partitioner.split_rows(rows);

        auto cpu_future = std::async(std::launch::async, [&] {
            cpu_fn(cpu_rows);
        });

        gpu_fn(gpu_rows); // usually enqueue CUDA kernels

        cpu_future.get();
    }
};

} // namespace unity