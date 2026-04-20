#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace unity {

struct ExecutionStats {
    double last_cpu_ms = 1.0;
    double last_gpu_ms = 1.0;
    double last_cpu_load = 0.0;
    double curr_cpu_load = 0.0;
};

enum class ParallelOp {
    Updater,
    MLP,
    Norm,
    MailboxUpdate
};

class HeteroPartitioner {
public:
    HeteroPartitioner(double pc = 1.0, double pg = 1.0)
        : pc_(pc), pg_(pg), ratio_(pg / pc) {}

    void update_ratio(const ExecutionStats& s) {
        double new_ratio = ratio_
            * (s.last_cpu_ms / std::max(1e-6, s.last_gpu_ms))
            * ((1.0 + s.curr_cpu_load) / (1.0 + s.last_cpu_load));
        ratio_ = std::clamp(new_ratio, min_ratio_, max_ratio_);
    }

    double ratio() const { return ratio_; }

    std::pair<size_t, size_t> split_count(size_t total_rows) const {
        size_t gpu_rows = static_cast<size_t>(
            std::round(total_rows * (ratio_ / (1.0 + ratio_))));
        gpu_rows = std::min(gpu_rows, total_rows);
        return {gpu_rows, total_rows - gpu_rows};
    }

private:
    double pc_;
    double pg_;
    double ratio_;
    double min_ratio_ = 0.1;
    double max_ratio_ = 32.0;
};

class CpuLoadSampler {
public:
    double sample() const { return 0.0; }
};

inline void bind_parallel(pybind11::module_& m) {
    namespace py = pybind11;

    py::class_<ExecutionStats>(m, "ExecutionStats")
        .def(py::init<>())
        .def_readwrite("last_cpu_ms", &ExecutionStats::last_cpu_ms)
        .def_readwrite("last_gpu_ms", &ExecutionStats::last_gpu_ms)
        .def_readwrite("last_cpu_load", &ExecutionStats::last_cpu_load)
        .def_readwrite("curr_cpu_load", &ExecutionStats::curr_cpu_load);

    py::enum_<ParallelOp>(m, "ParallelOp")
        .value("Updater", ParallelOp::Updater)
        .value("MLP", ParallelOp::MLP)
        .value("Norm", ParallelOp::Norm)
        .value("MailboxUpdate", ParallelOp::MailboxUpdate);

    py::class_<HeteroPartitioner>(m, "HeteroPartitioner")
        .def(py::init<double, double>(), py::arg("pc") = 1.0, py::arg("pg") = 1.0)
        .def("update_ratio", &HeteroPartitioner::update_ratio)
        .def("ratio", &HeteroPartitioner::ratio)
        .def("split_count", &HeteroPartitioner::split_count);

    py::class_<CpuLoadSampler>(m, "CpuLoadSampler")
        .def(py::init<>())
        .def("sample", &CpuLoadSampler::sample);
}

} // namespace unity