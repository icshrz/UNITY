#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#include <pybind11/pybind11.h>

namespace unity {

enum class Stage {
    Sample,
    Update,
    Forward,
    Mailbox,
    Done
};

struct BatchTask {
    int batch_id = -1;
    Stage stage = Stage::Sample;
};

class DependencyTracker {
public:
    void mark_update_started(int batch_id) { last_update_started_ = batch_id; }
    void mark_update_finished(int batch_id) { last_update_finished_ = batch_id; }
    void mark_forward_started(int batch_id) { last_forward_started_ = batch_id; }
    void mark_forward_finished(int batch_id) { last_forward_finished_ = batch_id; }

    bool can_launch_forward(int batch_id) const { return batch_id <= last_update_finished_ + 1; }
    bool can_launch_next_sample(int next_batch_id) const { return next_batch_id <= last_forward_started_ + 2; }

private:
    int last_update_started_ = -1;
    int last_update_finished_ = -1;
    int last_forward_started_ = -1;
    int last_forward_finished_ = -1;
};

class PipelineEngine {
public:
    using StageFn = std::function<void(int)>;

    PipelineEngine() = default;
    ~PipelineEngine() { stop(); }

    void set_sample_fn(StageFn fn) { sample_fn_ = std::move(fn); }
    void set_update_fn(StageFn fn) { update_fn_ = std::move(fn); }
    void set_forward_fn(StageFn fn) { forward_fn_ = std::move(fn); }
    void set_mailbox_fn(StageFn fn) { mailbox_fn_ = std::move(fn); }

    void start() {}
    void submit_batch(int batch_id) {
        if (sample_fn_) sample_fn_(batch_id);
        if (update_fn_) update_fn_(batch_id);
        if (forward_fn_) forward_fn_(batch_id);
        if (mailbox_fn_) mailbox_fn_(batch_id);
    }
    void stop() {}

private:
    StageFn sample_fn_;
    StageFn update_fn_;
    StageFn forward_fn_;
    StageFn mailbox_fn_;
};

inline void bind_concurrency(pybind11::module_& m) {
    namespace py = pybind11;

    py::enum_<Stage>(m, "Stage")
        .value("Sample", Stage::Sample)
        .value("Update", Stage::Update)
        .value("Forward", Stage::Forward)
        .value("Mailbox", Stage::Mailbox)
        .value("Done", Stage::Done);

    py::class_<BatchTask>(m, "BatchTask")
        .def(py::init<>())
        .def_readwrite("batch_id", &BatchTask::batch_id)
        .def_readwrite("stage", &BatchTask::stage);

    py::class_<DependencyTracker>(m, "DependencyTracker")
        .def(py::init<>())
        .def("mark_update_started", &DependencyTracker::mark_update_started)
        .def("mark_update_finished", &DependencyTracker::mark_update_finished)
        .def("mark_forward_started", &DependencyTracker::mark_forward_started)
        .def("mark_forward_finished", &DependencyTracker::mark_forward_finished)
        .def("can_launch_forward", &DependencyTracker::can_launch_forward)
        .def("can_launch_next_sample", &DependencyTracker::can_launch_next_sample);

    py::class_<PipelineEngine>(m, "PipelineEngine")
        .def(py::init<>())
        .def("start", &PipelineEngine::start)
        .def("submit_batch", &PipelineEngine::submit_batch)
        .def("stop", &PipelineEngine::stop);
}

} // namespace unity