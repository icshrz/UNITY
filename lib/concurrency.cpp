// concurrency.cpp
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <atomic>
#include <optional>

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

template <typename T>
class BlockingQueue {
public:
    void push(T x) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            q_.push(std::move(x));
        }
        cv_.notify_one();
    }

    bool pop(T& out) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return stop_ || !q_.empty(); });
        if (stop_ && q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
    }

private:
    std::queue<T> q_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
};

class PipelineEngine {
public:
    using StageFn = std::function<void(int)>;

    void set_sample_fn(StageFn fn)  { sample_fn_ = std::move(fn); }
    void set_update_fn(StageFn fn)  { update_fn_ = std::move(fn); }
    void set_forward_fn(StageFn fn) { forward_fn_ = std::move(fn); }
    void set_mailbox_fn(StageFn fn) { mailbox_fn_ = std::move(fn); }

    void start() {
        sample_worker_ = std::thread([&] { run_sample(); });
        update_worker_ = std::thread([&] { run_update(); });
        forward_worker_ = std::thread([&] { run_forward(); });
        mailbox_worker_ = std::thread([&] { run_mailbox(); });
    }

    void submit_batch(int batch_id) {
        sample_q_.push(BatchTask{batch_id, Stage::Sample});
    }

    void stop() {
        sample_q_.stop();
        update_q_.stop();
        forward_q_.stop();
        mailbox_q_.stop();
        if (sample_worker_.joinable()) sample_worker_.join();
        if (update_worker_.joinable()) update_worker_.join();
        if (forward_worker_.joinable()) forward_worker_.join();
        if (mailbox_worker_.joinable()) mailbox_worker_.join();
    }

private:
    void run_sample() {
        BatchTask t;
        while (sample_q_.pop(t)) {
            sample_fn_(t.batch_id);
            update_q_.push(BatchTask{t.batch_id, Stage::Update});
        }
    }

    void run_update() {
        BatchTask t;
        while (update_q_.pop(t)) {
            update_fn_(t.batch_id);
            forward_q_.push(BatchTask{t.batch_id, Stage::Forward});
        }
    }

    void run_forward() {
        BatchTask t;
        while (forward_q_.pop(t)) {
            forward_fn_(t.batch_id);
            mailbox_q_.push(BatchTask{t.batch_id, Stage::Mailbox});
        }
    }

    void run_mailbox() {
        BatchTask t;
        while (mailbox_q_.pop(t)) {
            mailbox_fn_(t.batch_id);
        }
    }

private:
    BlockingQueue<BatchTask> sample_q_, update_q_, forward_q_, mailbox_q_;
    std::thread sample_worker_, update_worker_, forward_worker_, mailbox_worker_;
    StageFn sample_fn_, update_fn_, forward_fn_, mailbox_fn_;
};

} // namespace unity