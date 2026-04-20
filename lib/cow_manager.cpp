#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/*
 * UNITY-style Copy-on-Write manager
 *
 * Core idea:
 * 1. Each node ID has a "currently visible version".
 * 2. Updates within one minibatch are ordered by timestamp.
 * 3. An in-place overwrite is allowed only when:
 *      (a) this occurrence is the latest timestamp for the node in the minibatch
 *      (b) the previous visible version is no longer live (ref_count == 0)
 * 4. Otherwise, write to a temporary COW slot and publish it as the new visible version.
 * 5. Consumers always read through the batch-local visible mapping.
 *
 * This file is intended as a system-side implementation skeleton, not a drop-in
 * replacement for CUDA kernels. The actual tensor copy / kernel launch should be
 * plugged into the callback interfaces below.
 */

namespace unity {

using NodeID = uint32_t;
using Timestamp = int64_t;
using Address = uint64_t;

enum class WriteMode {
    InPlace,
    CopyOnWrite
};

struct VersionHandle {
    Address addr = 0;
    Timestamp ts = -1;
    bool from_cow = false;
    uint64_t generation = 0;
};

struct UpdateOccurrence {
    NodeID nid = 0;
    Timestamp ts = -1;
    size_t occurrence_idx = 0;
    bool is_latest_occurrence = false;
};

struct CowSlot {
    Address addr = 0;
    size_t bytes = 0;
    bool in_use = false;
};

class CowAllocator {
public:
    explicit CowAllocator(std::vector<CowSlot> slots) : slots_(std::move(slots)) {}

    std::optional<CowSlot> allocate(size_t bytes) {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& s : slots_) {
            if (!s.in_use && s.bytes >= bytes) {
                s.in_use = true;
                return s;
            }
        }
        return std::nullopt;
    }

    bool release(Address addr) {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& s : slots_) {
            if (s.addr == addr) {
                s.in_use = false;
                return true;
            }
        }
        return false;
    }

private:
    std::mutex mu_;
    std::vector<CowSlot> slots_;
};

struct NodeVersionState {
    Address base_addr = 0;                 // original slot in node-memory region
    VersionHandle visible;                 // latest visible version in this minibatch
    std::atomic<int32_t> ref_count{0};     // live readers of the previously visible state
    Timestamp latest_seen_ts = -1;         // max timestamp encountered in current minibatch
    bool initialized = false;
};

struct PublishedVersion {
    NodeID nid = 0;
    VersionHandle vh;
};

class BatchVisibleTable {
public:
    void init_node(NodeID nid, Address base_addr, Timestamp init_ts = -1) {
        std::unique_lock<std::shared_mutex> lock(mu_);
        auto& s = table_[nid];
        s.base_addr = base_addr;
        s.visible = VersionHandle{base_addr, init_ts, false, 0};
        s.latest_seen_ts = init_ts;
        s.initialized = true;
        s.ref_count.store(0, std::memory_order_relaxed);
    }

    void record_latest_timestamp(NodeID nid, Timestamp ts) {
        std::unique_lock<std::shared_mutex> lock(mu_);
        auto it = table_.find(nid);
        if (it == table_.end()) return;
        it->second.latest_seen_ts = std::max(it->second.latest_seen_ts, ts);
    }

    std::optional<VersionHandle> visible(NodeID nid) const {
        std::shared_lock<std::shared_mutex> lock(mu_);
        auto it = table_.find(nid);
        if (it == table_.end() || !it->second.initialized) return std::nullopt;
        return it->second.visible;
    }

    std::optional<NodeVersionState> snapshot(NodeID nid) const {
        std::shared_lock<std::shared_mutex> lock(mu_);
        auto it = table_.find(nid);
        if (it == table_.end() || !it->second.initialized) return std::nullopt;
        return NodeVersionState{
            it->second.base_addr,
            it->second.visible,
            it->second.ref_count.load(std::memory_order_acquire),
            it->second.latest_seen_ts,
            true
        };
    }

    void inc_ref(NodeID nid) {
        std::shared_lock<std::shared_mutex> lock(mu_);
        auto it = table_.find(nid);
        if (it != table_.end()) {
            it->second.ref_count.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    void dec_ref(NodeID nid) {
        std::shared_lock<std::shared_mutex> lock(mu_);
        auto it = table_.find(nid);
        if (it != table_.end()) {
            it->second.ref_count.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    int32_t ref_count(NodeID nid) const {
        std::shared_lock<std::shared_mutex> lock(mu_);
        auto it = table_.find(nid);
        if (it == table_.end()) return 0;
        return it->second.ref_count.load(std::memory_order_acquire);
    }

    bool publish_new_version(NodeID nid, const VersionHandle& new_vh) {
        std::unique_lock<std::shared_mutex> lock(mu_);
        auto it = table_.find(nid);
        if (it == table_.end() || !it->second.initialized) return false;

        // Monotonic rule: only replace by same-or-larger timestamp
        if (new_vh.ts < it->second.visible.ts) {
            return false;
        }
        it->second.visible = new_vh;
        it->second.latest_seen_ts = std::max(it->second.latest_seen_ts, new_vh.ts);
        return true;
    }

    std::vector<PublishedVersion> dump_visible_versions() const {
        std::vector<PublishedVersion> out;
        std::shared_lock<std::shared_mutex> lock(mu_);
        out.reserve(table_.size());
        for (const auto& kv : table_) {
            if (!kv.second.initialized) continue;
            out.push_back(PublishedVersion{kv.first, kv.second.visible});
        }
        return out;
    }

private:
    mutable std::shared_mutex mu_;
    std::unordered_map<NodeID, NodeVersionState> table_;
};

struct WriteDecision {
    WriteMode mode = WriteMode::CopyOnWrite;
    Address dst_addr = 0;
    VersionHandle old_visible;
};

class CowManager {
public:
    using WriteFn = std::function<void(Address dst_addr, NodeID nid, Timestamp ts)>;
    using ReleaseHook = std::function<void(Address addr)>;

    CowManager(std::shared_ptr<BatchVisibleTable> visible_table,
               std::shared_ptr<CowAllocator> allocator,
               size_t state_bytes)
        : visible_table_(std::move(visible_table)),
          allocator_(std::move(allocator)),
          state_bytes_(state_bytes) {}

    // Pre-pass after minibatch sampling:
    // initialize visible states and record the max timestamp per node.
    void build_batch_metadata(const std::vector<UpdateOccurrence>& occs,
                              const std::unordered_map<NodeID, Address>& base_addr_map) {
        // initialize nodes
        for (const auto& kv : base_addr_map) {
            visible_table_->init_node(kv.first, kv.second);
        }
        // record latest ts
        for (const auto& occ : occs) {
            visible_table_->record_latest_timestamp(occ.nid, occ.ts);
        }
    }

    WriteDecision decide_write(const UpdateOccurrence& occ) {
        auto snap_opt = visible_table_->snapshot(occ.nid);
        if (!snap_opt.has_value()) {
            throw std::runtime_error("decide_write: node not initialized");
        }
        const auto& snap = *snap_opt;

        WriteDecision d;
        d.old_visible = snap.visible;

        const bool no_live_readers = snap.ref_count.load(std::memory_order_acquire) == 0;
        const bool ts_is_latest = occ.is_latest_occurrence && (occ.ts >= snap.latest_seen_ts);

        if (ts_is_latest && no_live_readers) {
            d.mode = WriteMode::InPlace;
            d.dst_addr = snap.base_addr;
        } else {
            auto cow = allocator_->allocate(state_bytes_);
            if (!cow.has_value()) {
                throw std::runtime_error("COW allocator exhausted");
            }
            d.mode = WriteMode::CopyOnWrite;
            d.dst_addr = cow->addr;
        }

        return d;
    }

    // Called when the updater computation for one occurrence finishes.
    // The caller is responsible for actually writing the state to dst_addr first.
    bool publish_after_write(const UpdateOccurrence& occ, const WriteDecision& d) {
        VersionHandle new_vh{
            d.dst_addr,
            occ.ts,
            d.mode == WriteMode::CopyOnWrite,
            next_generation_++
        };
        return visible_table_->publish_new_version(occ.nid, new_vh);
    }

    // Readers should obtain addresses only through the visible mapping.
    Address resolve_visible_addr(NodeID nid) const {
        auto v = visible_table_->visible(nid);
        if (!v.has_value()) {
            throw std::runtime_error("resolve_visible_addr: missing visible version");
        }
        return v->addr;
    }

    // A dependent kernel will read the currently visible version.
    void retain_visible(NodeID nid) {
        visible_table_->inc_ref(nid);
    }

    void release_visible(NodeID nid) {
        visible_table_->dec_ref(nid);
    }

    // Reclaim COW slots that are no longer visible and no longer referenced.
    // The caller supplies the old visible version used before replacement.
    void try_reclaim_old_version(NodeID nid, const VersionHandle& old_vh,
                                 const ReleaseHook& release_hook = nullptr) {
        // Only COW slots need explicit reclamation. Base address is persistent.
        if (!old_vh.from_cow) return;

        // If old version is still the current visible one, cannot reclaim.
        auto current = visible_table_->visible(nid);
        if (!current.has_value()) return;
        if (current->addr == old_vh.addr) return;

        // Readers of the old visible version are tracked per node. In a full implementation,
        // generation-specific refcounts would be even safer. Here we assume release happens
        // after dependent kernels signal completion in order.
        const int32_t rc = visible_table_->ref_count(nid);
        if (rc == 0) {
            allocator_->release(old_vh.addr);
            if (release_hook) release_hook(old_vh.addr);
        }
    }

    std::vector<PublishedVersion> dump_batch_visible_table() const {
        return visible_table_->dump_visible_versions();
    }

private:
    std::shared_ptr<BatchVisibleTable> visible_table_;
    std::shared_ptr<CowAllocator> allocator_;
    size_t state_bytes_;
    std::atomic<uint64_t> next_generation_{1};
};

// ------------------------------
// Example host-side driver
// ------------------------------

struct DummyUpdater {
    // simulate writing updated state to destination address
    void operator()(Address dst_addr, NodeID nid, Timestamp ts) const {
        (void)dst_addr;
        (void)nid;
        (void)ts;
        // In real code:
        //   launch updater CUDA kernel / CPU op
        //   output tensor writes directly to dst_addr
    }
};

static std::vector<UpdateOccurrence>
mark_latest_occurrences(const std::vector<std::pair<NodeID, Timestamp>>& input) {
    std::unordered_map<NodeID, Timestamp> max_ts;
    for (const auto& x : input) {
        max_ts[x.first] = std::max(max_ts[x.first], x.second);
    }

    std::vector<UpdateOccurrence> occs;
    occs.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const auto [nid, ts] = input[i];
        occs.push_back(UpdateOccurrence{
            nid,
            ts,
            i,
            ts == max_ts[nid]
        });
    }
    return occs;
}

void example_cow_flow() {
    auto visible = std::make_shared<BatchVisibleTable>();
    auto allocator = std::make_shared<CowAllocator>(std::vector<CowSlot>{
        {0x90000000, 4096, false},
        {0x90001000, 4096, false},
        {0x90002000, 4096, false},
        {0x90003000, 4096, false},
    });

    CowManager mgr(visible, allocator, /*state_bytes=*/1024);

    std::unordered_map<NodeID, Address> base_addrs = {
        {1, 0x1000},
        {6, 0x2000},
        {9, 0x3000}
    };

    std::vector<std::pair<NodeID, Timestamp>> sampled = {
        {6, 10}, {9, 11}, {1, 12}, {6, 15}, {1, 18}
    };
    auto occs = mark_latest_occurrences(sampled);

    mgr.build_batch_metadata(occs, base_addrs);

    DummyUpdater updater;

    for (const auto& occ : occs) {
        // downstream kernels may still rely on the current visible version
        mgr.retain_visible(occ.nid);

        auto decision = mgr.decide_write(occ);
        updater(decision.dst_addr, occ.nid, occ.ts);
        const bool ok = mgr.publish_after_write(occ, decision);
        assert(ok);

        // once dependent kernels finish:
        mgr.release_visible(occ.nid);
        mgr.try_reclaim_old_version(occ.nid, decision.old_visible);
    }

    const auto final_map = mgr.dump_batch_visible_table();
    std::cout << "[COW] Final visible mapping:\n";
    for (const auto& x : final_map) {
        std::cout << "  nid=" << x.nid
                  << " visible=0x" << std::hex << x.vh.addr << std::dec
                  << " ts=" << x.vh.ts
                  << " cow=" << x.vh.from_cow
                  << " gen=" << x.vh.generation << "\n";
    }
}

} // namespace unity

int main() {
    unity::example_cow_flow();
    return 0;
}