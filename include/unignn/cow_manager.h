#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
    Address base_addr = 0;
    VersionHandle visible;
    std::atomic<int32_t> ref_count{0};
    Timestamp latest_seen_ts = -1;
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
        s.ref_count.store(0);
    }

    void record_latest_timestamp(NodeID nid, Timestamp ts) {
        std::unique_lock<std::shared_mutex> lock(mu_);
        auto& s = table_[nid];
        s.latest_seen_ts = std::max(s.latest_seen_ts, ts);
    }

    std::optional<VersionHandle> visible(NodeID nid) const {
        std::shared_lock<std::shared_mutex> lock(mu_);
        auto it = table_.find(nid);
        if (it == table_.end() || !it->second.initialized) return std::nullopt;
        return it->second.visible;
    }

    void inc_ref(NodeID nid) { table_[nid].ref_count.fetch_add(1); }
    void dec_ref(NodeID nid) { table_[nid].ref_count.fetch_sub(1); }
    int32_t ref_count(NodeID nid) const {
        auto it = table_.find(nid);
        return it == table_.end() ? 0 : it->second.ref_count.load();
    }

    bool publish_new_version(NodeID nid, const VersionHandle& new_vh) {
        std::unique_lock<std::shared_mutex> lock(mu_);
        auto& s = table_[nid];
        if (new_vh.ts < s.visible.ts) return false;
        s.visible = new_vh;
        s.latest_seen_ts = std::max(s.latest_seen_ts, new_vh.ts);
        return true;
    }

    std::vector<PublishedVersion> dump_visible_versions() const {
        std::vector<PublishedVersion> out;
        std::shared_lock<std::shared_mutex> lock(mu_);
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
    CowManager(std::shared_ptr<BatchVisibleTable> visible_table,
               std::shared_ptr<CowAllocator> allocator,
               size_t state_bytes)
        : visible_table_(std::move(visible_table)),
          allocator_(std::move(allocator)),
          state_bytes_(state_bytes) {}

    void build_batch_metadata(const std::vector<UpdateOccurrence>& occs,
                              const std::unordered_map<NodeID, Address>& base_addr_map) {
        for (const auto& kv : base_addr_map) {
            visible_table_->init_node(kv.first, kv.second);
        }
        for (const auto& occ : occs) {
            visible_table_->record_latest_timestamp(occ.nid, occ.ts);
        }
    }

    WriteDecision decide_write(const UpdateOccurrence& occ) {
        auto curr = visible_table_->visible(occ.nid);
        if (!curr.has_value()) throw std::runtime_error("missing visible version");

        WriteDecision d;
        d.old_visible = *curr;

        bool no_live_readers = visible_table_->ref_count(occ.nid) == 0;
        bool can_inplace = occ.is_latest_occurrence && no_live_readers;

        if (can_inplace) {
            d.mode = WriteMode::InPlace;
            d.dst_addr = curr->addr;
        } else {
            auto slot = allocator_->allocate(state_bytes_);
            if (!slot.has_value()) throw std::runtime_error("COW allocator exhausted");
            d.mode = WriteMode::CopyOnWrite;
            d.dst_addr = slot->addr;
        }
        return d;
    }

    bool publish_after_write(const UpdateOccurrence& occ, const WriteDecision& d) {
        VersionHandle vh{d.dst_addr, occ.ts, d.mode == WriteMode::CopyOnWrite, next_generation_++};
        return visible_table_->publish_new_version(occ.nid, vh);
    }

    Address resolve_visible_addr(NodeID nid) const {
        auto v = visible_table_->visible(nid);
        if (!v.has_value()) throw std::runtime_error("missing visible addr");
        return v->addr;
    }

    void retain_visible(NodeID nid) { visible_table_->inc_ref(nid); }
    void release_visible(NodeID nid) { visible_table_->dec_ref(nid); }

    std::vector<PublishedVersion> dump_batch_visible_table() const {
        return visible_table_->dump_visible_versions();
    }

private:
    std::shared_ptr<BatchVisibleTable> visible_table_;
    std::shared_ptr<CowAllocator> allocator_;
    size_t state_bytes_;
    std::atomic<uint64_t> next_generation_{1};
};

inline void bind_cow(pybind11::module_& m) {
    namespace py = pybind11;

    py::enum_<WriteMode>(m, "WriteMode")
        .value("InPlace", WriteMode::InPlace)
        .value("CopyOnWrite", WriteMode::CopyOnWrite);

    py::class_<VersionHandle>(m, "VersionHandle")
        .def(py::init<>())
        .def_readwrite("addr", &VersionHandle::addr)
        .def_readwrite("ts", &VersionHandle::ts)
        .def_readwrite("from_cow", &VersionHandle::from_cow)
        .def_readwrite("generation", &VersionHandle::generation);

    py::class_<UpdateOccurrence>(m, "UpdateOccurrence")
        .def(py::init<>())
        .def_readwrite("nid", &UpdateOccurrence::nid)
        .def_readwrite("ts", &UpdateOccurrence::ts)
        .def_readwrite("occurrence_idx", &UpdateOccurrence::occurrence_idx)
        .def_readwrite("is_latest_occurrence", &UpdateOccurrence::is_latest_occurrence);

    py::class_<CowSlot>(m, "CowSlot")
        .def(py::init<>())
        .def_readwrite("addr", &CowSlot::addr)
        .def_readwrite("bytes", &CowSlot::bytes)
        .def_readwrite("in_use", &CowSlot::in_use);

    py::class_<WriteDecision>(m, "WriteDecision")
        .def(py::init<>())
        .def_readwrite("mode", &WriteDecision::mode)
        .def_readwrite("dst_addr", &WriteDecision::dst_addr)
        .def_readwrite("old_visible", &WriteDecision::old_visible);

    py::class_<PublishedVersion>(m, "PublishedVersion")
        .def(py::init<>())
        .def_readwrite("nid", &PublishedVersion::nid)
        .def_readwrite("vh", &PublishedVersion::vh);

    py::class_<CowAllocator, std::shared_ptr<CowAllocator>>(m, "CowAllocator")
        .def(py::init<std::vector<CowSlot>>())
        .def("allocate", &CowAllocator::allocate)
        .def("release", &CowAllocator::release);

    py::class_<BatchVisibleTable, std::shared_ptr<BatchVisibleTable>>(m, "BatchVisibleTable")
        .def(py::init<>())
        .def("init_node", &BatchVisibleTable::init_node)
        .def("record_latest_timestamp", &BatchVisibleTable::record_latest_timestamp)
        .def("visible", &BatchVisibleTable::visible)
        .def("inc_ref", &BatchVisibleTable::inc_ref)
        .def("dec_ref", &BatchVisibleTable::dec_ref)
        .def("ref_count", &BatchVisibleTable::ref_count)
        .def("publish_new_version", &BatchVisibleTable::publish_new_version)
        .def("dump_visible_versions", &BatchVisibleTable::dump_visible_versions);

    py::class_<CowManager>(m, "CowManager")
        .def(py::init<std::shared_ptr<BatchVisibleTable>,
                      std::shared_ptr<CowAllocator>,
                      size_t>())
        .def("build_batch_metadata", &CowManager::build_batch_metadata)
        .def("decide_write", &CowManager::decide_write)
        .def("publish_after_write", &CowManager::publish_after_write)
        .def("resolve_visible_addr", &CowManager::resolve_visible_addr)
        .def("retain_visible", &CowManager::retain_visible)
        .def("release_visible", &CowManager::release_visible)
        .def("dump_batch_visible_table", &CowManager::dump_batch_visible_table);
}

} // namespace unity