#pragma once

#include <cstdint>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace unity {

enum class RegionType {
    NodeFeature,
    EdgeFeature,
    NodeMemory,
    Mailbox
};

struct GlobalMapEntry {
    uint64_t phys_addr = 0;
    uint32_t bytes = 0;
    bool valid = false;
    bool in_dram = true;
    uint32_t version = 0;
};

struct BatchVersionEntry {
    uint64_t visible_addr = 0;
    uint64_t base_addr = 0;
    uint64_t cow_addr = 0;
    int64_t latest_ts = -1;
    int32_t ref_count = 0;
    bool use_cow = false;
};

class GlobalMappingTable {
public:
    void resize(size_t n) { table_.resize(n); }

    void update(uint32_t id, const GlobalMapEntry& e) {
        std::unique_lock lock(mu_);
        if (id >= table_.size()) table_.resize(id + 1);
        table_[id] = e;
    }

    std::optional<GlobalMapEntry> lookup(uint32_t id) const {
        std::shared_lock lock(mu_);
        if (id >= table_.size()) return std::nullopt;
        if (!table_[id].valid) return std::nullopt;
        return table_[id];
    }

    size_t size() const { return table_.size(); }

private:
    mutable std::shared_mutex mu_;
    std::vector<GlobalMapEntry> table_;
};

class MiniBatchSubTable {
public:
    void build(const GlobalMappingTable& global, const std::vector<uint32_t>& active_ids) {
        entries_.clear();
        for (auto id : active_ids) {
            auto e = global.lookup(id);
            if (e.has_value()) entries_[id] = *e;
        }
    }

    std::optional<GlobalMapEntry> lookup(uint32_t id) const {
        auto it = entries_.find(id);
        if (it == entries_.end()) return std::nullopt;
        return it->second;
    }

    size_t size() const { return entries_.size(); }

private:
    std::unordered_map<uint32_t, GlobalMapEntry> entries_;
};

class BatchVersionTable {
public:
    void init_node(uint32_t nid, uint64_t base_addr, int64_t init_ts = -1) {
        auto& e = table_[nid];
        e.base_addr = base_addr;
        e.visible_addr = base_addr;
        e.latest_ts = init_ts;
    }

    bool can_overwrite(uint32_t nid, int64_t ts, bool is_latest_occurrence) const {
        auto it = table_.find(nid);
        if (it == table_.end()) return false;
        const auto& e = it->second;
        return is_latest_occurrence && e.ref_count == 0 && ts >= e.latest_ts;
    }

    void publish_new_version(uint32_t nid, uint64_t new_addr, int64_t ts, bool use_cow) {
        auto& e = table_[nid];
        if (ts >= e.latest_ts) {
            e.latest_ts = ts;
            e.visible_addr = new_addr;
            e.use_cow = use_cow;
            if (use_cow) e.cow_addr = new_addr;
        }
    }

    uint64_t visible_addr(uint32_t nid) const {
        auto it = table_.find(nid);
        return it == table_.end() ? 0 : it->second.visible_addr;
    }

    void inc_ref(uint32_t nid) { table_[nid].ref_count++; }
    void dec_ref(uint32_t nid) { table_[nid].ref_count--; }
    int32_t ref_count(uint32_t nid) const {
        auto it = table_.find(nid);
        return it == table_.end() ? 0 : it->second.ref_count;
    }
    bool contains(uint32_t nid) const { return table_.count(nid) > 0; }

private:
    std::unordered_map<uint32_t, BatchVersionEntry> table_;
};

inline void bind_table(pybind11::module_& m) {
    namespace py = pybind11;

    py::enum_<RegionType>(m, "RegionType")
        .value("NodeFeature", RegionType::NodeFeature)
        .value("EdgeFeature", RegionType::EdgeFeature)
        .value("NodeMemory", RegionType::NodeMemory)
        .value("Mailbox", RegionType::Mailbox);

    py::class_<GlobalMapEntry>(m, "GlobalMapEntry")
        .def(py::init<>())
        .def_readwrite("phys_addr", &GlobalMapEntry::phys_addr)
        .def_readwrite("bytes", &GlobalMapEntry::bytes)
        .def_readwrite("valid", &GlobalMapEntry::valid)
        .def_readwrite("in_dram", &GlobalMapEntry::in_dram)
        .def_readwrite("version", &GlobalMapEntry::version);

    py::class_<BatchVersionEntry>(m, "BatchVersionEntry")
        .def(py::init<>())
        .def_readwrite("visible_addr", &BatchVersionEntry::visible_addr)
        .def_readwrite("base_addr", &BatchVersionEntry::base_addr)
        .def_readwrite("cow_addr", &BatchVersionEntry::cow_addr)
        .def_readwrite("latest_ts", &BatchVersionEntry::latest_ts)
        .def_readwrite("ref_count", &BatchVersionEntry::ref_count)
        .def_readwrite("use_cow", &BatchVersionEntry::use_cow);

    py::class_<GlobalMappingTable>(m, "GlobalMappingTable")
        .def(py::init<>())
        .def("resize", &GlobalMappingTable::resize)
        .def("update", &GlobalMappingTable::update)
        .def("lookup", &GlobalMappingTable::lookup)
        .def("size", &GlobalMappingTable::size);

    py::class_<MiniBatchSubTable>(m, "MiniBatchSubTable")
        .def(py::init<>())
        .def("build", &MiniBatchSubTable::build)
        .def("lookup", &MiniBatchSubTable::lookup)
        .def("size", &MiniBatchSubTable::size);

    py::class_<BatchVersionTable>(m, "BatchVersionTable")
        .def(py::init<>())
        .def("init_node", &BatchVersionTable::init_node)
        .def("can_overwrite", &BatchVersionTable::can_overwrite)
        .def("publish_new_version", &BatchVersionTable::publish_new_version)
        .def("visible_addr", &BatchVersionTable::visible_addr)
        .def("inc_ref", &BatchVersionTable::inc_ref)
        .def("dec_ref", &BatchVersionTable::dec_ref)
        .def("ref_count", &BatchVersionTable::ref_count)
        .def("contains", &BatchVersionTable::contains);
}

} // namespace unity