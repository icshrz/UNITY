// table.cpp
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <optional>
#include <mutex>
#include <shared_mutex>
#include <algorithm>

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
    uint64_t visible_addr = 0;   // latest visible version
    uint64_t base_addr = 0;      // original slot
    uint64_t cow_addr = 0;       // temp COW slot if any
    int64_t  latest_ts = -1;     // monotonic timestamp
    int32_t  ref_count = 0;      // unfinished dependent reads
    bool     use_cow = false;
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

private:
    mutable std::shared_mutex mu_;
    std::vector<GlobalMapEntry> table_;
};

class MiniBatchSubTable {
public:
    void build(const GlobalMappingTable& global, const std::vector<uint32_t>& active_ids) {
        entries_.clear();
        entries_.reserve(active_ids.size());
        for (auto id : active_ids) {
            auto e = global.lookup(id);
            if (e.has_value()) entries_[id] = *e;
        }
    }

    const GlobalMapEntry* lookup(uint32_t id) const {
        auto it = entries_.find(id);
        return it == entries_.end() ? nullptr : &it->second;
    }

private:
    std::unordered_map<uint32_t, GlobalMapEntry> entries_;
};

class BatchVersionTable {
public:
    void init_node(uint32_t nid, uint64_t base_addr, int64_t init_ts) {
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

private:
    std::unordered_map<uint32_t, BatchVersionEntry> table_;
};

} // namespace unity