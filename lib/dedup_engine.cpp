#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/*
 * UNITY-style temporal-aware deduplication engine
 *
 * Core idea:
 * 1. Repeated node/edge IDs share the same static NF/EF.
 * 2. Timestamp-dependent parts (time embeddings / memory-dependent features)
 *    must remain per-occurrence.
 * 3. So we:
 *      (a) deduplicate unique IDs
 *      (b) compute reusable feature-only transforms once per unique ID
 *      (c) keep timestamp-dependent transforms per occurrence
 *      (d) reassemble the final virtual input tensor according to the original occurrence order
 *
 * This file uses simple std::vector<float> tensors for clarity.
 * Replace with torch::Tensor / custom tensor views / CUDA kernels in your project.
 */

namespace unity {

using NodeID = uint32_t;
using EdgeID = uint32_t;
using Timestamp = int64_t;

struct Vec {
    std::vector<float> data;

    Vec() = default;
    explicit Vec(size_t n, float v = 0.f) : data(n, v) {}
    explicit Vec(std::vector<float> x) : data(std::move(x)) {}

    size_t size() const { return data.size(); }

    float& operator[](size_t i) { return data[i]; }
    const float& operator[](size_t i) const { return data[i]; }
};

static Vec concat(const std::vector<Vec>& xs) {
    size_t total = 0;
    for (const auto& x : xs) total += x.size();
    Vec out(total);
    size_t off = 0;
    for (const auto& x : xs) {
        std::copy(x.data.begin(), x.data.end(), out.data.begin() + off);
        off += x.size();
    }
    return out;
}

static Vec add(const Vec& a, const Vec& b) {
    assert(a.size() == b.size());
    Vec out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
    return out;
}

static Vec relu(const Vec& x) {
    Vec out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = std::max(0.f, x[i]);
    return out;
}

static Vec linear_stub(const Vec& x, size_t out_dim, float scale = 1.0f) {
    // Placeholder for an MLP/Linear operator.
    // In real code, replace with CPU/GPU linear kernel.
    Vec out(out_dim, 0.f);
    if (x.size() == 0) return out;
    for (size_t i = 0; i < out_dim; ++i) {
        float s = 0.f;
        for (size_t j = 0; j < x.size(); ++j) {
            s += x[j] * (0.01f * float((i + j) % 17 + 1));
        }
        out[i] = s * scale;
    }
    return out;
}

struct Occurrence {
    size_t occ_idx = 0;
    NodeID nid = 0;
    std::optional<EdgeID> eid;
    Timestamp ts = -1;
};

struct UniqueIDMap {
    std::vector<NodeID> unique_nids;
    std::vector<int32_t> occ_to_unique_nid_idx; // occurrence -> unique_nids index

    std::vector<EdgeID> unique_eids;
    std::vector<int32_t> occ_to_unique_eid_idx; // occurrence -> unique_eids index, -1 if none
};

class DedupIndexer {
public:
    static UniqueIDMap build(const std::vector<Occurrence>& occs) {
        UniqueIDMap m;
        m.occ_to_unique_nid_idx.resize(occs.size(), -1);
        m.occ_to_unique_eid_idx.resize(occs.size(), -1);

        std::unordered_map<NodeID, int32_t> nid2u;
        std::unordered_map<EdgeID, int32_t> eid2u;

        for (size_t i = 0; i < occs.size(); ++i) {
            NodeID nid = occs[i].nid;
            auto itn = nid2u.find(nid);
            if (itn == nid2u.end()) {
                int32_t idx = static_cast<int32_t>(m.unique_nids.size());
                nid2u[nid] = idx;
                m.unique_nids.push_back(nid);
                m.occ_to_unique_nid_idx[i] = idx;
            } else {
                m.occ_to_unique_nid_idx[i] = itn->second;
            }

            if (occs[i].eid.has_value()) {
                EdgeID eid = *occs[i].eid;
                auto ite = eid2u.find(eid);
                if (ite == eid2u.end()) {
                    int32_t idx = static_cast<int32_t>(m.unique_eids.size());
                    eid2u[eid] = idx;
                    m.unique_eids.push_back(eid);
                    m.occ_to_unique_eid_idx[i] = idx;
                } else {
                    m.occ_to_unique_eid_idx[i] = ite->second;
                }
            }
        }
        return m;
    }
};

struct FeatureStore {
    std::unordered_map<NodeID, Vec> node_features;  // NF
    std::unordered_map<EdgeID, Vec> edge_features;  // EF
};

struct DynamicStore {
    // timestamp-dependent / state-dependent parts
    std::unordered_map<NodeID, Vec> node_memory;    // NM visible at current batch
};

class TimeEncoder {
public:
    explicit TimeEncoder(size_t dim) : dim_(dim) {}

    Vec encode(Timestamp ts) const {
        Vec out(dim_);
        const float base = float(ts % 100000) * 1e-4f;
        for (size_t i = 0; i < dim_; ++i) {
            out[i] = base + 0.01f * float((i % 13) + 1);
        }
        return out;
    }

private:
    size_t dim_;
};

struct ReusableFeatureCache {
    std::unordered_map<NodeID, Vec> node_feat_proj_cache;
    std::unordered_map<EdgeID, Vec> edge_feat_proj_cache;
};

struct DedupOutputs {
    // Reassembled per-occurrence inputs
    std::vector<Vec> per_occurrence_virtual_inputs;

    // Optional debug views
    std::vector<Vec> reusable_node_proj;
    std::vector<Vec> reusable_edge_proj;
    std::vector<Vec> per_occ_time_proj;
    std::vector<Vec> per_occ_memory_proj;
};

class TemporalAwareDedupEngine {
public:
    struct Config {
        size_t node_feat_proj_dim = 32;
        size_t edge_feat_proj_dim = 32;
        size_t time_proj_dim = 16;
        size_t mem_proj_dim = 32;
        bool use_edge_feature = true;
        bool use_node_memory = true;
    };

    explicit TemporalAwareDedupEngine(Config cfg)
        : cfg_(cfg), time_encoder_(cfg.time_proj_dim) {}

    DedupOutputs run(const std::vector<Occurrence>& occs,
                     const FeatureStore& feat_store,
                     const DynamicStore& dyn_store,
                     ReusableFeatureCache* cache = nullptr) const {
        DedupOutputs out;
        const UniqueIDMap idmap = DedupIndexer::build(occs);

        // --------------------------------------------
        // Step 1: reusable transforms for unique node IDs
        // --------------------------------------------
        std::vector<Vec> unique_node_proj(idmap.unique_nids.size());
        for (size_t i = 0; i < idmap.unique_nids.size(); ++i) {
            const NodeID nid = idmap.unique_nids[i];

            if (cache) {
                auto it = cache->node_feat_proj_cache.find(nid);
                if (it != cache->node_feat_proj_cache.end()) {
                    unique_node_proj[i] = it->second;
                    continue;
                }
            }

            auto fit = feat_store.node_features.find(nid);
            if (fit == feat_store.node_features.end()) {
                throw std::runtime_error("Missing node feature for nid=" + std::to_string(nid));
            }

            // Feature-only reusable subcomputation
            Vec proj = relu(linear_stub(fit->second, cfg_.node_feat_proj_dim, 1.0f));
            unique_node_proj[i] = proj;

            if (cache) {
                cache->node_feat_proj_cache[nid] = proj;
            }
        }

        // --------------------------------------------
        // Step 2: reusable transforms for unique edge IDs
        // --------------------------------------------
        std::vector<Vec> unique_edge_proj(idmap.unique_eids.size());
        if (cfg_.use_edge_feature) {
            for (size_t i = 0; i < idmap.unique_eids.size(); ++i) {
                const EdgeID eid = idmap.unique_eids[i];

                if (cache) {
                    auto it = cache->edge_feat_proj_cache.find(eid);
                    if (it != cache->edge_feat_proj_cache.end()) {
                        unique_edge_proj[i] = it->second;
                        continue;
                    }
                }

                auto eit = feat_store.edge_features.find(eid);
                if (eit == feat_store.edge_features.end()) {
                    throw std::runtime_error("Missing edge feature for eid=" + std::to_string(eid));
                }

                Vec proj = relu(linear_stub(eit->second, cfg_.edge_feat_proj_dim, 1.0f));
                unique_edge_proj[i] = proj;

                if (cache) {
                    cache->edge_feat_proj_cache[eid] = proj;
                }
            }
        }

        // --------------------------------------------
        // Step 3: per-occurrence timestamp-dependent transforms
        // --------------------------------------------
        std::vector<Vec> occ_time_proj(occs.size());
        std::vector<Vec> occ_mem_proj(occs.size());

        for (size_t i = 0; i < occs.size(); ++i) {
            const auto& occ = occs[i];

            Vec te = time_encoder_.encode(occ.ts);
            occ_time_proj[i] = relu(linear_stub(te, cfg_.time_proj_dim, 1.0f));

            if (cfg_.use_node_memory) {
                auto mit = dyn_store.node_memory.find(occ.nid);
                if (mit == dyn_store.node_memory.end()) {
                    throw std::runtime_error("Missing node memory for nid=" + std::to_string(occ.nid));
                }
                occ_mem_proj[i] = relu(linear_stub(mit->second, cfg_.mem_proj_dim, 1.0f));
            }
        }

        // --------------------------------------------
        // Step 4: reassemble the per-occurrence virtual input
        // --------------------------------------------
        out.per_occurrence_virtual_inputs.reserve(occs.size());
        for (size_t i = 0; i < occs.size(); ++i) {
            const int32_t nid_u = idmap.occ_to_unique_nid_idx[i];
            assert(nid_u >= 0);

            std::vector<Vec> pieces;
            pieces.push_back(unique_node_proj[nid_u]);

            if (cfg_.use_edge_feature) {
                const int32_t eid_u = idmap.occ_to_unique_eid_idx[i];
                if (eid_u >= 0) {
                    pieces.push_back(unique_edge_proj[eid_u]);
                }
            }

            pieces.push_back(occ_time_proj[i]);

            if (cfg_.use_node_memory) {
                pieces.push_back(occ_mem_proj[i]);
            }

            out.per_occurrence_virtual_inputs.push_back(concat(pieces));
        }

        out.reusable_node_proj = std::move(unique_node_proj);
        out.reusable_edge_proj = std::move(unique_edge_proj);
        out.per_occ_time_proj = std::move(occ_time_proj);
        out.per_occ_memory_proj = std::move(occ_mem_proj);

        return out;
    }

private:
    Config cfg_;
    TimeEncoder time_encoder_;
};

// ------------------------------
// Example
// ------------------------------

static Vec make_range_vec(size_t n, float start) {
    Vec v(n);
    for (size_t i = 0; i < n; ++i) v[i] = start + float(i);
    return v;
}

void example_dedup_flow() {
    // Original occurrence order after sampling:
    // repeated node IDs 6 and 1, repeated edge ID 100
    std::vector<Occurrence> occs = {
        {0, 6, 100, 10},
        {1, 9, 101, 11},
        {2, 1, 102, 12},
        {3, 6, 100, 15},
        {4, 1, 103, 18},
    };

    FeatureStore feat;
    feat.node_features[1] = make_range_vec(8, 1.0f);
    feat.node_features[6] = make_range_vec(8, 6.0f);
    feat.node_features[9] = make_range_vec(8, 9.0f);

    feat.edge_features[100] = make_range_vec(4, 100.0f);
    feat.edge_features[101] = make_range_vec(4, 101.0f);
    feat.edge_features[102] = make_range_vec(4, 102.0f);
    feat.edge_features[103] = make_range_vec(4, 103.0f);

    DynamicStore dyn;
    dyn.node_memory[1] = make_range_vec(6, 10.0f);
    dyn.node_memory[6] = make_range_vec(6, 20.0f);
    dyn.node_memory[9] = make_range_vec(6, 30.0f);

    ReusableFeatureCache cache;

    TemporalAwareDedupEngine::Config cfg;
    cfg.node_feat_proj_dim = 16;
    cfg.edge_feat_proj_dim = 8;
    cfg.time_proj_dim = 6;
    cfg.mem_proj_dim = 10;
    cfg.use_edge_feature = true;
    cfg.use_node_memory = true;

    TemporalAwareDedupEngine engine(cfg);
    auto out = engine.run(occs, feat, dyn, &cache);

    std::cout << "[DEDUP] unique reusable node projections = "
              << out.reusable_node_proj.size() << "\n";
    std::cout << "[DEDUP] unique reusable edge projections = "
              << out.reusable_edge_proj.size() << "\n";
    std::cout << "[DEDUP] per-occurrence virtual inputs = "
              << out.per_occurrence_virtual_inputs.size() << "\n";

    for (size_t i = 0; i < out.per_occurrence_virtual_inputs.size(); ++i) {
        std::cout << "  occ=" << i
                  << " virtual_input_dim="
                  << out.per_occurrence_virtual_inputs[i].size()
                  << " nid=" << occs[i].nid
                  << " ts=" << occs[i].ts
                  << "\n";
    }
}

} // namespace unity

int main() {
    unity::example_dedup_flow();
    return 0;
}