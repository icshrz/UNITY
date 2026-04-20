#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
};

struct Occurrence {
    size_t occ_idx = 0;
    NodeID nid = 0;
    std::optional<EdgeID> eid;
    Timestamp ts = -1;
};

struct UniqueIDMap {
    std::vector<NodeID> unique_nids;
    std::vector<int32_t> occ_to_unique_nid_idx;
    std::vector<EdgeID> unique_eids;
    std::vector<int32_t> occ_to_unique_eid_idx;
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
            auto nid = occs[i].nid;
            if (!nid2u.count(nid)) {
                nid2u[nid] = static_cast<int32_t>(m.unique_nids.size());
                m.unique_nids.push_back(nid);
            }
            m.occ_to_unique_nid_idx[i] = nid2u[nid];

            if (occs[i].eid.has_value()) {
                auto eid = *occs[i].eid;
                if (!eid2u.count(eid)) {
                    eid2u[eid] = static_cast<int32_t>(m.unique_eids.size());
                    m.unique_eids.push_back(eid);
                }
                m.occ_to_unique_eid_idx[i] = eid2u[eid];
            }
        }
        return m;
    }
};

struct FeatureStore {
    std::unordered_map<NodeID, Vec> node_features;
    std::unordered_map<EdgeID, Vec> edge_features;
};

struct DynamicStore {
    std::unordered_map<NodeID, Vec> node_memory;
};

class TimeEncoder {
public:
    explicit TimeEncoder(size_t dim) : dim_(dim) {}

    Vec encode(Timestamp ts) const {
        Vec out(dim_);
        float base = float(ts % 100000) * 1e-4f;
        for (size_t i = 0; i < dim_; ++i) {
            out.data[i] = base + 0.01f * float((i % 13) + 1);
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
    std::vector<Vec> per_occurrence_virtual_inputs;
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
        auto idmap = DedupIndexer::build(occs);

        out.reusable_node_proj.resize(idmap.unique_nids.size());
        for (size_t i = 0; i < idmap.unique_nids.size(); ++i) {
            NodeID nid = idmap.unique_nids[i];
            if (cache && cache->node_feat_proj_cache.count(nid)) {
                out.reusable_node_proj[i] = cache->node_feat_proj_cache[nid];
            } else {
                auto it = feat_store.node_features.find(nid);
                if (it == feat_store.node_features.end()) {
                    throw std::runtime_error("missing node feature");
                }
                out.reusable_node_proj[i] = it->second;
                if (cache) cache->node_feat_proj_cache[nid] = it->second;
            }
        }

        out.reusable_edge_proj.resize(idmap.unique_eids.size());
        for (size_t i = 0; i < idmap.unique_eids.size(); ++i) {
            EdgeID eid = idmap.unique_eids[i];
            if (cache && cache->edge_feat_proj_cache.count(eid)) {
                out.reusable_edge_proj[i] = cache->edge_feat_proj_cache[eid];
            } else {
                auto it = feat_store.edge_features.find(eid);
                if (it == feat_store.edge_features.end()) {
                    throw std::runtime_error("missing edge feature");
                }
                out.reusable_edge_proj[i] = it->second;
                if (cache) cache->edge_feat_proj_cache[eid] = it->second;
            }
        }

        out.per_occ_time_proj.resize(occs.size());
        out.per_occ_memory_proj.resize(occs.size());
        out.per_occurrence_virtual_inputs.resize(occs.size());

        for (size_t i = 0; i < occs.size(); ++i) {
            out.per_occ_time_proj[i] = time_encoder_.encode(occs[i].ts);
            if (cfg_.use_node_memory) {
                auto it = dyn_store.node_memory.find(occs[i].nid);
                if (it == dyn_store.node_memory.end()) {
                    throw std::runtime_error("missing node memory");
                }
                out.per_occ_memory_proj[i] = it->second;
            }

            Vec assembled;
            int32_t nidx = idmap.occ_to_unique_nid_idx[i];
            assembled.data.insert(assembled.data.end(),
                                  out.reusable_node_proj[nidx].data.begin(),
                                  out.reusable_node_proj[nidx].data.end());

            if (cfg_.use_edge_feature && idmap.occ_to_unique_eid_idx[i] >= 0) {
                int32_t eidx = idmap.occ_to_unique_eid_idx[i];
                assembled.data.insert(assembled.data.end(),
                                      out.reusable_edge_proj[eidx].data.begin(),
                                      out.reusable_edge_proj[eidx].data.end());
            }

            assembled.data.insert(assembled.data.end(),
                                  out.per_occ_time_proj[i].data.begin(),
                                  out.per_occ_time_proj[i].data.end());

            if (cfg_.use_node_memory) {
                assembled.data.insert(assembled.data.end(),
                                      out.per_occ_memory_proj[i].data.begin(),
                                      out.per_occ_memory_proj[i].data.end());
            }

            out.per_occurrence_virtual_inputs[i] = std::move(assembled);
        }

        return out;
    }

private:
    Config cfg_;
    TimeEncoder time_encoder_;
};

inline void bind_dedup(pybind11::module_& m) {
    namespace py = pybind11;

    py::class_<Vec>(m, "Vec")
        .def(py::init<>())
        .def(py::init<size_t, float>(), py::arg("n"), py::arg("v") = 0.0f)
        .def_readwrite("data", &Vec::data)
        .def("size", &Vec::size);

    py::class_<Occurrence>(m, "Occurrence")
        .def(py::init<>())
        .def_readwrite("occ_idx", &Occurrence::occ_idx)
        .def_readwrite("nid", &Occurrence::nid)
        .def_readwrite("eid", &Occurrence::eid)
        .def_readwrite("ts", &Occurrence::ts);

    py::class_<UniqueIDMap>(m, "UniqueIDMap")
        .def(py::init<>())
        .def_readwrite("unique_nids", &UniqueIDMap::unique_nids)
        .def_readwrite("occ_to_unique_nid_idx", &UniqueIDMap::occ_to_unique_nid_idx)
        .def_readwrite("unique_eids", &UniqueIDMap::unique_eids)
        .def_readwrite("occ_to_unique_eid_idx", &UniqueIDMap::occ_to_unique_eid_idx);

    py::class_<DedupIndexer>(m, "DedupIndexer")
        .def_static("build", &DedupIndexer::build);

    py::class_<FeatureStore>(m, "FeatureStore")
        .def(py::init<>())
        .def_readwrite("node_features", &FeatureStore::node_features)
        .def_readwrite("edge_features", &FeatureStore::edge_features);

    py::class_<DynamicStore>(m, "DynamicStore")
        .def(py::init<>())
        .def_readwrite("node_memory", &DynamicStore::node_memory);

    py::class_<ReusableFeatureCache>(m, "ReusableFeatureCache")
        .def(py::init<>())
        .def_readwrite("node_feat_proj_cache", &ReusableFeatureCache::node_feat_proj_cache)
        .def_readwrite("edge_feat_proj_cache", &ReusableFeatureCache::edge_feat_proj_cache);

    py::class_<DedupOutputs>(m, "DedupOutputs")
        .def(py::init<>())
        .def_readwrite("per_occurrence_virtual_inputs", &DedupOutputs::per_occurrence_virtual_inputs)
        .def_readwrite("reusable_node_proj", &DedupOutputs::reusable_node_proj)
        .def_readwrite("reusable_edge_proj", &DedupOutputs::reusable_edge_proj)
        .def_readwrite("per_occ_time_proj", &DedupOutputs::per_occ_time_proj)
        .def_readwrite("per_occ_memory_proj", &DedupOutputs::per_occ_memory_proj);

    py::class_<TemporalAwareDedupEngine::Config>(m, "DedupConfig")
        .def(py::init<>())
        .def_readwrite("node_feat_proj_dim", &TemporalAwareDedupEngine::Config::node_feat_proj_dim)
        .def_readwrite("edge_feat_proj_dim", &TemporalAwareDedupEngine::Config::edge_feat_proj_dim)
        .def_readwrite("time_proj_dim", &TemporalAwareDedupEngine::Config::time_proj_dim)
        .def_readwrite("mem_proj_dim", &TemporalAwareDedupEngine::Config::mem_proj_dim)
        .def_readwrite("use_edge_feature", &TemporalAwareDedupEngine::Config::use_edge_feature)
        .def_readwrite("use_node_memory", &TemporalAwareDedupEngine::Config::use_node_memory);

    py::class_<TemporalAwareDedupEngine>(m, "TemporalAwareDedupEngine")
        .def(py::init<TemporalAwareDedupEngine::Config>())
        .def("run", &TemporalAwareDedupEngine::run,
             py::arg("occs"),
             py::arg("feat_store"),
             py::arg("dyn_store"),
             py::arg("cache") = nullptr);
}

} // namespace unity