#include <iostream>
#include <string>
#include <cstdlib>
#include <random>
#include <omp.h>
#include <cmath>
#include "unignn/core.h"
#include <cuda_runtime.h>  // For GPU support

namespace py = pybind11;

typedef int NodeIDType;
typedef int EdgeIDType;
typedef float TimeStampType;


class ParallelSampler
{
public:
    std::vector<EdgeIDType> indptr;
    std::vector<EdgeIDType> indices;
    std::vector<EdgeIDType> eid;
    std::vector<TimeStampType> ts;
    NodeIDType num_nodes;
    EdgeIDType num_edges;
    int num_thread_per_worker;
    int num_workers;
    int num_threads;
    int num_layers;
    std::vector<int> num_neighbors;
    bool recent;
    bool prop_time;
    int num_history;
    TimeStampType window_duration;
    std::vector<std::vector<std::vector<EdgeIDType>::size_type>> ts_ptr;
    omp_lock_t *ts_ptr_lock;
    std::vector<TemporalGraphBlock> ret;

    // Constructor
    ParallelSampler(std::vector<EdgeIDType> &_indptr, std::vector<EdgeIDType> &_indices,
                    std::vector<EdgeIDType> &_eid, std::vector<TimeStampType> &_ts,
                    int _num_thread_per_worker, int _num_workers, int _num_layers,
                    std::vector<int> &_num_neighbors, bool _recent, bool _prop_time,
                    int _num_history, TimeStampType _window_duration)
        : indptr(_indptr), indices(_indices), eid(_eid), ts(_ts), prop_time(_prop_time),
          num_thread_per_worker(_num_thread_per_worker), num_workers(_num_workers),
          num_layers(_num_layers), num_neighbors(_num_neighbors), recent(_recent),
          num_history(_num_history), window_duration(_window_duration)
    {
        omp_set_num_threads(num_thread_per_worker * num_workers);
        num_threads = num_thread_per_worker * num_workers;
        num_nodes = indptr.size() - 1;
        num_edges = indices.size();
        ts_ptr_lock = (omp_lock_t *)malloc(num_nodes * sizeof(omp_lock_t));
        for (int i = 0; i < num_nodes; i++)
            omp_init_lock(&ts_ptr_lock[i]);
        ts_ptr.resize(num_history + 1);
        for (auto it = ts_ptr.begin(); it != ts_ptr.end(); it++)
        {
            it->resize(indptr.size() - 1);
            #pragma omp parallel for
            for (auto itt = indptr.begin(); itt < indptr.end() - 1; itt++)
                (*it)[itt - indptr.begin()] = *itt;
        }
    }

    // Reset function to reinitialize data
    void reset()
    {
        for (auto it = ts_ptr.begin(); it != ts_ptr.end(); it++)
        {
            it->resize(indptr.size() - 1);
            #pragma omp parallel for
            for (auto itt = indptr.begin(); itt < indptr.end() - 1; itt++)
                (*it)[itt - indptr.begin()] = *itt;
        }
    }

    // GPU-based sampling kernel (for demo purpose, will use a dummy kernel)
    __global__ void sample_kernel(NodeIDType *root_nodes, TimeStampType *root_ts, EdgeIDType *indptr, 
                                   EdgeIDType *indices, EdgeIDType *eid, TimeStampType *ts, 
                                   int *out_neighbors, int num_neighbors)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < num_neighbors)
        {
            NodeIDType n = root_nodes[tid];
            TimeStampType nts = root_ts[tid];
            // Simulate some processing work (such as neighbor sampling)
            out_neighbors[tid] = indices[tid] + n;
        }
    }

    // Function to sample a layer using both CPU and GPU
    void sample_layer_gpu(std::vector<NodeIDType> &_root_nodes, std::vector<TimeStampType> &_root_ts, 
                          int neighs, bool use_ptr, bool from_root)
    {
        // First, check if GPU is available and can be used.
        int device_count = 0;
        cudaGetDeviceCount(&device_count);

        if (device_count > 0)  // GPU available
        {
            NodeIDType *d_root_nodes;
            TimeStampType *d_root_ts;
            EdgeIDType *d_indptr, *d_indices, *d_eid;
            TimeStampType *d_ts;
            int *d_out_neighbors;

            // Allocate memory on GPU
            cudaMalloc(&d_root_nodes, _root_nodes.size() * sizeof(NodeIDType));
            cudaMalloc(&d_root_ts, _root_ts.size() * sizeof(TimeStampType));
            cudaMalloc(&d_indptr, indptr.size() * sizeof(EdgeIDType));
            cudaMalloc(&d_indices, indices.size() * sizeof(EdgeIDType));
            cudaMalloc(&d_eid, eid.size() * sizeof(EdgeIDType));
            cudaMalloc(&d_ts, ts.size() * sizeof(TimeStampType));
            cudaMalloc(&d_out_neighbors, _root_nodes.size() * sizeof(int));

            // Copy data to GPU
            cudaMemcpy(d_root_nodes, _root_nodes.data(), _root_nodes.size() * sizeof(NodeIDType), cudaMemcpyHostToDevice);
            cudaMemcpy(d_root_ts, _root_ts.data(), _root_ts.size() * sizeof(TimeStampType), cudaMemcpyHostToDevice);
            cudaMemcpy(d_indptr, indptr.data(), indptr.size() * sizeof(EdgeIDType), cudaMemcpyHostToDevice);
            cudaMemcpy(d_indices, indices.data(), indices.size() * sizeof(EdgeIDType), cudaMemcpyHostToDevice);
            cudaMemcpy(d_eid, eid.data(), eid.size() * sizeof(EdgeIDType), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ts, ts.data(), ts.size() * sizeof(TimeStampType), cudaMemcpyHostToDevice);

            // Launch the GPU kernel
            int block_size = 256; // Arbitrary block size for example
            int num_blocks = (neighs + block_size - 1) / block_size;
            sample_kernel<<<num_blocks, block_size>>>(d_root_nodes, d_root_ts, d_indptr, d_indices, 
                                                     d_eid, d_ts, d_out_neighbors, neighs);

            // Copy the results back from GPU to CPU
            std::vector<int> out_neighbors(_root_nodes.size());
            cudaMemcpy(out_neighbors.data(), d_out_neighbors, _root_nodes.size() * sizeof(int), cudaMemcpyDeviceToHost);

            // Free GPU memory
            cudaFree(d_root_nodes);
            cudaFree(d_root_ts);
            cudaFree(d_indptr);
            cudaFree(d_indices);
            cudaFree(d_eid);
            cudaFree(d_ts);
            cudaFree(d_out_neighbors);

            // Handle the sampled neighbors (you may want to store them in a TemporalGraphBlock)
            // For example, this could be part of your TemporalGraphBlock creation logic.
        }
        else
        {
            // If no GPU is available, fall back to the CPU-based sampling method
            sample_layer(_root_nodes, _root_ts, neighs, use_ptr, from_root);
        }
    }
};

// Python绑定
PYBIND11_MODULE(parallel_sampler, m)
{
    py::class_<ParallelSampler>(m, "ParallelSampler")
        .def(py::init<std::vector<EdgeIDType>&, std::vector<EdgeIDType>&, 
                     std::vector<EdgeIDType>&, std::vector<TimeStampType>&, 
                     int, int, int, std::vector<int>&, bool, bool, int, TimeStampType>())
        .def("sample_layer_gpu", &ParallelSampler::sample_layer_gpu);
}
