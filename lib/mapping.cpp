#include "mapping.h"

namespace unignn {

// 构造函数
Mapping::Mapping(Table& table) : table(table) {}

// 获取指定子图的特征指针
std::vector<void*> Mapping::getPointersForSubgraph(int subgraph_id) {
    std::vector<void*> pointers;
    const auto& records = table.getRecords();
    
    for (const auto& record : records) {
        if (record.subgraph_id == subgraph_id) {
            pointers.push_back(record.address);  // 记录对应子图的特征指针
        }
    }

    return pointers;
}

// 将C++指针数组转为Python数组
PyObject* Mapping::toPythonArray(std::vector<void*>& pointers) {
    npy_intp dims[1] = {static_cast<npy_intp>(pointers.size())};
    PyObject* array = PyArray_SimpleNewFromData(1, dims, NPY_UINT64, reinterpret_cast<void*>(pointers.data()));
    return array;  // 返回Python数组对象
}

// 合并特定子图的节点特征并返回一个连续的float数组
std::vector<float> Mapping::combineToTensor(int subgraph_id) {
    std::vector<void*> pointers = getPointersForSubgraph(subgraph_id);
    size_t total_size = 0;

    // 计算总的大小
    for (void* ptr : pointers) {
        // 假设每个数据块的大小是已知的或固定的（例如，4个float）
        total_size += sizeof(ptr);  // 这里需要确保每个块的大小是正确的
    }

    // 创建一个合并后的数据存储
    std::vector<float> combined_data(total_size / sizeof(float));

    size_t idx = 0;
    for (void* ptr : pointers) {
        // 假设每个指针的数据是 float 类型
        float* data_ptr = static_cast<float*>(ptr);
        size_t block_size = sizeof(ptr) / sizeof(float);  // 假设每个块的大小已知

        // 填充合并后的数据
        for (size_t j = 0; j < block_size; ++j) {
            combined_data[idx + j] = data_ptr[j];
        }
        idx += block_size;
    }

    return combined_data;  // 返回合并后的数据
}

// 将特定子图的节点特征数据转为CUDA统一内存张量
torch::Tensor Mapping::toCudaTensor(int subgraph_id) {
    std::vector<float> data = combineToTensor(subgraph_id);
    torch::Tensor tensor = torch::from_blob(data.data(), {static_cast<long>(data.size())}, torch::kFloat32).to(torch::kCUDA);

    return tensor;  // 返回转换后的CUDA张量
}

} // namespace unignn
