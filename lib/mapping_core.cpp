#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <torch/torch.h>
#include <torch/script.h>
#include <yaml-cpp/yaml.h>
#include <chrono>



namespace py = pybind11;

struct TensorMetadata {
    std::string device;
    std::string dtype;
    std::vector<int> shape;
};


    

class FeatureCache {
private:
    size_t max_node_cache_size;  // 节点特征缓存的最大容量
    size_t max_edge_cache_size;  // 边特征缓存的最大容量
    
    // 存储特征的实际数据
    std::unordered_map<int, std::vector<float>> node_feature_cache;
    std::unordered_map<int, std::vector<float>> edge_feature_cache;
    std::unordered_map<int, std::pair<bool, float*>> node_feature_state;
    std::unordered_map<int, std::pair<bool, float*>> edge_feature_state;

public:
    FeatureCache(size_t max_node_size, size_t max_edge_size) 
        : max_node_cache_size(max_node_size), max_edge_cache_size(max_edge_size) {}

    // 获取节点特征并应用LRU缓存策略
    bool get_node_feature(int id, int i, std::vector<float>& result, std::ifstream& file, size_t dim) {
        // 如果缓存中存在该特征，直接返回
        if (node_feature_state.find(id) != node_feature_state.end() && node_feature_state[id].first) {
            // 从缓存中读取数据
            std::memcpy(&result[i * dim], node_feature_state[id].second, dim * sizeof(float));
            return true;  // 返回命中
        }

        // 从文件中读取特征
        file.seekg(id * dim * sizeof(float), std::ios::beg);  // 计算偏移量
        //std::vector<float> feature(dim);  // 使用vector存储数据
        file.read(reinterpret_cast<char*>(&result[i * dim]), dim * sizeof(float));  // 读取特征数据
        
        // 将特征数据直接存入缓存
        // if (node_feature_cache.size() < max_node_cache_size) {
        //     node_feature_cache[id] = std::move(feature);  // 将特征数据存入缓存，避免临时变量
        //     node_feature_state[id] = {true, node_feature_cache[id].data()};  // 将数据指针存入缓存状态
        // }

        // 将加载的特征数据复制到result
        //std::memcpy(&result[i * dim], node_feature_cache[id].data(), dim * sizeof(float));

        return false;  // 返回未命中
    }

    // 获取边特征并应用LRU缓存策略
    bool get_edge_feature(int id, int i, std::vector<float>& result, std::ifstream& file, size_t dim) {
        // 如果缓存中存在该特征，直接返回
        if (edge_feature_state.find(id) != edge_feature_state.end() && edge_feature_state[id].first) {
            // 从缓存中读取数据
            std::memcpy(&result[i * dim], edge_feature_state[id].second, dim * sizeof(float));
            return true;  // 返回命中
        }

        // 从文件中读取特征
        file.seekg(id * dim * sizeof(float), std::ios::beg);  // 计算偏移量
        //std::vector<float> feature(dim);  // 使用vector存储数据
        file.read(reinterpret_cast<char*>(&result[i * dim]), dim * sizeof(float));  // 读取特征数据
        
        // 将特征数据直接存入缓存
        // if (edge_feature_cache.size() < max_edge_cache_size) {
        //     edge_feature_cache[id] = std::move(feature);  // 将特征数据存入缓存，避免临时变量
        //     edge_feature_state[id] = {true, edge_feature_cache[id].data()};  // 将数据指针存入缓存状态
        // }

        // 将加载的特征数据复制到result
        //std::memcpy(&result[i * dim], feature.data(), dim * sizeof(float));
        //std::memcpy(&result[i * dim], edge_feature_cache[id].data(), dim * sizeof(float));

        return false;  // 返回未命中
    }

    void get_feature_bat(size_t ids, std::ifstream& file, size_t dim, float rate){
            size_t num_elements = static_cast<size_t>(ids * rate * dim);
    
    std::vector<float> data(num_elements);
    
    file.seekg(0, std::ios::beg);
    
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));
    }



    void cache_all_features(const std::string& dataset_name, size_t dim_node, size_t dim_edge) {
        std::ifstream node_file("/home/zhr/tgl-main/DATA/" + dataset_name + "/node_features.bin", std::ios::binary);
        std::ifstream edge_file("/home/zhr/tgl-main/DATA/" + dataset_name + "/edge_features.bin", std::ios::binary);

        if (!node_file.is_open()) {
            std::cerr << "无法打开节点特征文件！\n";
            return;
        }
        if (!edge_file.is_open()) {
            std::cerr << "无法打开边特征文件！\n";
            return;
        }

        size_t node_id = 0;
        size_t edge_id = 0;

        // 读取节点特征并缓存
        while (node_file) {
            std::vector<float> node_feature(dim_node);
            node_file.read(reinterpret_cast<char*>(node_feature.data()), dim_node * sizeof(float));
            if (node_file) {
                node_feature_cache[node_id] = std::move(node_feature);
                node_feature_state[node_id] = {true, node_feature_cache[node_id].data()};
                ++node_id;
            }
        }

        // 读取边特征并缓存
        while (edge_file) {
            std::vector<float> edge_feature(dim_edge);
            edge_file.read(reinterpret_cast<char*>(edge_feature.data()), dim_edge * sizeof(float));
            if (edge_file) {
                edge_feature_cache[edge_id] = std::move(edge_feature);
                edge_feature_state[edge_id] = {true, edge_feature_cache[edge_id].data()};
                ++edge_id;
            }
        }

        node_file.close();
        edge_file.close();
    }

    void cache_id_features(const std::vector<std::vector<long>>& nid_sets, const std::vector<std::vector<long>>& eid_sets, const std::string& dataset_name, size_t dim_node, size_t dim_edge) { 
    std::ifstream node_file("/home/zhr/tgl-main/DATA/" + dataset_name + "/node_features.bin", std::ios::binary);
    std::ifstream edge_file("/home/zhr/tgl-main/DATA/" + dataset_name + "/edge_features.bin", std::ios::binary);

    if (!node_file.is_open()) {
        std::cerr << "无法打开节点特征文件！\n";
        return;
    }
    if (!edge_file.is_open()) {
        std::cerr << "无法打开边特征文件！\n";
        return;
    }

    // 缓存节点特征
    for (const auto& nid_set : nid_sets) {
        for (const auto& nid : nid_set) {
            node_file.seekg(nid * dim_node * sizeof(float), std::ios::beg);  // 定位到对应的节点特征位置
            std::vector<float> node_feature(dim_node);
            node_file.read(reinterpret_cast<char*>(node_feature.data()), dim_node * sizeof(float));  // 读取节点特征

            if (node_file) {
                node_feature_cache[nid] = std::move(node_feature);  // 将特征数据存入缓存
                node_feature_state[nid] = {true, node_feature_cache[nid].data()};  // 更新缓存状态
            }
        }
    }

    // 缓存边特征
    for (const auto& eid_set : eid_sets) {
        for (const auto& eid : eid_set) {
            edge_file.seekg(eid * dim_edge * sizeof(float), std::ios::beg);  // 定位到对应的边特征位置
            std::vector<float> edge_feature(dim_edge);
            edge_file.read(reinterpret_cast<char*>(edge_feature.data()), dim_edge * sizeof(float));  // 读取边特征

            if (edge_file) {
                edge_feature_cache[eid] = std::move(edge_feature);  // 将特征数据存入缓存
                edge_feature_state[eid] = {true, edge_feature_cache[eid].data()};  // 更新缓存状态
            }
        }
    }

    node_file.close();
    edge_file.close();
}

void clear_cache(){
    node_feature_cache.clear();
    edge_feature_cache.clear();

    // 清空缓存状态
    node_feature_state.clear();
    edge_feature_state.clear();
}
};

// class MailCache {
// private:
//     size_t max_cache_size;  // 节点特征缓存的最大容量
    
//     // 存储特征的实际数据
//     std::unordered_map<int, std::vector<float>> node_mailbox_cache;
//     std::unordered_map<int, std::pair<bool, float*>> node_mailbox_state;

// public:
//     MailCache(size_t max_size) 
//         : max_cache_size(max_size) {}

// bool get_node_mailbox(int id, int i, py::dict& mailbox, std::ifstream& file, size_t dim) {
//         if (node_mailbox_state.find(id) != node_mailbox_state.end() && node_mailbox_state[id].first) {
//             return true;  // 返回命中
//         }
//         py::array_t<float> mail_data;
//         mail_data = py::array_t<float>(dim);  // 创建一个大小为 dim 的新数组
//         mailbox[id] = mail_data;
//         file.seekg(id * dim * sizeof(float), std::ios::beg); 
//         file.read(reinterpret_cast<char*>(mail_data.mutable_data()), dim * sizeof(float));
//         node_mailbox_state[id] = {true, mailbox[id].data()};  // 更新缓存状态


//         return false;  // 返回未命中
//     }

// void clear_cache(){
//     node_mailbox_state.clear();
// }

// void write_back(const std::vector<long>& ids, py::dict& mailbox, size_t dim, std::string dataset_name) {
//         std::ofstream file("/home/zhr/tgl-main/DATA/" + dataset_name + "/mailbox.bin", std::ios::binary | std::ios::out);
        


//         // 遍历给定的 ids，将对应的数据写回文件
//         for (auto id : ids) {
//             float* data = node_mailbox_state[id].second;

//                 // 计算每个节点的起始位置
//             file.seekp(id * dim * sizeof(float), std::ios::beg);

//                 // 写回节点的 mailbox 数据
//             file.write(reinterpret_cast<char*>(data), dim * sizeof(float));
//             if (node_mailbox_state.find(id) != node_mailbox_state.end() && node_mailbox_state[id].first) {
//                 node_mailbox_state[id] = {true, mailbox[id].data()};
//             }
//         }

//         file.close();
//         std::cout << "Cache has been successfully written back to " << filename << std::endl;
//     }
// };



FeatureCache feature_cache(100000, 100000);  // 例如设置最多缓存1000个节点特征
// MailCache mailbox_cache(100000);

TensorMetadata load_metadata(const std::string& dataset, const std::string& feature_type) {
    // 解析 YAML 文件
    std::string metadata_file = "/home/zhr/tgl-main/DATA/" + dataset + "/metadata.yml";
    YAML::Node config = YAML::LoadFile(metadata_file);
    if (config[feature_type]) {
        // 获取元数据
        TensorMetadata metadata;
        metadata.device = config[feature_type]["device"].as<std::string>();
        metadata.dtype = config[feature_type]["dtype"].as<std::string>();
        metadata.shape = config[feature_type]["shape"].as<std::vector<int>>();
        return metadata;
    } else {
        throw std::runtime_error("Metadata for " + feature_type + " not found in " + metadata_file);
    }
}

class FeatureManager {
public:
    // Separate containers to manage node and edge features




    FeatureManager() {}

 

    py::array gather_node_features(const std::vector<long>& ids, const std::string& dataset_name, size_t dim_node) {
    // 构建文件路径
    std::string filename = "/home/zhr/tgl-main/DATA/" + dataset_name + "/node_features.bin";
    
    // 打开文件
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw std::runtime_error("File could not be opened.");
    }

    // 计算元素数量（len(ids)）
    size_t num_elements = ids.size();

    // 创建一个二维数组来存储选定的节点特征
    std::vector<float> result(num_elements * dim_node);

    // 读取每个 id 对应的特征
    for (size_t i = 0; i < num_elements; ++i) {
        // 根据 id 和特征维度计算偏移量
        long id = ids[i];
        file.seekg(id * dim_node * sizeof(float), std::ios::beg); // 设置文件指针到 id 对应的偏移位置
        file.read(reinterpret_cast<char*>(&result[i * dim_node]), dim_node * sizeof(float)); // 读取特征数据
    }

    file.close();

    // 返回 py::array 作为二维数组
    return py::array(py::buffer_info(
        result.data(),                   // 数据指针
        sizeof(float),                   // 每个元素的大小
        py::format_descriptor<float>::format(), // 数据类型的格式描述符
        2,                               // 维度数量
        {num_elements, dim_node},        // 维度
        std::vector<size_t>{dim_node * sizeof(float), sizeof(float)}  // 步长
    ));
}

// 加载指定 ID 的边特征
py::array gather_edge_features(const std::vector<long>& ids, const std::string& dataset_name, size_t dim_edge) {
    // 构建文件路径
    std::string filename = "/home/zhr/tgl-main/DATA/" + dataset_name + "/edge_features.bin";
    
    // 打开文件
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw std::runtime_error("File could not be opened.");
    }

    // 计算元素数量（len(ids)）
    size_t num_elements = ids.size();

    // 创建一个二维数组来存储选定的边特征
    std::vector<float> result(num_elements * dim_edge);

    // 读取每个 id 对应的特征
    for (size_t i = 0; i < num_elements; ++i) {
        // 根据 id 和特征维度计算偏移量
        long id = ids[i];
        file.seekg(id * dim_edge * sizeof(float), std::ios::beg); // 设置文件指针到 id 对应的偏移位置
        file.read(reinterpret_cast<char*>(&result[i * dim_edge]), dim_edge * sizeof(float)); // 读取特征数据
    }

    file.close();

    // 返回 py::array 作为二维数组
    return py::array(py::buffer_info(
        result.data(),                   // 数据指针
        sizeof(float),                   // 每个元素的大小
        py::format_descriptor<float>::format(), // 数据类型的格式描述符
        2,                               // 维度数量
        {num_elements, dim_edge},        // 维度
        std::vector<size_t>{dim_edge * sizeof(float), sizeof(float)}  // 步长
    ));
}

py::list gather_node_uni(const std::vector<std::vector<long>>& id_sets, const std::string& dataset_name, size_t dim_node) {
    std::string filename = "/home/zhr/tgl-main/DATA/" + dataset_name + "/node_features.bin";
    py::list result_list;

    // 处理每个 id set
    for (const auto& ids : id_sets) {
        size_t num_elements = ids.size();
        std::vector<float> result(num_elements * dim_node);

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            throw std::runtime_error("File could not be opened.");
        }

        // 检查并加载每个 id 对应的特征
        for (size_t i = 0; i < num_elements; ++i) {
            long id = ids[i];

            // 检查缓存中是否已加载特征
            bool feature = feature_cache.get_node_feature(id, i, result, file, dim_node);
        }

        file.close();

        // 创建 py::array 并将其添加到返回的列表中
        result_list.append(py::array(py::buffer_info(
            result.data(),                          // 数据指针
            sizeof(float),                          // 每个元素的大小
            py::format_descriptor<float>::format(), // 数据类型的格式描述符
            2,                                      // 维度数量
            {num_elements, dim_node},               // 维度
            std::vector<size_t>{dim_node * sizeof(float), sizeof(float)} // 步长
        )));
    }

    return result_list; // 返回包含多个 py::array 的列表
}

// 用于处理多个ids的gather_edge_uni函数
py::list gather_edge_uni(const std::vector<std::vector<long>>& id_sets, const std::string& dataset_name, size_t dim_edge) {
    std::string filename = "/home/zhr/tgl-main/DATA/" + dataset_name + "/edge_features.bin";
    py::list result_list;

    // 处理每个 id set
    for (const auto& ids : id_sets) {
        size_t num_elements = ids.size();
        std::vector<float> result(num_elements * dim_edge);

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            throw std::runtime_error("File could not be opened.");
        }

        // 检查并加载每个 id 对应的特征
        for (size_t i = 0; i < num_elements; ++i) {
            long id = ids[i];

            // 检查缓存中是否已加载特征
            bool feature = feature_cache.get_edge_feature(id, i, result, file, dim_edge);
        }

        file.close();

        // 创建 py::array 并将其添加到返回的列表中
        result_list.append(py::array(py::buffer_info(
            result.data(),                          // 数据指针
            sizeof(float),                          // 每个元素的大小
            py::format_descriptor<float>::format(), // 数据类型的格式描述符
            2,                                      // 维度数量
            {num_elements, dim_edge},               // 维度
            std::vector<size_t>{dim_edge * sizeof(float), sizeof(float)} // 步长
        )));
    }

    return result_list; // 返回包含多个 py::array 的列表
}

void gather_node_bat(const std::vector<std::vector<long>>& id_sets, const std::string& dataset_name, size_t dim_node, float rate) {
    std::string filename = "/home/zhr/tgl-main/DATA/" + dataset_name + "/node_features.bin";
    py::list result_list;

    // 处理每个 id set
    for (const auto& ids : id_sets) {
        size_t num_elements = ids.size();
        std::vector<float> result(num_elements * dim_node);

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            throw std::runtime_error("File could not be opened.");
        }
            // 检查缓存中是否已加载特征
        feature_cache.get_feature_bat(num_elements, file, dim_node, rate);

        file.close();

    }

}

// 用于处理多个ids的gather_edge_uni函数
void gather_edge_bat(const std::vector<std::vector<long>>& id_sets, const std::string& dataset_name, size_t dim_edge, float rate) {
    std::string filename = "/home/zhr/tgl-main/DATA/" + dataset_name + "/edge_features.bin";
    py::list result_list;

    // 处理每个 id set
    for (const auto& ids : id_sets) {
        size_t num_elements = ids.size();
        std::vector<float> result(num_elements * dim_edge);

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            throw std::runtime_error("File could not be opened.");
        }

            // 检查缓存中是否已加载特征
        feature_cache.get_feature_bat(num_elements, file, dim_edge, rate);

        file.close();
    }

}

py::array load_features(const std::string& filename, const TensorMetadata& metadata) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw std::runtime_error("File could not be opened.");
    }

    // 计算元素数量
    size_t num_elements = 1;
    for (size_t dim : metadata.shape) {
        num_elements *= dim;
    }

    // 读取数据到 std::vector
    std::vector<float> result(num_elements);
    file.read(reinterpret_cast<char*>(result.data()), num_elements * sizeof(float));
    if (!file) {
        std::cerr << "Error reading data from file." << std::endl;
        throw std::runtime_error("File read error.");
    }

    file.close();

    // 创建 py::array，避免内存复制
    return py::array(py::buffer_info(
        result.data(),                     // 数据指针
        sizeof(float),                     // 每个元素的大小
        py::format_descriptor<float>::format(), // 数据类型的格式描述符
        metadata.shape.size(),                      // 维度数量
        metadata.shape,                             // 维度
        std::vector<size_t>(metadata.shape.size(), sizeof(float)) // 步长
    ));
}

void cache_features(const std::string& dataset_name, size_t dim_node, size_t dim_edge){
    feature_cache.cache_all_features(dataset_name, dim_node, dim_edge);
}

void cache_id_features(const std::vector<std::vector<long>>& nids, const std::vector<std::vector<long>>& eids, const std::string& dataset_name, size_t dim_node, size_t dim_edge){
    feature_cache.cache_id_features(nids, eids, dataset_name, dim_node, dim_edge);
}

void clear_cache(){
    feature_cache.clear_cache();
}

// void load_mailbox(const std::vector<long>& ids, py::dict& mailbox, size_t dim){
//     std::string filename = "/home/zhr/tgl-main/DATA/" + dataset_name + "/mailbox.bin";


//         size_t num_elements = ids.size();

//         std::ifstream file(filename, std::ios::binary);
//         if (!file.is_open()) {
//             std::cerr << "Error opening file: " << filename << std::endl;
//             throw std::runtime_error("File could not be opened.");
//         }

//         // 检查并加载每个 id 对应的特征
//         for (size_t i = 0; i < num_elements; ++i) {
//             long id = ids[i];

//             // 检查缓存中是否已加载特征
//             bool hit = mailbox_cache.get_node_mailbox(id, mailbox, file, dim);
//         }

//         file.close();

// }

// void write_back(const std::vector<long>& ids, py::dict& mailbox, size_t dim, std::string dataset_name){
//     mailbox_cache.write_back(ids, mailbox, dataset_name)
// }


// 加载节点特征
py::array load_nfeatures(const std::string& dataset) {
    TensorMetadata node_metadata = load_metadata(dataset, "node");
    std::string filename = "/home/zhr/tgl-main/DATA/" + dataset + "/node_features.bin";
    return load_features(filename, node_metadata);
}

// 加载边缘特征
py::array load_efeatures(const std::string& dataset) {
    TensorMetadata edge_metadata = load_metadata(dataset, "edge");
    std::string filename = "/home/zhr/tgl-main/DATA/" + dataset + "/edge_features.bin";
    return load_features(filename, edge_metadata);
}

std::vector<size_t> load_shape(const std::string& dataset, const std::string& tensor_type) {
    // 读取 YAML 格式的元数据文件
    std::string metadata_file = "/home/zhr/tgl-main/DATA/"+ dataset + "/metadata.yml";

    YAML::Node config = YAML::LoadFile(metadata_file);

    // 获取指定张量类型的形状
    const YAML::Node& tensor_info = config[tensor_type];

    std::vector<size_t> shape;
    if (tensor_info["shape"]) {
        for (auto dim : tensor_info["shape"]) {
            shape.push_back(dim.as<size_t>());
        }
    }

    return shape;
}

std::string load_dtype(const std::string& dataset, const std::string& tensor_type) {
    std::string metadata_file = "/home/zhr/tgl-main/DATA/"+ dataset + "/metadata.yml";

    YAML::Node config = YAML::LoadFile(metadata_file);

    // 获取指定张量类型的形状
    const YAML::Node& tensor_info = config[tensor_type];

    std::string dtype;
    dtype = tensor_info["dtype"].as<std::string>();
    return dtype;
}

std::tuple<std::vector<size_t>, std::vector<size_t>> load_shapes(const std::string& dataset) {
    // 加载节点和边的形状
    std::vector<size_t> node_shape = load_shape(dataset, "node");
    std::vector<size_t> edge_shape = load_shape(dataset, "edge");

    // 返回节点和边的形状
    return std::make_tuple(node_shape, edge_shape);
}

std::tuple<std::string, std::string> load_dtypes(const std::string& dataset){
    std::string node_dtype = load_dtype(dataset, "node");
    std::string edge_dtype = load_dtype(dataset, "edge");

    // 返回节点和边的形状
    return std::make_tuple(node_dtype, edge_dtype);
}




};

// Pybind11 bindings
PYBIND11_MODULE(feature_manager, m) {
    py::class_<FeatureManager>(m, "FeatureManager")
        .def(py::init<>())
        .def("gather_edge_features", &FeatureManager::gather_edge_features)
        .def("gather_node_features", &FeatureManager::gather_node_features)
        .def("gather_edge_uni", &FeatureManager::gather_edge_uni)
        .def("gather_node_uni", &FeatureManager::gather_node_uni)
        .def("load_nfeatures", &FeatureManager::load_nfeatures)
        .def("load_efeatures", &FeatureManager::load_efeatures)
        .def("load_shape", &FeatureManager::load_shape)
        .def("load_shapes", &FeatureManager::load_shapes)
        .def("load_dtype", &FeatureManager::load_dtype)
        .def("load_dtypes", &FeatureManager::load_dtypes)
        .def("cache_features", &FeatureManager::cache_features)
        .def("cache_id_features", &FeatureManager::cache_id_features)
        .def("clear_cache", &FeatureManager::clear_cache)
        .def("gather_node_bat", &FeatureManager::gather_node_bat)
        .def("gather_edge_bat", &FeatureManager::gather_edge_bat);
        // .def("load_mailbox", &FeatureManager::load_mailbox)
        // .def("write_back", &FeatureManager::write_back);
}
