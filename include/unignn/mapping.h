#ifndef MAPPING_H
#define MAPPING_H

#include "table.h"
#include <vector>
#include <tuple>
#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h>

#include "unignn/core.h"
namespace unignn {

class Mapping {
public:
    Mapping(Table& table);
    
    // 获取特定子图的节点特征指针
    std::vector<void*> getPointersForSubgraph(int subgraph_id);

    // 将C++指针数组转为Python数组
    static PyObject* toPythonArray(std::vector<void*>& pointers);
    
    // Python接口：将特定子图的数据合并为张量
    std::vector<float> combineToTensor(int subgraph_id);
    
    // Python接口：将特定子图的数据转为CUDA统一内存张量
    torch::Tensor toCudaTensor(int subgraph_id);

private:
    Table& table;  // 存储表格对象引用
};

} // namespace unignn

// pybind11接口函数声明
PYBIND11_MODULE(mapping, m) {
    pybind11::class_<Mapping>(m, "Mapping")
        .def(pybind11::init<Table&>())
        .def("getPointersForSubgraph", &Mapping::getPointersForSubgraph)
        .def("combineToTensor", &Mapping::combineToTensor)
        .def("toCudaTensor", &Mapping::toCudaTensor);
}

#endif // MAPPING_H
