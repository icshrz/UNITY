#include "table.h"

void Table::addMapping(int node_id, void* address, int subgraph_id) {
    MappingRecord record = {node_id, address, subgraph_id};
    records.push_back(record);
}

const std::vector<MappingRecord>& Table::getRecords() const {
    return records;
}
