#ifndef TABLE_H
#define TABLE_H

#include <vector>
#include <tuple>

struct MappingRecord {
    int node_id;   
    void* address;   
    int subgraph_id;    
};

class Table {
public:
    Table() = default;

    void addMapping(int node_id, void* address, int subgraph_id);
    const std::vector<MappingRecord>& getRecords() const;

private:
    std::vector<MappingRecord> records;  
};

#endif
