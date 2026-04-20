#ifndef BLOCK_MANAGE_H
#define BLOCK_MANAGE_H

#include <unordered_map>
#include <vector>
#include <stdexcept>
#include "core.h"  // Include core.h to interface with Python

namespace unignn {

const size_t FEATURE_SIZE = 1024;  // 1KB per feature
const size_t BLOCK_SIZE = 4096;    // Block size in bytes

enum BlockType { STATIC, DYNAMIC };
enum BlockValidity { VALID, INVALID };

class BlockManager {
public:
    // Constructor
    BlockManager(size_t total_features);

    // Destructor
    ~BlockManager();

    // Get the total number of blocks (static + dynamic)
    size_t getTotalBlocks() const;

    // Load a static block from SSD (test_feature.pt)
    void loadStaticBlockFromSSD(size_t block_id, void* address);

    // Load a dynamic block from SSD (dy_feature.pt)
    void loadDynamicBlockFromSSD(size_t block_id, void* address);

    // Write a dynamic block back to SSD (dy_feature.pt)
    void writeDynamicBlockToSSD(size_t block_id);

    // Access block data
    std::vector<char>& getBlockData(size_t block_id);

    // Add feature data to dynamic block
    void addFeatureToDynamicBlock(size_t block_id, const std::vector<char>& feature_data);

    // Invalidate dynamic block (e.g., to reuse the space)
    void invalidateDynamicBlock(size_t block_id);

    // Calculate the static block ID for a given feature ID
    size_t calculateStaticBlockId(size_t feature_id) const;

    // Calculate the dynamic block ID for a given feature ID
    size_t calculateDynamicBlockId(size_t feature_id) const;

    // Get the ref count of a block (sum of all feature accesses)
    size_t getBlockRefCount(size_t block_id) const;

    // Increment ref count for a feature within a block
    void incrementFeatureRefCount(size_t block_id, size_t feature_id);

private:
    size_t total_features_;
    size_t total_static_blocks_;
    size_t total_dynamic_blocks_;

    std::unordered_map<size_t, BlockType> block_types_;  // Block ID -> Block Type
    std::unordered_map<size_t, BlockValidity> block_validity_;  // Block ID -> Block Validity
    std::unordered_map<size_t, std::vector<char>> memory_;  // Block ID -> Block Data
    std::unordered_map<size_t, std::unordered_map<size_t, size_t>> feature_ref_counts_;  // Block ID -> Feature ID -> Ref Count

    // Helper function to manage static blocks
    void loadStaticBlock(size_t block_id);

    // Helper function to manage dynamic blocks
    void loadDynamicBlock(size_t block_id);
};

} // namespace unignn

#endif // BLOCK_MANAGE_H
