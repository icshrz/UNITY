#include "unignn/block_manager.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>

namespace unignn {

// Constructor
BlockManager::BlockManager(size_t total_features)
    : total_features_(total_features) {

    // Calculate the total number of static blocks
    total_static_blocks_ = total_features_ / (BLOCK_SIZE / FEATURE_SIZE);
    if (total_features_ % (BLOCK_SIZE / FEATURE_SIZE) != 0) {
        total_static_blocks_++;  // Round up if there's a remainder
    }

    // Total dynamic blocks (estimate)
    total_dynamic_blocks_ = 1000;

    // Initialize block types and validity
    for (size_t i = 0; i < total_static_blocks_; ++i) {
        block_types_[i] = BlockType::STATIC;
        block_validity_[i] = VALID;  // Static blocks are always valid
    }
}

BlockManager::~BlockManager() {
    // Cleanup if necessary
}

// Get the total number of blocks
size_t BlockManager::getTotalBlocks() const {
    return total_static_blocks_ + total_dynamic_blocks_;
}

// Load a static block from SSD
void BlockManager::loadStaticBlockFromSSD(size_t block_id, void* address) { 
    if (block_types_.find(block_id) == block_types_.end() || block_types_[block_id] != BlockType::STATIC) {
        throw std::out_of_range("Static block ID is invalid");
    }

    std::cout << "Loading static block " << block_id << " from SSD to memory..." << std::endl;

    // Load data from test_feature.pt (manually open and read the binary file)
    std::ifstream file("test_feature.pt", std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open test_feature.pt");
    }

    size_t start_idx = block_id * (BLOCK_SIZE / FEATURE_SIZE);
    size_t end_idx = std::min(start_idx + (BLOCK_SIZE / FEATURE_SIZE), total_features_);
    size_t data_size = (end_idx - start_idx) * FEATURE_SIZE;

    // Seek to the correct position in the file
    file.seekg(start_idx * FEATURE_SIZE, std::ios::beg);

    // Read the required block data into memory
    file.read(reinterpret_cast<char*>(address), data_size);
    if (file.gcount() != data_size) {
        throw std::runtime_error("Failed to read the expected amount of data");
    }

    memory_[block_id] = std::vector<char>((char*)address, (char*)address + data_size);

    file.close();
}

// Load a dynamic block from SSD
void BlockManager::loadDynamicBlockFromSSD(size_t block_id, void* address) {
    if (block_types_.find(block_id) == block_types_.end() || block_types_[block_id] != BlockType::DYNAMIC) {
        throw std::out_of_range("Dynamic block ID is invalid");
    }

    if (block_validity_[block_id] == INVALID) {
        throw std::runtime_error("Cannot load invalid block");
    }

    // Load data from dy_feature.pt (manually open and read the binary file)
    std::ifstream file("dy_feature.pt", std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open dy_feature.pt");
    }

    size_t start_idx = block_id * (BLOCK_SIZE / FEATURE_SIZE);
    size_t end_idx = std::min(start_idx + (BLOCK_SIZE / FEATURE_SIZE), total_features_);
    size_t data_size = (end_idx - start_idx) * FEATURE_SIZE;

    // Seek to the correct position in the file
    file.seekg(start_idx * FEATURE_SIZE, std::ios::beg);

    // Copy the relevant data to the provided address
    file.read(reinterpret_cast<char*>(address), data_size);
    if (file.gcount() != data_size) {
        throw std::runtime_error("Failed to read the expected amount of data");
    }

    file.close();
}

// Write a dynamic block back to SSD
void BlockManager::writeDynamicBlockToSSD(size_t block_id) {
    if (block_types_.find(block_id) == block_types_.end() || block_types_[block_id] != BlockType::DYNAMIC) {
        throw std::out_of_range("Dynamic block ID is invalid");
    }

    if (memory_.find(block_id) != memory_.end()) {
        std::cout << "Writing dynamic block " << block_id << " from memory to SSD..." << std::endl;

        // Open dy_feature.pt in binary mode for writing
        std::ofstream file("dy_feature.pt", std::ios::binary | std::ios::in | std::ios::out);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open dy_feature.pt for writing");
        }

        size_t start_idx = block_id * (BLOCK_SIZE / FEATURE_SIZE);
        size_t data_size = memory_[block_id].size();

        // Seek to the correct position in the file to overwrite data
        file.seekp(start_idx * FEATURE_SIZE, std::ios::beg);

        // Write the block data back to the file
        file.write(reinterpret_cast<const char*>(memory_[block_id].data()), data_size);

        file.close();
    } else {
        std::cout << "Dynamic block " << block_id << " is not in memory!" << std::endl;
    }
}

// Access block data
std::vector<char>& BlockManager::getBlockData(size_t block_id) {
    if (memory_.find(block_id) == memory_.end()) {
        throw std::out_of_range("Block data is not in memory");
    }
    return memory_[block_id];
}

// Add feature data to dynamic block
void BlockManager::addFeatureToDynamicBlock(size_t block_id, const std::vector<char>& feature_data) {
    if (block_types_[block_id] != BlockType::DYNAMIC) {
        throw std::out_of_range("Cannot add feature to static block");
    }

    if (block_validity_[block_id] == INVALID) {
        throw std::runtime_error("Cannot modify invalid dynamic block");
    }

    memory_[block_id] = feature_data;
    std::cout << "Feature added to dynamic block " << block_id << " in memory." << std::endl;
}

// Invalidate dynamic block
void BlockManager::invalidateDynamicBlock(size_t block_id) {
    block_validity_[block_id] = INVALID;
    std::cout << "Dynamic block " << block_id << " has been invalidated." << std::endl;
}

// Calculate static block ID for feature
size_t BlockManager::calculateStaticBlockId(size_t feature_id) const {
    return feature_id / (BLOCK_SIZE / FEATURE_SIZE);
}

// Calculate dynamic block ID for feature
size_t BlockManager::calculateDynamicBlockId(size_t feature_id) const {
    return feature_id / (BLOCK_SIZE / FEATURE_SIZE);
}

// Get ref count of a block
size_t BlockManager::getBlockRefCount(size_t block_id) const {
    size_t total_ref_count = 0;
    auto feature_ref_map = feature_ref_counts_.find(block_id);
    if (feature_ref_map != feature_ref_counts_.end()) {
        for (auto& feature_ref : feature_ref_map->second) {
            total_ref_count += feature_ref.second;
        }
    }
    return total_ref_count;
}

// Increment ref count for a feature in a block
void BlockManager::incrementFeatureRefCount(size_t block_id, size_t feature_id) {
    feature_ref_counts_[block_id][feature_id]++;
}

} // namespace unignn
