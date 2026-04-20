import torch
import numpy as np

# Parameters
num_static_features = 100000  # Number of static features (e.g., nodes/edges)
num_dynamic_features = 50000  # Number of dynamic features (non-sequential)
feature_size = 1024  # Each feature is 1KB

# Create static features (test_feature.pt) - Sequential data
static_features = torch.randint(0, 256, (num_static_features, feature_size), dtype=torch.uint8)
torch.save(static_features, "test_feature.pt")

# Create dynamic features (dy_feature.pt) - Random order of feature IDs
dynamic_features = torch.randint(0, 256, (num_dynamic_features, feature_size), dtype=torch.uint8)
torch.save(dynamic_features, "dy_feature.pt")

print("test_feature.pt and dy_feature.pt files have been generated.")

