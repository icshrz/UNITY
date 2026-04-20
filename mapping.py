from feature_manager import FeatureManager
import torch
import numpy as np

class FeatureLoader:
    def __init__(self):
        # 创建 FeatureManager 实例
        self.manager = 1

    def load_node_features(self, dataset_name, node_ids):
        """
        Load node features for the given dataset and node IDs.
        
        Args:
            dataset_name (str): The name of the dataset.
            node_ids (list of int): A list of node IDs to load the features for.
        """
        self.manager.load_node_features(dataset_name, node_ids)

    def unload_node_features(self, node_ids):
        """
        Unload node features for the given node IDs.
        
        Args:
            node_ids (list of int): A list of node IDs to unload the features for.
        """
        self.manager.unload_node_features(node_ids)

    def load_edge_features(self, dataset_name, edge_ids):
        """
        Load edge features for the given dataset and edge IDs.
        
        Args:
            dataset_name (str): The name of the dataset.
            edge_ids (list of int): A list of edge IDs to load the features for.
        """
        self.manager.load_edge_features(dataset_name, edge_ids)

    def unload_edge_features(self, edge_ids):
        """
        Unload edge features for the given edge IDs.
        
        Args:
            edge_ids (list of int): A list of edge IDs to unload the features for.
        """
        self.manager.unload_edge_features(edge_ids)

    def get_feature_state(self):
        """
        Get the current state of node and edge features (whether loaded or not and their memory addresses).
        """
        self.manager.get_feature_state()

    def load_node_features(self, node_ids, dataset_name):
        # Load node features for the given ids and dataset
        fet = self.manager.load_continuous_node_features(node_ids, dataset_name)
        ret = fet[node_ids]
        return ret

    def load_edge_features(self, edge_ids, dataset_name):
        # Load edge features for the given ids and dataset
        fet = self.manager.load_continuous_edge_features(edge_ids, dataset_name)
        ret = fet[edge_ids]
        return ret

    def load_nfeat(self, dataset_name):
        node_feat = FeatureManager.load_nfeatures()
        return node_feat
    
    def load_efeat(self, dataset_name):
        edge_feat = FeatureManager.load_nfeatures()
        return edge_feat

# Example usage
if __name__ == "__main__":
    loader = FeatureLoader()
    
    # 加载节点特征和边特征
    loader.load_node_features("your_dataset_name", [1, 2, 3])  # Example node IDs
    loader.load_edge_features("your_dataset_name", [101, 102])  # Example edge IDs
    
    # 查询特征状态
    loader.get_feature_state()
    
    # 卸载特征
    loader.unload_node_features([1, 2])
    loader.unload_edge_features([101])
