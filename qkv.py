# qkv_processor.py
import torch
import torch.nn as nn
import numpy as np


class QKVProcessor(nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, dim_out, num_dst_nodes, wq, wk, wv, time_zero, time_enc):
        super(QKVProcessor, self).__init__()
        self.dim_node = dim_node_feat
        self.dim_edge = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.num_dst_nodes = num_dst_nodes
        self.w_q = wq
        self.w_k = wk
        self.w_v = wv
        self.time_zero = time_zero
        self.time_enc = time_enc

    def process_q(self, node: torch.Tensor, begin: int = 0, end: int = 0, size: int = 1) -> torch.Tensor:
        # h: [N, dim_node_feat], zero_time_feat: [N, dim_time]
        # weight: [dim_out, dim_node_feat+dim_time]
        time = self.time_zero[begin:end]
        time = time.repeat(size, 1) 
        out = self.w_q(torch.cat([node, time], dim=1))
        return out


    def process_k(self, node: torch.Tensor, edge: torch.Tensor, begin: int = 0, end: int = 0, size: int = 1) -> torch.Tensor:
        # h: [M, dim_node_feat], edge_feat: [M, dim_edge_feat], time_feat: [M, dim_time]
        time = self.time_enc[begin:end]
        time = time.repeat(size, 1) 
        kout = self.w_k(torch.cat([node, edge, time], dim=1))        
        return kout
    
    def process_v(self, node: torch.Tensor, edge: torch.Tensor, begin: int = 0, end: int = 0, size: int = 1) -> torch.Tensor:
        # h: [M, dim_node_feat], edge_feat: [M, dim_edge_feat], time_feat: [M, dim_time]
        time = self.time_enc[begin:end]
        time = time.repeat(size, 1) 
        vout = self.w_v(torch.cat([node, edge, time], dim=1))        
        return vout
