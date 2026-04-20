import torch
import dgl
import math
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import time
from typing import Tuple


class TimeEncode(torch.nn.Module):

    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim, dtype=np.float32))).reshape(dim, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output

class EdgePredictor(torch.nn.Module):

    def __init__(self, dim_in):
        super(EdgePredictor, self).__init__()
        self.dim_in = dim_in
        self.src_fc = torch.nn.Linear(dim_in, dim_in)
        self.dst_fc = torch.nn.Linear(dim_in, dim_in)
        self.out_fc = torch.nn.Linear(dim_in, 1)

    def forward(self, h, neg_samples=1):
        num_edge = h.shape[0] // (neg_samples + 2)
        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge:2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge:])
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)




# class TransformerAttentionLayer(nn.Module):
#     def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, combined=False, num_block=4):
#         super(TransformerAttentionLayer, self).__init__()
#         self.num_head = num_head
#         self.dim_node_feat = dim_node_feat
#         self.dim_edge_feat = dim_edge_feat
#         self.dim_time = dim_time
#         self.dim_out = dim_out
#         self.dropout = nn.Dropout(dropout)
#         self.att_dropout = nn.Dropout(att_dropout)
#         self.att_act = nn.LeakyReLU(0.2)
#         self.combined = combined
#         self.num_block = num_block  # 分块数量
#         self.time_enc = TimeEncode(dim_time)

#         # 定义线性层
#         self.w_q = nn.Linear(dim_node_feat + dim_time, dim_out)
#         self.w_k = nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
#         self.w_v = nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
#         self.w_out = nn.Linear(dim_node_feat + dim_out, dim_out)
#         self.layer_norm = nn.LayerNorm(dim_out)


    # @torch.jit.script
    # def process_q_script(h: torch.Tensor, zero_time_feat: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, block_size: int = 10) -> torch.Tensor:
    #     # h: [N, dim_node_feat], zero_time_feat: [N, dim_time]
    #     # weight: [dim_out, dim_node_feat+dim_time]
    #     N = h.size(0)
    #     dim_node = h.size(1)
    #     out = torch.empty((N, weight.size(0)), device=h.device)

    #     # 将 weight 拆分为两部分，避免每次拼接：w1 对应节点特征，w2 对应时间特征
    #     w1 = weight[:, :dim_node]           # [dim_out, dim_node_feat]
    #     w2 = weight[:, dim_node:]            # [dim_out, dim_time]
        
    #     for i in range(0, N, block_size):
    #         # 处理每个块，block_size 为 10，剩余部分会小于 10
    #         end = min(i + block_size, N)
    #         tmp1 = torch.matmul(h[i:end], w1.t())   # [block_size, dim_out]
    #         tmp2 = torch.matmul(zero_time_feat[i:end], w2.t())  # [block_size, dim_out]
            
    #         # 扩展 bias 以便它与计算结果的维度一致
    #         bias_block = bias.unsqueeze(0).expand(tmp1.size(0), -1)  # [block_size, dim_out]
            
    #         out[i:end] = (tmp1 + tmp2 + bias_block)  # 合并结果
    #     return out


    # @torch.jit.script
    # def process_kv_script(h: torch.Tensor, edge_feat: torch.Tensor, time_feat: torch.Tensor,
    #                     weight_k: torch.Tensor, bias_k: torch.Tensor,
    #                     weight_v: torch.Tensor, bias_v: torch.Tensor, block_size: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    #     # h: [M, dim_node_feat], edge_feat: [M, dim_edge_feat], time_feat: [M, dim_time]
    #     M = edge_feat.size(0)
    #     dim_node = h.size(1)
    #     dim_edge = edge_feat.size(1)
    #     K_out = torch.empty((M, weight_k.size(0)), device=h.device)
    #     V_out = torch.empty((M, weight_v.size(0)), device=h.device)

    #     # 对 K 层的权重进行分解：w1_k 对应节点，w2_k 对应边特征，w3_k 对应时间
    #     w1_k = weight_k[:, :dim_node]                      # [dim_out, dim_node_feat]
    #     w2_k = weight_k[:, dim_node:dim_node+dim_edge]        # [dim_out, dim_edge_feat]
    #     w3_k = weight_k[:, dim_node+dim_edge:]              # [dim_out, dim_time]
    #     # 对 V 层的权重同理
    #     w1_v = weight_v[:, :dim_node]
    #     w2_v = weight_v[:, dim_node:dim_node+dim_edge]
    #     w3_v = weight_v[:, dim_node+dim_edge:]

    #     for i in range(0, M, block_size):
    #         # 处理每个块，block_size 为 10，剩余部分会小于 10
    #         end = min(i + block_size, M)
    #         tmp1_k = torch.matmul(h[i:end], w1_k.t())
    #         tmp2_k = torch.matmul(edge_feat[i:end], w2_k.t())
    #         tmp3_k = torch.matmul(time_feat[i:end], w3_k.t())
    #         # 扩展 bias_k 以便它与计算结果的维度一致
    #         bias_k_block = bias_k.unsqueeze(0).expand(tmp1_k.size(0), -1)  # [block_size, dim_out]
    #         K_out[i:end] = (tmp1_k + tmp2_k + tmp3_k + bias_k_block)
            
    #         tmp1_v = torch.matmul(h[i:end], w1_v.t())
    #         tmp2_v = torch.matmul(edge_feat[i:end], w2_v.t())
    #         tmp3_v = torch.matmul(time_feat[i:end], w3_v.t())
    #         # 扩展 bias_v 以便它与计算结果的维度一致
    #         bias_v_block = bias_v.unsqueeze(0).expand(tmp1_v.size(0), -1)  # [block_size, dim_out]
    #         V_out[i:end] = (tmp1_v + tmp2_v + tmp3_v + bias_v_block)
        
    #     return K_out, V_out




#     def forward(self, b):
#         # 假设 b.edata['dt'] 为时间差，b.edata['f'] 为边特征，
#         # b.srcdata['h'] 包含了所有节点特征，前 b.num_dst_nodes() 个为目标节点，其余为边关联的源节点数据（散乱存储）

#         # 计算时间编码
#         time_feat = self.time_enc(b.edata['dt'])
#         zero_time = torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=b.srcdata['h'].device)
#         zero_time_feat = self.time_enc(zero_time)

#         # 这里调用 TorchScript 融合的函数进行 per-sample 计算
#         time_q_start = time.time()
#         Q_out = self.process_q_script(b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat, self.w_q.weight, self.w_q.bias)
#         # 假设 b.srcdata['h'][b.num_dst_nodes():] 对应每条边的相关节点特征（散乱存储）
#         K_out, V_out = self.process_kv_script(b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat,
#                                                self.w_k.weight, self.w_k.bias,
#                                                self.w_v.weight, self.w_v.bias)
#         time_q_end = time.time()
#         print(f"Time taken for computing Q, K, V: {time_q_end - time_q_start:.4f} seconds")
#         # 后续注意力计算不变
#         Q_out = Q_out[b.edges()[1]]
#         Q_out = torch.reshape(Q_out, (Q_out.shape[0], self.num_head, -1))
#         K_out = torch.reshape(K_out, (K_out.shape[0], self.num_head, -1))
#         V_out = torch.reshape(V_out, (V_out.shape[0], self.num_head, -1))

#         att = torch.sum(Q_out * K_out, dim=2)
#         att = self.att_act(att)
#         att = dgl.ops.edge_softmax(b, att)
#         att = self.att_dropout(att)

#         V_out = torch.reshape(V_out * att[:, :, None], (V_out.shape[0], -1))
#         # 将计算结果写回节点数据，这里为了示例用 concat 进行拼接（可根据需要修改）
#         b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V_out.shape[1]), device=b.srcdata['h'].device), V_out], dim=0)
#         b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))

#         rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
#         rst = self.w_out(rst)
#         rst = F.relu(self.dropout(rst))
#         return self.layer_norm(rst)


# class TransfomerAttentionLayer(torch.nn.Module):
#     def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, combined=False, num_block=4):
#         super(TransfomerAttentionLayer, self).__init__()
#         self.num_head = num_head
#         self.dim_node_feat = dim_node_feat
#         self.dim_edge_feat = dim_edge_feat
#         self.dim_time = dim_time
#         self.dim_out = dim_out
#         self.dropout = torch.nn.Dropout(dropout)
#         self.att_dropout = torch.nn.Dropout(att_dropout)
#         self.att_act = torch.nn.LeakyReLU(0.2)
#         self.combined = combined
#         self.num_block = num_block  # Define number of blocks to split the computation
#         self.time_enc = TimeEncode(dim_time)

#         # Initialize original w_q, w_k, w_v
#         self.w_q = nn.Linear(dim_node_feat + dim_time, dim_out)
#         self.w_k = nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
#         self.w_v = nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)

#         self.w_out = torch.nn.Linear(dim_node_feat + dim_out, dim_out)
#         self.layer_norm = torch.nn.LayerNorm(dim_out)

#     def forward(self, b):
#         assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0)

#         # Time encoding for Q and K/V
#         time_q_start = time.time()
#         time_feat = self.time_enc(b.edata['dt'])
#         zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=torch.device('cuda:0')))

#         # Process Q (nodes + time)
#         Q_out = self.process_q(b, zero_time_feat)
        
#         # Process K and V (nodes + edges + time)
#         K_out, V_out = self.process_kv(b, time_feat)

#         time_q_end = time.time()
#         Q_out = Q_out[b.edges()[1]]
#         # Reshape Q, K, V for multi-head attention
#         time_att_start = time.time()
#         Q_out = torch.reshape(Q_out, (Q_out.shape[0], self.num_head, -1))
#         K_out = torch.reshape(K_out, (K_out.shape[0], self.num_head, -1))
#         V_out = torch.reshape(V_out, (V_out.shape[0], self.num_head, -1))

#         # Compute attention scores
#         att = torch.sum(Q_out * K_out, dim=2)  # Shape: (num_edges, num_head), dot product between Q and K
#         att = self.att_act(att)
        
#         # Apply edge softmax
#         att = dgl.ops.edge_softmax(b, att)  # Apply softmax over edges
#         att = self.att_dropout(att)

#         # Apply attention weights to V
#         V_out = torch.reshape(V_out * att[:, :, None], (V_out.shape[0], -1))
#         b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V_out.shape[1]), device=torch.device('cuda:0')), V_out], dim=0)
#         b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))

#         rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
#         rst = self.w_out(rst)
#         rst = torch.nn.functional.relu(self.dropout(rst))

#         time_att_end = time.time()

#         print(f"Time taken for computing Q, K, V: {time_q_end - time_q_start:.4f} seconds")
#         print(f"Time taken for attention computation: {time_att_end - time_att_start:.4f} seconds")

#         return self.layer_norm(rst)

#     def process_q(self, b, zero_time_feat):
#         """
#         Process Q by combining node features and time features before linear transformation.
#         """
#         outputs = []
#         # Combine node features and time features
#         for i in range(b.num_dst_nodes()):

#             # Apply the original weight for Q after combining the features
#             Q_out = self.w_q(torch.cat([b.srcdata['h'][i:i+1], zero_time_feat[i:i+1]], dim=1))
#             outputs.append(Q_out)

#         return torch.cat(outputs, dim=0)

#     def process_kv(self, b, time_feat):
#         """
#         Process K and V by combining node features, edge features, and time features before linear transformation.
#         """
#         K_outputs, V_outputs = [], []
        
#         # Combine node features, edge features, and time features
#         for i in range(b.num_edges()):
#             # Process K (node features + edge features + time features)

#             # Apply the original weight for K after combining the features
#             K_out = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes() + i:b.num_dst_nodes() + i + 1], b.edata['f'][i:i+1], time_feat[i:i+1]], dim=1))

#             # Process V (node features + edge features + time features)

#             # Apply the original weight for V after combining the features
#             V_out = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes() + i:b.num_dst_nodes() + i + 1], b.edata['f'][i:i+1], time_feat[i:i+1]], dim=1))

#             K_outputs.append(K_out)
#             V_outputs.append(V_out)

#         return torch.cat(K_outputs, dim=0), torch.cat(V_outputs, dim=0)

        
class TransformerAttentionLayer(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, combined=False, num_block=4):
        super(TransformerAttentionLayer, self).__init__()
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(att_dropout)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.combined = combined
        self.num_block = num_block  # Define number of blocks to split the computation
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        self.w_q = torch.nn.Linear(dim_node_feat + dim_time, dim_out)
        self.w_k = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
        self.w_v = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)

        self.w_out = torch.nn.Linear(dim_node_feat + dim_out, dim_out)
        self.layer_norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b, Q_out=None, K_out=None, V_out=None, times=1):
        assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0)
        
        Q = Q_out
        K = K_out
        V = V_out
        time_q_start = time.time()
        if b.num_edges() == 0:
            # return torch.zeros((b.num_dst_nodes(), self.dim_out), device=torch.device('cuda:0'))
            return torch.zeros((b.num_dst_nodes(), self.dim_out), device=torch.device('cpu'))

        if Q_out==None and self.dim_time > 0:
            time_feat = self.time_enc(b.edata['dt'])
            # zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=torch.device('cuda:0')))
            zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=torch.device('cpu')))

            # xq = b.srcdata['h'][:b.num_dst_nodes()].repeat(times, 1) 
            # zq = zero_time_feat.repeat(times, 1) 
            # xk = b.srcdata['h'][b.num_dst_nodes():].repeat(times, 1) 
            # yk = b.edata['f'].repeat(times, 1) 
            # zk = time_feat.repeat(times, 1) 
            Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
            K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
            V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
            # Q = self.w_q(torch.cat([xq, zq], dim=1))[b.edges()[1]]
            # K = self.w_k(torch.cat([xk, yk, zk], dim=1))
            # V = self.w_v(torch.cat([xk, yk, zk], dim=1))
            # print(f"tensor: {b.edges()[0] - b.num_dst_nodes()}")
            # print(f"tensor shape: {(b.edges()[0] - b.num_dst_nodes()).shape[0]}")
            # print(f"xk: {xk.shape[1]}, yk: {yk.shape[1]}, zk {zk.shape[1]}, cat:{torch.cat([xk, yk, zk], dim=1).shape[1]}")
        elif Q_out==None and self.dim_time == 0:
            # xq = b.srcdata['h'][:b.num_dst_nodes()].repeat(times, 1) 
            # xk = b.srcdata['h'][b.num_dst_nodes():].repeat(times, 1) 
            # yk = b.edata['f'].repeat(times, 1) 
            Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()]], dim=1))[b.edges()[1]]
            K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
            V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
            # Q = self.w_q(xq)[b.edges()[1]]
            # K = self.w_k(torch.cat([xk, yk], dim=1))
            # V = self.w_v(torch.cat([xk, yk], dim=1))
        time_q_end = time.time()
        print(f"Time taken for qkv computation: {time_q_end - time_q_start:.4f} seconds")
        Q = Q[b.edges()[1]]
        time_att_start = time.time()
        Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
        K = torch.reshape(K, (K.shape[0], self.num_head, -1))
        V = torch.reshape(V, (V.shape[0], self.num_head, -1))
        att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
        att = self.att_dropout(att)
        V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
        
        # b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device('cuda:0')), V], dim=0)
        b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device('cpu')), V], dim=0)

        b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
        rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
        rst = self.w_out(rst)
        rst = torch.nn.functional.relu(self.dropout(rst))
        time_att_end = time.time()
        # print(f"subgraph src nodes: {b.num_src_nodes()}")
        # print(f"subgraph dst nodes: {b.num_dst_nodes()}")
        # print(f"subgraph nf shape: {b.srcdata['h'].shape}")
        print(f"Time taken for att computation: {time_att_end - time_att_start:.4f} seconds")

        return self.layer_norm(rst)      


# class TransformerAttentionLayer(torch.nn.Module):
#     def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, combined=False, num_block=4):
#         super(TransformerAttentionLayer, self).__init__()
#         self.num_head = num_head
#         self.dim_node_feat = dim_node_feat
#         self.dim_edge_feat = dim_edge_feat
#         self.dim_time = dim_time
#         self.dim_out = dim_out
#         self.dropout = torch.nn.Dropout(dropout)
#         self.att_dropout = torch.nn.Dropout(att_dropout)
#         self.att_act = torch.nn.LeakyReLU(0.2)
#         self.combined = combined
#         self.num_block = num_block  # Define number of blocks to split the computation
#         self.time_enc = TimeEncode(dim_time)
#         self.w_q = torch.nn.Linear(dim_node_feat + dim_time, dim_out)
#         self.w_k = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
#         self.w_v = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)

#         self.w_out = torch.nn.Linear(dim_node_feat + dim_out, dim_out)
#         self.layer_norm = torch.nn.LayerNorm(dim_out)

#     def forward(self, b):
#         assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0)
#         time_q_start = time.time()
#         time_feat = self.time_enc(b.edata['dt'])
#         zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=torch.device('cuda:0')))
#         Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
#         K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
#         V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
#         time_q_end = time.time()

#         # Reshape Q, K, V
#         time_att_start = time.time()
#         Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
#         K = torch.reshape(K, (K.shape[0], self.num_head, -1))
#         V = torch.reshape(V, (V.shape[0], self.num_head, -1))
#         att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
#         att = self.att_dropout(att)
#         V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
#         b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device('cuda:0')), V], dim=0)
#         b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
#         rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
#         rst = self.w_out(rst)
#         rst = torch.nn.functional.relu(self.dropout(rst))
#         time_att_end = time.time()
#         print(f"subgraph src nodes: {b.num_src_nodes()}")
#         print(f"subgraph dst nodes: {b.num_dst_nodes()}")
#         print(f"subgraph nf shape: {b.srcdata['h'].shape}")
#         print(f"Time taken for computing Q, K, V: {time_q_end - time_q_start:.4f} seconds")
#         print(f"Time taken for attention computation: {time_att_end - time_att_start:.4f} seconds")

#         return self.layer_norm(rst)
          





class IdentityNormLayer(torch.nn.Module):

    def __init__(self, dim_out):
        super(IdentityNormLayer, self).__init__()
        self.norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b, Q_out=None, K_out=None, V_out=None, times=1):
        return self.norm(b.srcdata['h'])

class JODIETimeEmbedding(torch.nn.Module):

    def __init__(self, dim_out):
        super(JODIETimeEmbedding, self).__init__()
        self.dim_out = dim_out

        class NormalLinear(torch.nn.Linear):
        # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.time_emb = NormalLinear(1, dim_out)
    
    def forward(self, h, mem_ts, ts):
        time_diff = (ts - mem_ts) / (ts + 1)
        rst = h * (1 + self.time_emb(time_diff.unsqueeze(1)))
        return rst
            