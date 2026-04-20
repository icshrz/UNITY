import argparse
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True, help='dataset name')
args = parser.parse_args()


def gen_edges_wiki_talk():
    df = pd.read_csv('DATA/WIKITALK/wiki-talk-temporal.txt',
         sep=' ', header=None, names=['src', 'dst', 'time'],
         dtype={'src': np.int32, 'dst': np.int32, 'time': np.float32})

    num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
    num_edges = df.shape[0]
    train_end = int(np.ceil(num_edges * 0.70))
    valid_end = int(np.ceil(num_edges * 0.85))
    print('num_nodes:', num_nodes)
    print('num_edges:', num_edges)
    print('train_end:', train_end)
    print('valid_end:', valid_end)

    df['int_roll'] = np.zeros(num_edges, dtype=np.int32)
    ext_roll = np.zeros(num_edges, dtype=np.int32)
    ext_roll[train_end:] = 1
    ext_roll[valid_end:] = 2
    df['ext_roll'] = ext_roll

    df.to_csv('DATA/WIKITALK/edges.csv')
    # 生成边特征 (Edge Features)
    edge_feats = torch.randn(num_edges, 128, dtype=torch.float32)
    torch.save(edge_feats, 'DATA/WIKITALK/edge_features.pt')

    # 生成节点特征 (Node Features)
    node_feats = torch.randn(num_nodes, 128, dtype=torch.float32)
    torch.save(node_feats, 'DATA/WIKITALK/node_features.pt')

    # 生成稀疏邻接矩阵 (利用边列表)
    row = df['src'].values
    col = df['dst'].values
    data = np.ones(num_edges)

    adj_matrix = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    # 保存稀疏邻接矩阵为npz格式
    np.savez_compressed('DATA/WIKITALK/ext_full.npz', adj_matrix)
    np.savez_compressed('DATA/WIKITALK/int_full.npz', adj_matrix)
    np.savez_compressed('DATA/WIKITALK/int_train.npz', adj_matrix[:train_end, :train_end])

    print("文件生成完毕，数据集已成功构建!")


if args.data == 'wiki-talk':
    gen_edges_wiki_talk()
else:
    print('not handling dataset:', args.data)
