import argparse
import os
from tqdm import tqdm  # 导入 tqdm

parser = argparse.ArgumentParser()
# ... [其他参数解析代码保持不变] ...

parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--rand_edge_features', type=int, default=0, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=0, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
args=parser.parse_args()



os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import time
import random
import dgl
import numpy as np
from modules import *
from sampler import *
from utils import *
from qkv import *
from layers import *
from pipeline import *
from mapping import *
import time
import psutil
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
ids_tensor = torch.tensor([0, 1])

def get_memory_info():
    process = psutil.Process(os.getpid())  # 获取当前进程的 pid
    mem_info = process.memory_info()  # 获取内存信息
    return mem_info.rss  # 返回常驻内存大小（单位为字节）
# set_seed(0)
process = psutil.Process(os.getpid())


fm = FeatureManager()

# node_feats = torch.from_numpy(fm.load_nfeatures(args.data))
# edge_feats = torch.from_numpy(fm.load_efeatures(args.data))

node_shape, edge_shape = fm.load_shapes(args.data)
node_dtype, edge_dtype = fm.load_dtypes(args.data)

#print(len(node_feats))
#print(len(edge_feats))


#node_feats, edge_feats = load_feat(args.data, args.rand_edge_features, args.rand_node_features)

# node_metadata = {
#     'dtype': str(node_feats.dtype),
#     'shape': list(node_feats.shape),
#     'device': str(node_feats.device)
# }
# edge_metadata = {
#     'dtype': str(edge_feats.dtype),
#     'shape': list(edge_feats.shape),
#     'device': str(edge_feats.device)
# }

# all_tensors_metadata = {
#     'node': node_metadata,
#     'edge': edge_metadata
# }
# filename = 'DATA/WIKITALK/metadata.yml'
# with open(filename, 'w') as f:
#     yaml.dump(all_tensors_metadata, f)
#print(node_feats.shape)
#print(edge_feats.shape)
g, df = load_graph(args.data)
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]

def get_inductive_links(df, train_edge_end, val_edge_end):
    train_df = df[:train_edge_end]
    test_df = df[val_edge_end:]

    total_node_set = set(np.unique(np.hstack([df['src'].values, df['dst'].values])))
    train_node_set = set(np.unique(np.hstack([train_df['src'].values, train_df['dst'].values])))
    new_node_set = total_node_set - train_node_set

    del total_node_set, train_node_set

    inductive_inds = []
    for index, (_, row) in enumerate(test_df.iterrows()):
        if row.src in new_node_set or row.dst in new_node_set:
            inductive_inds.append(val_edge_end + index)

    print('Inductive links', len(inductive_inds), len(test_df))
    return [i for i in range(val_edge_end)] + inductive_inds

if args.use_inductive:
    inductive_inds = get_inductive_links(df, train_edge_end, val_edge_end)
    df = df.iloc[inductive_inds]

gnn_dim_node = 0 if node_shape is None else node_shape[1]
gnn_dim_edge = 0 if edge_shape is None else edge_shape[1]


#fm.cache_features(args.data, gnn_dim_node, gnn_dim_edge)

combine_first = False
if 'combine_neighs' in train_param and train_param['combine_neighs']:
    combine_first = True
model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first).cuda()
mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None
creterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
    if mailbox is not None:
        mailbox.move_to_gpu()

sampler = None
if not ('no_sample' in sample_param and sample_param['no_sample']):
    # sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
    #                           sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
    #                           False, False,
    #                           sample_param['history'], float(sample_param['duration']))
    sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy'] == 'recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))

if args.use_inductive:
    test_df = df[val_edge_end:]
    inductive_nodes = set(test_df.src.values).union(test_df.dst.values)  # 修正这里应为 union dst
    print("inductive nodes", len(inductive_nodes))
    neg_link_sampler = NegLinkInductiveSampler(inductive_nodes)
else:
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)

best_ap = 0
best_e = 0
val_losses = list()
group_indexes = list()
group_indexes.append(np.array(df[:train_edge_end].index // train_param['batch_size']))
if 'reorder' in train_param:
    # random chunk scheduling
    reorder = train_param['reorder']
    group_idx = list()
    for i in range(reorder):
        group_idx += list(range(0 - i, reorder - i))
    group_idx = np.repeat(np.array(group_idx), train_param['batch_size'] // reorder)
    group_idx = np.tile(group_idx, train_edge_end // train_param['batch_size'] + 1)[:train_edge_end]
    group_indexes.append(group_indexes[0] + group_idx)
    base_idx = group_indexes[0]
    for i in range(1, train_param['reorder']):
        additional_idx = np.zeros(train_param['batch_size'] // train_param['reorder'] * i) - 1
        group_indexes.append(np.concatenate([additional_idx, base_idx])[:base_idx.shape[0]])

# 定义一个简单的批次生成器
def get_batches(df, batch_size):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i+batch_size]

def get_batch_by_id(df, batch_size, batch_id):
    start_idx = batch_id * batch_size
    end_idx = min((batch_id + 1) * batch_size, len(df))
    return df.iloc[start_idx:end_idx]

def sample(rows):
    # Sample for the next batch
    root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows) * neg_samples)]).astype(np.int32)
    ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
    time_sample_start = time.time()    
    ret = None
    if sampler is not None:
        if 'no_neg' in sample_param and sample_param['no_neg']:
            pos_root_end = len(rows) * 2 // 3
            sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
        else:
            sampler.sample(root_nodes, ts)
        ret = sampler.get_ret()
    # Gather (convert to graph blocks, etc.)
    
    if gnn_param['arch'] != 'identity':
        mfgs = to_dgl_blocks(ret, sample_param['history'])
    else:
        mfgs = node_to_dgl_blocks(root_nodes, ts)

    time_sample_end = time.time()
    print(f"Time taken for sample: {time_sample_end - time_sample_start:.4f} seconds")
    return mfgs, root_nodes, ts, ret

def gather(mfgs):
    time_gather_start = time.time()
    nid, eid = [], []
    for id, b in enumerate(mfgs):
        nid.append(b[0].srcdata['ID'].long().tolist())
        print(f"nid shape: {len(nid[id])}")
        if gnn_param['arch'] != 'identity':
            eid.append(b[0].edata['ID'].long().tolist())
            print(f"eid shape: {len(eid[id])}")
    # time_cache_start = time.time()
    # fm.cache_id_features(nid, eid, args.data, gnn_dim_node, gnn_dim_edge)
    # time_cache_end = time.time()
    # fm.clear_cache()
    time_gnode_start = time.time()
    # nf = fm.gather_node_uni(nid, args.data, gnn_dim_node)
    # nf = fm.gather_node_uni(nid, args.data, gnn_dim_node)
    # nf = fm.gather_node_uni(nid, args.data, gnn_dim_node)
    # nf = fm.gather_node_uni(nid, args.data, gnn_dim_node) 
    # time_gnode_end = time.time()
    # time_gedge_start = time.time()
    # if eid != []:
    #     ef = fm.gather_edge_uni(eid, args.data, gnn_dim_edge)
    # else:
    #     ef = []
    # time_gedge_end = time.time()
    nf, ef = [], []
    for id in nid:
        nf.append(torch.zeros(len(id), gnn_dim_node).cuda())

    for id in eid:
        ef.append(torch.zeros(len(id), gnn_dim_edge).cuda())
    # for id, b in enumerate(mfgs):
    #     b[0].srcdata['h'] = torch.tensor(nf[id]).cuda()
    #     if ef != []:
    #         b[0].edata['f'] = torch.tensor(ef[id]).cuda()
    for id, b in enumerate(mfgs):
        b[0].srcdata['h'] = nf[id]
        if ef != []:
            b[0].edata['f'] = ef[id]
    #mfgs = prepare_input(mfgs, node_feats, edge_feats, d=args.data)
    time_gather_end = time.time()

    time_update_start = time.time()
    if mailbox is not None:
        mailbox.prep_input_mails(mfgs[0])
        model.memory_updater(mfgs[0])
    time_update_end = time.time()

    print(f"Time taken for update: {time_update_end - time_update_start:.4f} seconds")
    # print(f"node gather: {time_gnode_end - time_gnode_start:.4f} seconds; edge gather: {time_gedge_end - time_gedge_start:.4f} seconds")
    
    print(f"Time taken for prepare: {time_gather_end - time_gather_start:.4f} seconds")
    return mfgs  # Return the result of gather for next batch

def infer_div(mfgs, block_size, times):
    for mfg in mfgs[0]:
        x = times
        block_size = block_size
        b = mfg
        qnf = b.srcdata['h'][:b.num_dst_nodes()]
        kvnf = b.srcdata['h'][b.num_dst_nodes():]
        ef = b.edata['f']
        kvtf = b.edata['dt']
        wq = model.ret_wq()
        wk = model.ret_wk()
        wv = model.ret_wv()
        dim_node = b.srcdata['h'].shape[1]
        dim_edge = b.edata['f'].shape[1]
                        
        qkv_processor = QKVProcessor(dim_node, dim_edge, gnn_param['dim_time'], gnn_param['dim_out'], b.num_dst_nodes(),
                                                wq, wk, wv, model.ret_time_zero(b), model.ret_time(b))
        time_q_start = time.time()
        Q_Out = torch.empty((b.num_dst_nodes(), gnn_param['dim_out']), device=b.device)
        K_Out = torch.empty((b.num_src_nodes() - b.num_dst_nodes(), gnn_param['dim_out']), device=b.device)
        V_Out = torch.empty((b.num_src_nodes() - b.num_dst_nodes(), gnn_param['dim_out']), device=b.device)
                        
        # 创建CUDA流
                        
        # 并行处理Q计算
        for i in range(0, b.num_dst_nodes(), block_size):
            begin = i
            end = min(i + block_size, b.num_dst_nodes())
            qnf_block = b.srcdata['h'][i:end]

            qnf_block = qnf_block.repeat(x, 1)
                            
            # 在每个流中异步计算
            Q_out_Block = qkv_processor.process_q(qnf_block, begin, end, x)
            Q_Out[i:end] = Q_out_Block[0:end-i]

        # 并行处理K和V计算
        for i in range(0, b.num_src_nodes() - b.num_dst_nodes(), block_size):
            begin = i
            end = min(i + block_size, b.num_src_nodes() - b.num_dst_nodes())
            kvnf_block = b.srcdata['h'][i:end]
            ef_block = b.edata['f'][i:end]

            kvnf_block = kvnf_block.repeat(x, 1)  # 沿第一个维度复制 x 次
            ef_block = ef_block.repeat(x, 1) 

            K_out_Block = qkv_processor.process_k(kvnf_block, ef_block, begin, end, x)
            K_Out[i:end] = K_out_Block[0:end-i]

            V_out_Block = qkv_processor.process_v(kvnf_block, ef_block, begin, end, x)
            V_Out[i:end] = V_out_Block[0:end-i]


        time_q_end = time.time()
        print(f"Time taken for qkv computation: {time_q_end - time_q_start:.4f} seconds")
        return Q_Out, K_Out, V_Out    


def compute_aps(pred_pos, pred_neg, root_nodes, ts, ret):
    #pred_pos, pred_neg = model(mfgs, neg_samples=neg_samples)
    del_total_loss = 0
    del_total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
    del_total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
    y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
    y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
    aps = average_precision_score(y_true, y_pred)
    if neg_samples > 1:
        aucs_mrrs = torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float)
    else:
        aucs_mrrs = roc_auc_score(y_true, y_pred)
    if mailbox is not None:
        eid = rows['Unnamed: 0'].values
        time_gather_start = time.time() 
        # mem_edge_feats = torch.tensor(fm.gather_node_uni(eid, args.data, gnn_dim_node)[0]).cuda()
        mem_edge_feats = torch.zeros(eid.shape[0], gnn_dim_node)
        time_gather_end = time.time() 
        print(f"Time taken for another gather: {time_gather_end - time_gather_start:.4f} seconds")
        block = None
        time_update_start = time.time() 
        if memory_param['deliver_to'] == 'neighbors':
            block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
        mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, ts, mem_edge_feats, block, neg_samples=neg_samples)
        mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)
        time_update_end = time.time() 
        print(f"Time taken for another update: {time_update_end - time_update_start:.4f} seconds")
    
    return aps, aucs_mrrs, del_total_loss

path_saver = 'models/{}_{}.pkl'.format(args.model_name, args.config)
#path_saver = 'models/GDELT_TGN.pkl'.format(args.model_name)
#path_saver = 'models/REDDIT_1734609162.7953107.pkl'
#path_saver = 'models/WIKI_TGN.pkl'
print('Loading model at epoch {}...'.format(best_e))
# state_dict = torch.load(path_saver)
# print(state_dict.keys())
# print(model.state_dict().keys())
if os.path.exists(path_saver):
    print(f'Loading model from {path_saver}...')
    model.load_state_dict(torch.load(path_saver))
else:
    print(f'No model found at {path_saver}, using randomly initialized weights.')

neg_samples = args.eval_neg_samples
# for e in range(train_param['epoch']):
# print('Epoch {:d}:'.format(e))
eval_df = df[val_edge_end:]
shuffled_df = eval_df.sample(frac=1).reset_index(drop=True)
num_batches = len(shuffled_df) // train_param['batch_size'] + 1

aps = list()
aucs_mrrs = list()
correct = 0
total = 0
block_size, x = 10, 1
block_div = False
with torch.no_grad():
    total_loss = 0
    with tqdm(total=150, desc=f'Epoch {1}/{train_param["epoch"]}', unit='batch') as pbar:#total=num_batches
        for idx, rows in enumerate(get_batches(shuffled_df, train_param['batch_size'])):
            if idx >= 150:
                break
            time_batch_start = time.time()
            
            mfgs, root_nodes, ts, ret = sample(rows)
            mfgs = gather(mfgs)
            
            Q_Out , K_Out, V_Out= None, None, None

            #div
            if block_div:
                Q_Out , K_Out, V_Out = infer_div(mfgs, block_size, x)
            pred_pos, pred_neg = model(mfgs, neg_samples=neg_samples, Q_out=Q_Out, K_out=K_Out, V_out=V_Out, times = x)
            print(f"output1: {pred_pos.shape}")
            print(f"output2: {pred_neg.shape}")
            ap, auc, del_total_loss = compute_aps(pred_pos, pred_neg, root_nodes, ts, ret)
            aps.append(ap)
            aucs_mrrs.append(auc)
            total_loss += del_total_loss

            time_batch_end = time.time()

            print(f"Time taken for batch {idx} computation: {time_batch_end - time_batch_start:.4f} seconds")
            pbar.update(1)
ap = float(torch.tensor(aps).mean())
if neg_samples > 1:
    auc_mrr = float(torch.cat(aucs_mrrs).mean())
else:
    auc_mrr = float(torch.tensor(aucs_mrrs).mean())
#accuracy = correct / total
print(f"peak memory: {process.memory_info().rss / (1024 * 1024):.2f}")

#print(f'\tInference Accuracy: {accuracy:.4f}')
print(f'\ttest AP: {ap:.4f}  test AUC: {auc_mrr:.4f}')

