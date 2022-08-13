import dgl
import numpy as np
import pickle
import random
import torch
from sklearn.metrics import f1_score
from metric import ndcg, spearman_sci
import pdb


def convert_to_gpu(*data, device):
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    return tuple(res)


def set_random_seed(seed=0):
    """
    set random seed.
    :param seed: int, random seed to use
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_model(model, model_path):
    """Load the model.
    :param model: model
    :param model_path: model path
    """
    print(f"load model {model_path}")
    model.load_state_dict(torch.load(model_path))


def count_parameters_in_KB(model):
    """
    count the size of trainable parameters in model (KB)
    :param model: model
    :return:
    """
    param_num = np.sum(np.prod(v.size()) for v in model.parameters()) / 1e3
    return param_num

# 训练集的评估，返回spearman和ndcg
def get_rank_metrics(predicts, labels, NDCG_k, spearman=False):
    """
    calculate NDCG@k metric
    :param predicts: Tensor, shape (N, 1)
    :param labels: Tensor, shape (N, 1)
    :return:
    """
    if spearman:
        return ndcg(labels, predicts, NDCG_k), spearman_sci(labels, predicts)
    return ndcg(labels, predicts, NDCG_k)

# 验证集和测试集的评估，返回损失、spearman和ndcg
def rank_evaluate(predicts, labels, NDCG_k, loss_func, spearman=False):
    """
    evaluation used for validation or test
    :param predicts: Tensor, shape (N, 1)
    :param labels: Tensor, shape (N, 1)
    :param loss_func: loss function
    :return:
    """
    with torch.no_grad():
        loss = loss_func(predicts, labels)
    if spearman:
        ndcg_score, spear_score = get_rank_metrics(predicts, labels, NDCG_k, spearman)
        return loss, ndcg_score, spear_score
    else:
        ndcg_score = get_rank_metrics(predicts, labels, NDCG_k, spearman)
        return loss, ndcg_score


def load_fb15k_rel_data(data_path, cross_validation_shift=0, dataset_name='FB15k_rel'):
    """
    load fb15k data
    :param data_path: str, data file path
    :param cross_validation_shift: int, shift of data split
    :return:
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)  # data中包含节点的边、谓词类型、标签、节点特征、无效掩码

    # edge list
    edges = data['edges']   # 节点的边，两个tensor，表示对应位置的两个节点间有边
    labels = data['labels']   # 节点标签,有的可能为0，此时节点无效，无效掩码为1

    if 'semantic' in dataset_name:
        node_feats = pickle.load(open('./datasets/fb_lang.pk', 'rb'))
        node_feats = torch.from_numpy(node_feats).float()              # 把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
    elif 'concat' in dataset_name:          # 将data['features']和fb_lang进行拼接
        node_feat1 = data['features']
        node_feat2 = pickle.load(open('./datasets/fb_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
        node_feats = torch.cat([node_feat1, node_feat2], dim=1)
    elif 'two' in dataset_name:           # 最后有data['features']和fb_lang两种特征
        node_feat1 = data['features']
        node_feat2 = pickle.load(open('../datasets/fb_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
    else:
        node_feats = data['features']    # 只有data['features']
    invalid_masks = data['invalid_masks']    # 无效掩码
    edge_types = data['edge_types']     # 谓词类型，共1345个
    rel_num = (max(edge_types) + 1).item()   # 关系数量，最大边类型数（从0开始）加一

    # construct a heterogeneous graph 构造异构图
    hg = dgl.graph(edges)

    # generate edge norm
    g = hg.local_var()     # 返回一个图对象以在局部函数作用域中使用
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()   # 节点的入度
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0    # 显示哪些元素为正或负无穷大，把它置零
    node_norm = torch.from_numpy(norm).view(-1, 1)
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    edge_norm = g.edata['norm']

    # log transform for labels
    labels = torch.log(1 + labels)   # 对节点重要性的标签做一个转换

    # split dataset
    float_mask = np.ones(hg.number_of_nodes()) * -1.  # 对角线为-1.0
    label_mask = (invalid_masks == 0)     # 有效的节点标签的节点idx
    float_mask[label_mask] = np.random.RandomState(seed=0).permutation(np.linspace(0, 1, label_mask.sum()))  # 有效的节点idx

    # 1:2:7
    if cross_validation_shift == 0:
        val_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
        train_idx = np.where((0.2 < float_mask) & (float_mask <= 0.3))[0]
        test_idx = np.where(float_mask > 0.3)[0]
    elif cross_validation_shift == 1:
        val_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
        train_idx = np.where((0.4 < float_mask) & (float_mask <= 0.5))[0]
        test_idx = np.where((float_mask > 0.5) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    elif cross_validation_shift == 2:
        val_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
        train_idx = np.where((0.6 < float_mask) & (float_mask <= 0.7))[0]
        test_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    elif cross_validation_shift == 3:
        val_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
        train_idx = np.where((0.8 < float_mask) & (float_mask <= 0.9))[0]
        test_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    elif cross_validation_shift == 4:
        val_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
        train_idx = np.where((0 <= float_mask) & (float_mask <= 0.1))[0]
        test_idx = np.where((0.1 < float_mask) & (float_mask < 0.8))[0]
    else:
        raise ValueError(f'wrong value for parameter {cross_validation_shift}')

    # 7:1:2
    # if cross_validation_shift == 0:
    #     test_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
    #     val_idx = np.where((0.2 < float_mask) & (float_mask <= 0.3))[0]
    #     train_idx = np.where(float_mask > 0.3)[0]
    # elif cross_validation_shift == 1:
    #     test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
    #     val_idx = np.where((0.4 < float_mask) & (float_mask <= 0.5))[0]
    #     train_idx = np.where((float_mask > 0.5) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    # elif cross_validation_shift == 2:
    #     test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
    #     val_idx = np.where((0.6 < float_mask) & (float_mask <= 0.7))[0]
    #     train_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    # elif cross_validation_shift == 3:
    #     test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
    #     val_idx = np.where((0.8 < float_mask) & (float_mask <= 0.9))[0]
    #     train_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    # elif cross_validation_shift == 4:
    #     test_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
    #     val_idx = np.where((0 <= float_mask) & (float_mask <= 0.1))[0]
    #     train_idx = np.where((0.1 < float_mask) & (float_mask < 0.8))[0]
    # else:
    #     raise ValueError(f'wrong value for parameter {cross_validation_shift}')


    # 6:2:2
    # if cross_validation_shift == 0:
    #     test_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
    #     val_idx = np.where((0.2 < float_mask) & (float_mask <= 0.4))[0]
    #     train_idx = np.where(float_mask > 0.4)[0]
    # elif cross_validation_shift == 1:
    #     test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
    #     val_idx = np.where((0.4 < float_mask) & (float_mask <= 0.6))[0]
    #     train_idx = np.where((float_mask > 0.6) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    # elif cross_validation_shift == 2:
    #     test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
    #     val_idx = np.where((0.6 < float_mask) & (float_mask <= 0.8))[0]
    #     train_idx = np.where((float_mask > 0.8) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    # elif cross_validation_shift == 3:
    #     test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
    #     val_idx = np.where((0.8 < float_mask) & (float_mask <= 1.0))[0]
    #     train_idx = np.where((float_mask > 1.0) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    # elif cross_validation_shift == 4:
    #     test_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
    #     val_idx = np.where((0 <= float_mask) & (float_mask <= 0.2))[0]
    #     train_idx = np.where((0.2 < float_mask) & (float_mask < 0.8))[0]
    # else:
    #     raise ValueError(f'wrong value for parameter {cross_validation_shift}')

    # # 5:2:3
    # if cross_validation_shift == 0:
    #     test_idx = np.where((0. <= float_mask) & (float_mask <= 0.3))[0]
    #     val_idx = np.where((0.3 < float_mask) & (float_mask <= 0.5))[0]
    #     train_idx = np.where(float_mask > 0.5)[0]
    # elif cross_validation_shift == 1:
    #     test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.5))[0]
    #     val_idx = np.where((0.5 < float_mask) & (float_mask <= 0.7))[0]
    #     train_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    # elif cross_validation_shift == 2:
    #     test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.7))[0]
    #     val_idx = np.where((0.7 < float_mask) & (float_mask <= 0.9))[0]
    #     train_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    # elif cross_validation_shift == 3:
    #     test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.9))[0]
    #     val_idx = np.where((0.9 < float_mask) & ((0 <= float_mask) & (float_mask < 0.1)))[0]
    #     train_idx = np.where(((0.1 <= float_mask) & (float_mask < 0.6)))[0]
    # elif cross_validation_shift == 4:
    #     test_idx = np.where((0.8 <= float_mask) & ((0 <= float_mask) & (float_mask < 0.1)))[0]
    #     val_idx = np.where((0.1 <= float_mask) & (float_mask <= 0.3))[0]
    #     train_idx = np.where((0.3 < float_mask) & (float_mask < 0.8))[0]
    # else:
    #     raise ValueError(f'wrong value for parameter {cross_validation_shift}')

    # 3:4:3
    # if cross_validation_shift == 0:
    #     test_idx = np.where((0. <= float_mask) & (float_mask <= 0.3))[0]
    #     val_idx = np.where((0.3 < float_mask) & (float_mask <= 0.7))[0]
    #     train_idx = np.where(float_mask > 0.7)[0]
    # elif cross_validation_shift == 1:
    #     test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.5))[0]
    #     val_idx = np.where((0.5 < float_mask) & (float_mask <= 0.9))[0]
    #     train_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    # elif cross_validation_shift == 2:
    #     test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.7))[0]
    #     val_idx = np.where((0.7 < float_mask) | ((0 <= float_mask) & (float_mask < 0.1)))[0]
    #     train_idx = np.where((0.1 <= float_mask) & (float_mask <= 0.4))[0]
    # elif cross_validation_shift == 3:
    #     test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.9))[0]
    #     val_idx = np.where((0.9 < float_mask) & ((0 <= float_mask) & (float_mask < 0.3)))[0]
    #     train_idx = np.where(((0.3 <= float_mask) & (float_mask < 0.6)))[0]
    # elif cross_validation_shift == 4:
    #     test_idx = np.where((0.8 <= float_mask) & ((0 <= float_mask) & (float_mask < 0.1)))[0]
    #     val_idx = np.where((0.1 <= float_mask) & (float_mask <= 0.5))[0]
    #     train_idx = np.where((0.5 < float_mask) & (float_mask < 0.8))[0]
    # else:
    #     raise ValueError(f'wrong value for parameter {cross_validation_shift}')

    print(len(test_idx), len(val_idx), len(train_idx))
    if 'two' in dataset_name:
        return hg, edge_types, edge_norm, rel_num, node_feat1, node_feat2, labels, train_idx, val_idx, test_idx
    return hg, edge_types, edge_norm, rel_num, node_feats, labels, train_idx, val_idx, test_idx


def load_imdb_s_rel_data(data_path, cross_validation_shift=0, dataset_name='IMDB_S_rel'):
    """
    load imdb rel data
    :param data_path: str, data file path
    :param cross_validation_shift: int, shift of data split
    :return:
    """

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    if 'semantic' in dataset_name:
        node_feats = pickle.load(open('../datasets/imdb_s_lang.pk', 'rb'))
        node_feats = torch.from_numpy(node_feats).float()
    elif 'two' in dataset_name:
        node_feat1 = torch.from_numpy(pickle.load(open('../datasets/imdb_s_node2vec.pk', 'rb')))
        node_feat2 = pickle.load(open('../datasets/imdb_s_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
    elif 'concat' in dataset_name:
        node_feat1 = torch.from_numpy(pickle.load(open('../datasets/imdb_s_node2vec.pk', 'rb')))
        node_feat2 = pickle.load(open('../datasets/imdb_s_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
        node_feats = torch.cat([node_feat1, node_feat2], 1)
    else:
        node_feats = torch.from_numpy(pickle.load(open('../datasets/imdb_s_node2vec.pk', 'rb')))

    # edge list
    edges = data['edges']
    labels = data['labels'].float()
    invalid_masks = data['invalid_masks']
    edge_types = data['edge_types']
    # rel_num = (max(edge_types) + 1).item()
    rel_num = 30

    # construct a heterogeneous graph
    hg = dgl.graph(edges)

    # log transform for labels
    labels = torch.log(1 + labels)

    # split dataset
    float_mask = np.ones(hg.number_of_nodes()) * -1.
    label_mask = (invalid_masks == 0)
    float_mask[label_mask] = np.random.RandomState(seed=0).permutation(np.linspace(0, 1, label_mask.sum()))

    # 1:2:7
    if cross_validation_shift == 0:
        val_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
        train_idx = np.where((0.2 < float_mask) & (float_mask <= 0.3))[0]
        test_idx = np.where(float_mask > 0.3)[0]
    elif cross_validation_shift == 1:
        val_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
        train_idx = np.where((0.4 < float_mask) & (float_mask <= 0.5))[0]
        test_idx = np.where((float_mask > 0.5) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    elif cross_validation_shift == 2:
        val_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
        train_idx = np.where((0.6 < float_mask) & (float_mask <= 0.7))[0]
        test_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    elif cross_validation_shift == 3:
        val_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
        train_idx = np.where((0.8 < float_mask) & (float_mask <= 0.9))[0]
        test_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    elif cross_validation_shift == 4:
        val_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
        train_idx = np.where((0 <= float_mask) & (float_mask <= 0.1))[0]
        test_idx = np.where((0.1 < float_mask) & (float_mask < 0.8))[0]
    else:
        raise ValueError(f'wrong value for parameter {cross_validation_shift}')

    # 7:1:2
    # if cross_validation_shift == 0:
    #     test_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
    #     val_idx = np.where((0.2 < float_mask) & (float_mask <= 0.3))[0]
    #     train_idx = np.where(float_mask > 0.3)[0]
    # elif cross_validation_shift == 1:
    #     test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
    #     val_idx = np.where((0.4 < float_mask) & (float_mask <= 0.5))[0]
    #     train_idx = np.where((float_mask > 0.5) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    # elif cross_validation_shift == 2:
    #     test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
    #     val_idx = np.where((0.6 < float_mask) & (float_mask <= 0.7))[0]
    #     train_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    # elif cross_validation_shift == 3:
    #     test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
    #     val_idx = np.where((0.8 < float_mask) & (float_mask <= 0.9))[0]
    #     train_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    # elif cross_validation_shift == 4:
    #     test_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
    #     val_idx = np.where((0 <= float_mask) & (float_mask <= 0.1))[0]
    #     train_idx = np.where((0.1 < float_mask) & (float_mask < 0.8))[0]
    # else:
    #     raise ValueError(f'wrong value for parameter {cross_validation_shift}')

    # print(len(test_idx), len(val_idx), len(train_idx))
    if 'two' in dataset_name:
        return hg, edge_types, None, rel_num, node_feat1, node_feat2, labels, train_idx, val_idx, test_idx
    return hg, edge_types, None, rel_num, node_feats, labels, train_idx, val_idx, test_idx


def load_tmdb_rel_data(data_path, cross_validation_shift=0, dataset_name='TMDB_rel'):
    """
    load tmdb rel data
    :param data_path: str, data file path
    :param cross_validation_shift: int, shift of data split
    :return:
    """

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # edge list
    edges = data['edges']
    labels = data['labels'].float()
    invalid_masks = data['invalid_masks']
    edge_types = data['edge_types']
    # rel_num = (max(edge_types) + 1).item()
    rel_num = 34

    # node_feat1是结构信息，node_feat2是语义信息，node_feats是两者结合
    if 'semantic' in dataset_name:
        node_feats = pickle.load(open('./datasets/tmdb_lang.pk', 'rb'))
        node_feats = torch.from_numpy(node_feats).float()
    elif 'two' in dataset_name:
        node_feat1 = data['features']
        node_feat2 = pickle.load(open('../datasets/tmdb_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
    elif 'concat' in dataset_name:
        node_feat1 = data['features']
        node_feat2 = pickle.load(open('./datasets/tmdb_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
        node_feats = torch.cat([node_feat1, node_feat2], 1)
    else:
        node_feats = data['features']

    # construct a heterogeneous graph
    hg = dgl.graph(edges)

    # log transform for labels
    labels = torch.log(1 + labels)

    # split dataset
    float_mask = np.ones(hg.number_of_nodes()) * -1.
    label_mask = (invalid_masks == 0)
    float_mask[label_mask] = np.random.RandomState(seed=0).permutation(np.linspace(0, 1, label_mask.sum()))

    # 7:1:2
    # if cross_validation_shift == 0:
    #     test_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
    #     val_idx = np.where((0.2 < float_mask) & (float_mask <= 0.3))[0]
    #     train_idx = np.where(float_mask > 0.3)[0]
    # elif cross_validation_shift == 1:
    #     test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
    #     val_idx = np.where((0.4 < float_mask) & (float_mask <= 0.5))[0]
    #     train_idx = np.where((float_mask > 0.5) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    # elif cross_validation_shift == 2:
    #     test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
    #     val_idx = np.where((0.6 < float_mask) & (float_mask <= 0.7))[0]
    #     train_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    # elif cross_validation_shift == 3:
    #     test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
    #     val_idx = np.where((0.8 < float_mask) & (float_mask <= 0.9))[0]
    #     train_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    # elif cross_validation_shift == 4:
    #     test_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
    #     val_idx = np.where((0 <= float_mask) & (float_mask <= 0.1))[0]
    #     train_idx = np.where((0.1 < float_mask) & (float_mask < 0.8))[0]
    # else:
    #     raise ValueError(f'wrong value for parameter {cross_validation_shift}')

    # 6:2:2
    # if cross_validation_shift == 0:
    #     test_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
    #     val_idx = np.where((0.2 < float_mask) & (float_mask <= 0.4))[0]
    #     train_idx = np.where(float_mask > 0.4)[0]
    # elif cross_validation_shift == 1:
    #     test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
    #     val_idx = np.where((0.4 < float_mask) & (float_mask <= 0.6))[0]
    #     train_idx = np.where((float_mask > 0.6) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    # elif cross_validation_shift == 2:
    #     test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
    #     val_idx = np.where((0.6 < float_mask) & (float_mask <= 0.8))[0]
    #     train_idx = np.where((float_mask > 0.8) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    # elif cross_validation_shift == 3:
    #     test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
    #     val_idx = np.where((0.8 < float_mask) & (float_mask <= 1.0))[0]
    #     train_idx = np.where(((0 <= float_mask) & (float_mask < 0.6)))[0]
    # elif cross_validation_shift == 4:
    #     test_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
    #     val_idx = np.where((0 <= float_mask) & (float_mask <= 0.2))[0]
    #     train_idx = np.where((0.2 < float_mask) & (float_mask < 0.8))[0]
    # else:
    #     raise ValueError(f'wrong value for parameter {cross_validation_shift}')

    # # 5:2:3
    # if cross_validation_shift == 0:
    #     test_idx = np.where((0. <= float_mask) & (float_mask <= 0.3))[0]
    #     val_idx = np.where((0.3 < float_mask) & (float_mask <= 0.5))[0]
    #     train_idx = np.where(float_mask > 0.5)[0]
    # elif cross_validation_shift == 1:
    #     test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.5))[0]
    #     val_idx = np.where((0.5 < float_mask) & (float_mask <= 0.7))[0]
    #     train_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    # elif cross_validation_shift == 2:
    #     test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.7))[0]
    #     val_idx = np.where((0.7 < float_mask) & (float_mask <= 0.9))[0]
    #     train_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    # elif cross_validation_shift == 3:
    #     test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.9))[0]
    #     val_idx = np.where((0.9 < float_mask) | ((0 <= float_mask) & (float_mask < 0.1)))[0]
    #     train_idx = np.where(((0.1 <= float_mask) & (float_mask < 0.6)))[0]
    # elif cross_validation_shift == 4:
    #     test_idx = np.where((0.8 <= float_mask) | ((0 <= float_mask) & (float_mask < 0.1)))[0]
    #     val_idx = np.where((0.1 <= float_mask) & (float_mask <= 0.3))[0]
    #     train_idx = np.where((0.3 < float_mask) & (float_mask < 0.8))[0]
    # else:
    #     raise ValueError(f'wrong value for parameter {cross_validation_shift}')

    # 4:3:3
    # if cross_validation_shift == 0:
    #     val_idx = np.where((0. <= float_mask) & (float_mask <= 0.3))[0]
    #     train_idx = np.where((0.3 < float_mask) & (float_mask <= 0.7))[0]
    #     test_idx = np.where(float_mask > 0.7)[0]
    # elif cross_validation_shift == 1:
    #     val_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.5))[0]
    #     train_idx = np.where((0.5 < float_mask) & (float_mask <= 0.9))[0]
    #     test_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    # elif cross_validation_shift == 2:
    #     val_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.7))[0]
    #     train_idx = np.where((0.7 < float_mask) | ((0 <= float_mask) & (float_mask < 0.1)))[0]
    #     test_idx = np.where((0.1 <= float_mask) & (float_mask <= 0.4))[0]
    # elif cross_validation_shift == 3:
    #     val_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.9))[0]
    #     train_idx = np.where((0.9 < float_mask) | ((0 <= float_mask) & (float_mask < 0.3)))[0]
    #     test_idx = np.where(((0.3 <= float_mask) & (float_mask < 0.6)))[0]
    # elif cross_validation_shift == 4:
    #     val_idx = np.where((0.8 <= float_mask) | ((0 <= float_mask) & (float_mask < 0.1)))[0]
    #     train_idx = np.where((0.1 <= float_mask) & (float_mask <= 0.5))[0]
    #     test_idx = np.where((0.5 < float_mask) & (float_mask < 0.8))[0]
    # else:
    #     raise ValueError(f'wrong value for parameter {cross_validation_shift}')


    # 3:4:3
    # if cross_validation_shift == 0:
    #     test_idx = np.where((0. <= float_mask) & (float_mask <= 0.3))[0]
    #     val_idx = np.where((0.3 < float_mask) & (float_mask <= 0.7))[0]
    #     train_idx = np.where(float_mask > 0.7)[0]
    # elif cross_validation_shift == 1:
    #     test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.5))[0]
    #     val_idx = np.where((0.5 < float_mask) & (float_mask <= 0.9))[0]
    #     train_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    # elif cross_validation_shift == 2:
    #     test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.7))[0]
    #     val_idx = np.where((0.7 < float_mask) | ((0 <= float_mask) & (float_mask < 0.1)))[0]
    #     train_idx = np.where((0.1 <= float_mask) & (float_mask <= 0.4))[0]
    # elif cross_validation_shift == 3:
    #     test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.9))[0]
    #     val_idx = np.where((0.9 < float_mask) | ((0 <= float_mask) & (float_mask < 0.3)))[0]
    #     train_idx = np.where(((0.3 <= float_mask) & (float_mask < 0.6)))[0]
    # elif cross_validation_shift == 4:
    #     test_idx = np.where((0.8 <= float_mask) | ((0 <= float_mask) & (float_mask < 0.1)))[0]
    #     val_idx = np.where((0.1 <= float_mask) & (float_mask <= 0.5))[0]
    #     train_idx = np.where((0.5 < float_mask) & (float_mask < 0.8))[0]
    # else:
    #     raise ValueError(f'wrong value for parameter {cross_validation_shift}')


    # 2:1:7
    # if cross_validation_shift == 0:
    #     train_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
    #     val_idx = np.where((0.2 < float_mask) & (float_mask <= 0.3))[0]
    #     test_idx = np.where(float_mask > 0.3)[0]
    # elif cross_validation_shift == 1:
    #     train_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
    #     val_idx = np.where((0.4 < float_mask) & (float_mask <= 0.5))[0]
    #     test_idx = np.where((float_mask > 0.5) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    # elif cross_validation_shift == 2:
    #     train_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
    #     val_idx = np.where((0.6 < float_mask) & (float_mask <= 0.7))[0]
    #     test_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    # elif cross_validation_shift == 3:
    #     train_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
    #     val_idx = np.where((0.8 < float_mask) & (float_mask <= 0.9))[0]
    #     test_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    # elif cross_validation_shift == 4:
    #     train_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
    #     val_idx = np.where((0 <= float_mask) & (float_mask <= 0.1))[0]
    #     test_idx = np.where((0.1 < float_mask) & (float_mask < 0.8))[0]
    # else:
    #     raise ValueError(f'wrong value for parameter {cross_validation_shift}')

    # 1:2:7
    if cross_validation_shift == 0:
        val_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
        train_idx = np.where((0.2 < float_mask) & (float_mask <= 0.3))[0]
        test_idx = np.where(float_mask > 0.3)[0]
    elif cross_validation_shift == 1:
        val_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
        train_idx = np.where((0.4 < float_mask) & (float_mask <= 0.5))[0]
        test_idx = np.where((float_mask > 0.5) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    elif cross_validation_shift == 2:
        val_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
        train_idx = np.where((0.6 < float_mask) & (float_mask <= 0.7))[0]
        test_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    elif cross_validation_shift == 3:
        val_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
        train_idx = np.where((0.8 < float_mask) & (float_mask <= 0.9))[0]
        test_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    elif cross_validation_shift == 4:
        val_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
        train_idx = np.where((0 <= float_mask) & (float_mask <= 0.1))[0]
        test_idx = np.where((0.1 < float_mask) & (float_mask < 0.8))[0]
    else:
        raise ValueError(f'wrong value for parameter {cross_validation_shift}')

    # print(len(test_idx), len(val_idx), len(train_idx))

    # generate edge norm
    g = hg.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    node_norm = torch.from_numpy(norm).view(-1, 1)
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    edge_norm = g.edata['norm']

    if 'two' in dataset_name:
        return hg, edge_types, edge_norm, rel_num, node_feat1, node_feat2, labels, train_idx, val_idx, test_idx
    return hg, edge_types, edge_norm, rel_num, node_feats, labels, train_idx, val_idx, test_idx


def load_data(data_path, dataset_name, cross_validation_shift=0):
    """
    load dataset based on the input dataset name
    :param data_path: str, data file path
    :param dataset_name: dataset name
    :param cross_validation_shift: int, shift of data split
    :return:
    """

    if dataset_name.startswith('FB15k'):
        return load_fb15k_rel_data(data_path=data_path, cross_validation_shift=cross_validation_shift, dataset_name=dataset_name)
    elif dataset_name.startswith('IMDB_S'):
        return load_imdb_s_rel_data(data_path, cross_validation_shift, dataset_name)
    elif dataset_name.startswith('TMDB'):
        return load_tmdb_rel_data(data_path, cross_validation_shift, dataset_name)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset_name))


def get_centrality(graph):
    g = graph.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    theta = 1e-4
    centrality = torch.log(in_deg + theta)
    return centrality