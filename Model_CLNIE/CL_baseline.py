from sklearn.model_selection import train_test_split
from utils import set_random_seed, load_data, get_rank_metrics, rank_evaluate, get_centrality
from sklearn.neural_network import MLPRegressor
import torch
import numpy as np
from metric import overlap

from sklearn.linear_model import LinearRegression
import networkx as nx
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree, svm


import pandas as pd


def baseline(data_path, dataset):
    lr_ndcg_scores = []
    lr_spearmans = []
    lr_overlaps = []

    rf_ndcg_scores = []
    rf_spearmans = []
    rf_overlaps = []

    mlp_ndcg_scores = []
    mlp_spearmans = []
    mlp_overlaps = []

    for cross_id in range(5):
        g, edge_types, _, rel_num, node_feats, labels, train_idx, val_idx, test_idx = \
        load_data(data_path, dataset, cross_id)

        x_train = node_feats[train_idx]
        y_train = labels[train_idx]
        x_test = node_feats[test_idx]
        y_test = labels[test_idx]

        # lr
        lr = LinearRegression()
        # print(y_train.type())
        # print(y_train)
        lr.fit(x_train, y_train.int())
        predict_y_lr = torch.from_numpy(lr.predict(x_test))
        lr_test_ndcg, lr_test_spearman = \
            get_rank_metrics(predict_y_lr, y_test, 100, spearman=True)
        lr_test_overlap = overlap(y_test, predict_y_lr, 100)
        print("Test NDCG {:.4f} | Test Spearman {:.4f} | Test Overlap {:.4f}".
              format(lr_test_ndcg, lr_test_spearman, lr_test_overlap))

        lr_ndcg_scores.append(lr_test_ndcg)
        lr_spearmans.append(lr_test_spearman)
        lr_overlaps.append(lr_test_overlap)

        # RF
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train.int())
        predict_y_rf = torch.from_numpy(rf.predict(x_test))
        rf_test_ndcg, rf_test_spearman = \
            get_rank_metrics(predict_y_rf, y_test, 100, spearman=True)
        rf_test_overlap = overlap(y_test, predict_y_rf, 100)
        print("Test NDCG {:.4f} | Test Spearman {:.4f} | Test Overlap {:.4f}".
              format(rf_test_ndcg, rf_test_spearman, rf_test_overlap))

        rf_ndcg_scores.append(rf_test_ndcg)
        rf_spearmans.append(rf_test_spearman)
        rf_overlaps.append(rf_test_overlap)


        # mlp
        mlp = MLPRegressor()
        mlp.fit(x_train, y_train)
        predict_y_mlp = torch.from_numpy(mlp.predict(x_test))
        mlp_test_ndcg, mlp_test_spearman = \
            get_rank_metrics(predict_y_mlp, y_test, 100, spearman=True)
        mlp_test_overlap = overlap(y_test, predict_y_mlp, 100)
        print("Test NDCG {:.4f} | Test Spearman {:.4f} | Test Overlap {:.4f}".
              format(mlp_test_ndcg, mlp_test_spearman, mlp_test_overlap))

        mlp_ndcg_scores.append(mlp_test_ndcg)
        mlp_spearmans.append(mlp_test_spearman)
        mlp_overlaps.append(mlp_test_overlap)

    print('LR方法结果：')
    lr_ndcg_scores = np.array(lr_ndcg_scores)
    print('ndcg: ', lr_ndcg_scores, lr_ndcg_scores.mean(), np.std(lr_ndcg_scores))
    lr_spearmans = np.array(lr_spearmans)
    print('spearmans: ', lr_spearmans, lr_spearmans.mean(), np.std(lr_spearmans))
    lr_overlaps = np.array(lr_overlaps)
    print('over: ',lr_overlaps, lr_overlaps.mean(), np.std(lr_overlaps))

    print('RF方法结果：')
    rf_ndcg_scores = np.array(rf_ndcg_scores)
    print('ndcg: ', rf_ndcg_scores, rf_ndcg_scores.mean(), np.std(rf_ndcg_scores))
    rf_spearmans = np.array(rf_spearmans)
    print('spearmans: ', rf_spearmans, rf_spearmans.mean(), np.std(rf_spearmans))
    rf_overlaps = np.array(rf_overlaps)
    print('over: ',rf_overlaps, rf_overlaps.mean(), np.std(rf_overlaps))


    print('MLP方法结果：')
    mlp_ndcg_scores = np.array(mlp_ndcg_scores)
    print('ndcg: ', mlp_ndcg_scores, mlp_ndcg_scores.mean(), np.std(mlp_ndcg_scores))
    mlp_spearmans = np.array(mlp_spearmans)
    print('spearmans: ', mlp_spearmans, mlp_spearmans.mean(), np.std(mlp_spearmans))
    mlp_overlaps = np.array(mlp_overlaps)
    print('over: ', mlp_overlaps, mlp_overlaps.mean(), np.std(mlp_overlaps))


def pagerank(data_path, dataset):
    ndcg_scores = []
    spearmans = []
    overlaps = []

    for cross_id in range(5):
        g, edge_types, _, rel_num, node_feats, labels, train_idx, val_idx, test_idx = \
            load_data(data_path, dataset, cross_id)
        g = g.to_networkx()
        # g = g.subgraph(train_idx)
        pr = nx.pagerank_scipy(g)
        pre = []
        # print(len(pr))
        # print(pr)

        for i in train_idx:
            pre.append(pr[i])
        # print(pre)

        p = []


        for i in train_idx:
            p.append(labels[i])
        # print(p)

        # print(pr[train_idx])
        # nx.draw(g)
        # plt.show()

        # print(train_idx)

        # print(labels[train_idx])
        pr_test_ndcg, pr_test_spearman = \
            get_rank_metrics(torch.from_numpy(np.array(pre)), labels[train_idx], 100, spearman=True)
        pr_test_overlap = overlap(labels[train_idx], torch.from_numpy(np.array(pre)), 100)
        print("Test NDCG {:.4f} | Test Spearman {:.4f} | Test Overlap {:.4f}".
              format(pr_test_ndcg, pr_test_spearman, pr_test_overlap))

        ndcg_scores.append(pr_test_ndcg)
        spearmans.append(pr_test_spearman)
        overlaps.append(pr_test_overlap)

    print()
    ndcg_scores = np.array(ndcg_scores)
    print('ndcg: ', ndcg_scores, ndcg_scores.mean(), np.std(ndcg_scores))

    spearmans = np.array(spearmans)
    print('spearmans: ', spearmans, spearmans.mean(), np.std(spearmans))

    overlaps = np.array(overlaps)
    print(overlaps, overlaps.mean(), np.std(overlaps))


def personalized_pagerank(data_path, dataset):
    ndcg_scores = []
    spearmans = []
    overlaps = []

    for cross_id in range(5):
        g, edge_types, _, rel_num, node_feats, labels, train_idx, val_idx, test_idx = \
            load_data(data_path, dataset, cross_id)
        g = g.to_networkx()

        alpha = 1 / len(test_idx)
        node = {}
        for i in test_idx:
            node[i] = alpha

        ppr = nx.pagerank(g, personalization=node)

        pre = []
            # 预测值
        for i in test_idx:
            pre.append(ppr[i])

        ppr_test_ndcg, ppr_test_spearman = \
                get_rank_metrics(torch.from_numpy(np.array(pre)), labels[test_idx], 100, spearman=True)
        ppr_test_overlap = overlap(labels[test_idx], torch.from_numpy(np.array(pre)), 100)

        ndcg_scores.append(ppr_test_ndcg)
        spearmans.append(ppr_test_spearman)
        overlaps.append(ppr_test_overlap)
    print()
    ndcg_scores = np.array(ndcg_scores)
    print('ndcg: ', ndcg_scores, ndcg_scores.mean(), np.std(ndcg_scores))

    spearmans = np.array(spearmans)
    print('spearmans: ', spearmans, spearmans.mean(), np.std(spearmans))

    overlaps = np.array(overlaps)
    print(overlaps, overlaps.mean(), np.std(overlaps))


if __name__ == '__main__':
    data_path = '../datasets/fb15k_rel.pk'
    dataset = 'FB15k_rel'
    baseline(data_path, dataset)
    # pagerank(data_path, dataset)
    # personalized_pagerank(data_path,dataset)
