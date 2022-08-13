import torch

def topkpos(labels, k, wk):
    pos = {}
    neg = {}

    for n in range(len(labels)):
        base = labels[n]
        # 返回两个值，第一个为排序后数据，第二个为索引
        sort_p = torch.sort(abs(labels - base), descending=False, dim=0)
        sort_n = torch.sort(abs(labels - base), descending=True, dim=0)
        # 先把自己加进去
        # 使用指定的键返回项目的值，如果键不存在，则插入这个具有指定值的键
        pos.setdefault(n, []).append(n)
        # 接下来加k个最相近的
        for i in range(k):
            if sort_p[1][i].item() != n:
                pos.setdefault(n, []).append(sort_p[1][i].item())
            # neg.setdefault(n, []).append(sort_n[1][i].item())
        if len(pos[n]) != k:
            pos[n].pop()

        # neg.setdefault(n, []).append(n)
        # pos.setdefault(n, []).append(n)
        nk = int(wk * 0.01 * len(labels))
        for i in range(nk):
            neg.setdefault(n, []).append(sort_n[1][i].item())
        neg.setdefault(n, []).append(n)


    return pos, neg


