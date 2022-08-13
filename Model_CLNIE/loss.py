import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Loss(nn.Module):
    def __init__(self, hidden_dim, batch_size=1, tau=0.5):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.tau = tau
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, pos, neg) -> torch.Tensor:
        unsupervised_loss = self.loss(z1, z2)
        semsupervised_loss = self.label_loss2(z1, z2, pos, neg)
        return unsupervised_loss, semsupervised_loss

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        # 对z1、z2进行归一化
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        # 得到相似性
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)

        # 同一视图内
        refl_sim = f(self.sim(z1, z1))
        # 两个视图间
        between_sim = f(self.sim(z1, z2))

        # h = between_sim.diag()
        # a = refl_sim.sum(1)

        # 自己和别人/（自己和自己+自己和别人） 两个视图的 - 同一视图内自己和自己， 对列求和，得到每个节点和其余所有节点的相似性的和
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size = None):
        # 映射到计算对比损失的空间
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        # 两个视图的损失
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def cal_loss2(self, z1: torch.Tensor, z2: torch.Tensor, pos, neg):
        f = lambda x: torch.exp(x / self.tau)
        # 创建空的tensor
        pos_sim = torch.Tensor().cuda()
        neg_sim = torch.Tensor().cuda()

        for i in range(len(pos)):
            # 生成正例掩码
            mask_pos = np.zeros(len(z1), dtype=bool)
            pos[i].sort()  # 把正例的序号升序排列，包含锚节点
            pos_idx = pos[i].index(i)   # 找到锚节点下标
            for j in pos[i]:   # 把正例的mask设置为1
                mask_pos[j] = 1

            # between_sim = torch.cat([between_sim, f(self.sim(z1[mask], z2[mask])[0])[None,:].sum(1)], dim=0)

            # 计算得到正例之间的相似性，取锚节点的那一行，就是锚节点和其他所有正例节点的相似性
            sim_pos = self.sim(z1[mask_pos], z2[mask_pos])[pos_idx]
            pos_sim = torch.cat([pos_sim, f(sim_pos)], dim=0)
            # between_sim = torch.cat([between_sim, f(self.sim(z1[mask_pos], z2[mask_pos])[0])], dim=0)

            # 生成负例掩码
            mask_neg = np.zeros(len(z1), dtype=bool)
            neg[i].sort()
            neg_idx = neg[i].index(i) # 找到锚节点下标
            for j in neg[i]:
                mask_neg[j] = 1

            # 计算得到负例的相似性，取锚节点那一行，去掉自己和自己
            sim_neg = self.sim(z1[mask_neg], z2[mask_neg])[neg_idx]
            sim_neg1 = sim_neg[:neg_idx]
            sim_neg2 = sim_neg[neg_idx + 1:]
            sim_neg = torch.cat([sim_neg1, sim_neg2], dim=0)
            # sim_neg = self.sim(z1[mask_pos], z2[mask_pos])[:, 1:][neg_idx]
            neg_sim = torch.cat([neg_sim, torch.cat([f(sim_pos), f(sim_neg)], dim=0)], dim=0)

            # print()

        a = -torch.log(pos_sim.sum(dim=0) / neg_sim.sum(dim=0))
        return a

        # between = f(self.sim(z1, z2)).sum(dim=1)
        # return -torch.log(pos_sim.sum(dim=0) / between.sum(dim=0))

    def label_loss2(self, z1: torch.Tensor, z2: torch.Tensor, pos, neg):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.cal_loss2(h1, h2, pos, neg)
        l2 = self.cal_loss2(h2, h1, pos, neg)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret


