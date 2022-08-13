import torch
from torch import nn
import torch.nn.functional as F
from g_transformer.g_trans import GTRANBRel_feat, GTRANRel_feat
import dgl
import tqdm
from loss import Loss


def list_loss(y_pred, y_true, list_num=10, eps=1e-10):
    '''
    y_pred: [n_node, 1]
    y_true: [n_node, 1]
    '''
    n_node = y_pred.shape[0]

    ran_num = list_num - 1
    indices = torch.multinomial(torch.ones(n_node), n_node*ran_num, replacement=True).to(y_pred.device)

    list_pred = torch.index_select(y_pred, 0, indices).reshape(n_node, ran_num)
    list_true = torch.index_select(y_true, 0, indices).reshape(n_node, ran_num)

    list_pred = torch.cat([y_pred, list_pred], -1) # [n_node, list_num]
    list_true = torch.cat([y_true, list_true], -1) # [n_node, list_num]

    list_pred = F.softmax(list_pred, -1)
    list_true = F.softmax(list_true, -1)

    list_pred = list_pred + eps
    log_pred = torch.log(list_pred)

    return torch.mean(-torch.sum(list_true * log_pred, dim=1))


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class CrossAttention(nn.Module):
    def __init__(self, temperature, in_dim, out_dim, in_drop):
        super(CrossAttention, self).__init__()
        self.temperature = temperature

        self.in_dim = in_dim
        self.out_dim = out_dim

        # q, k, v
        self.w_q = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.w_k = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.w_v = nn.Linear(self.in_dim, self.out_dim, bias=False)

        # FFN
        self.FFN = nn.Sequential(
            nn.Linear(self.out_dim, int(self.out_dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(self.out_dim*0.5), self.out_dim),
            nn.Dropout(0.1)
        )

        self.layer_norm = nn.LayerNorm(self.out_dim, eps=1e-6)


    def forward(self, struct_h, cont_h):
        h = torch.stack([struct_h, cont_h], 1) # [n_node, 2, in_dim]

        q = self.w_q(h)
        k = self.w_k(h)
        v = self.w_v(h)

        attn = torch.matmul(q / self.temperature, k.transpose(1,2))
        attn = F.softmax(attn, dim=-1)
        attn_h = torch.matmul(attn, v)

        attn_o = self.FFN(attn_h) + attn_h
        attn_o = self.layer_norm(attn_o)

        struct_o = attn_o[:, 0, :]
        cont_o = attn_o[:, 1, :]

        return struct_o, cont_o


# full batch version
class rgtn(nn.Module):
    def __init__(self,
                 args,
                 g,
                 rel_num,
                 struct_in_dim,
                 content_in_dim,
                 centrality,
                 loss_function1):
        super(rgtn, self).__init__()
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        self.rel_emb = nn.Embedding(rel_num, args.pred_dim)
        self.struct_gtran = GTRANRel_feat(g, args.num_layers, rel_num, args.pred_dim, struct_in_dim,
                                          args.num_hidden, heads, args.in_drop, args.attn_drop,
                                          args.residual, args.norm, args.edge_mode, self.rel_emb)
        self.content_gtran = GTRANRel_feat(g, args.num_layers, rel_num, args.pred_dim, content_in_dim,
                                           args.num_hidden, heads, args.in_drop, args.attn_drop,
                                           args.residual, args.norm, args.edge_mode, self.rel_emb)
        self.loss_fn1 = loss_function1
        self.feat_drop = args.feat_drop
        self.graph = g
        self.scale = args.scale
        self.h_dim = args.num_hidden * heads[-2]
        self.cross_attention = CrossAttention(self.h_dim ** 0.5, self.h_dim, self.h_dim, args.in_drop)

        self.centrality = centrality


        self.loss_fn2 = Loss(self.h_dim)

        self.attn_vec = nn.Parameter(
            torch.FloatTensor(size=(self.h_dim, 1)))  # 类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并其绑定到这个module里面
        nn.init.xavier_uniform_(self.attn_vec)  # 随机初始化

        # 结构重要性分数
        self.output_layer1 = nn.Sequential(
            nn.Linear(self.h_dim, 1),
            nn.LeakyReLU(inplace=True)
        )

        # 语义重要性分数
        self.output_layer2 = nn.Sequential(
            nn.Linear(self.h_dim, 1),
            nn.LeakyReLU(inplace=True)
        )

        if self.scale:  # 使用中心性调整
            self.gamma = nn.Parameter(torch.FloatTensor(size=(1,)))
            self.beta = nn.Parameter(torch.FloatTensor(size=(1,)))
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

        self.loss_lambda = args.loss_lambda  # 添加无监督的损失？ 0.5
        self.bn_s = nn.BatchNorm1d(self.h_dim)  # 结构的归一化
        self.bn_c = nn.BatchNorm1d(self.h_dim)  # 语义的归一化

        self.loss_alpha = args.loss_alpha  # 添加排序损失
        self.loss_beta = args.loss_beta
        self.list_num = args.list_num  # 排序的数量

    def forward(self, struct_input, content_input, edge_types, pos, labels=None, idx=None, ret_feat=False):
        if self.feat_drop > 0:
            struct_input = F.dropout(struct_input[idx], self.feat_drop, self.training)
            content_input = F.dropout(content_input[idx], self.feat_drop, self.training)

        struct_h = self.struct_gtran(struct_input[idx], edge_types[idx])
        content_h = self.content_gtran(content_input[idx], edge_types[idx])

        struct_h1, content_h1 = self.cross_attention(struct_h, content_h)
        # add_norm
        struct_h1 = self.bn_s(struct_h + struct_h1)
        content_h1 = self.bn_c(content_h + content_h1)

        # attention-based aggregation
        q = torch.stack([struct_h1, content_h1], 1)
        attn_last = torch.matmul(q, self.attn_vec) # [N_node, 2, 1]
        attn_last = F.softmax(attn_last, 1)

        logit_struct = self.output_layer1(q[:, 0, :])
        if self.scale:
            logit_struct = nn.functional.relu((self.centrality * self.gamma + self.beta).unsqueeze(-1) * logit_struct)

        logit_content = self.output_layer2(q[:, 1, :])
        logit_all = torch.stack([logit_struct, logit_content], 1) # [N_node, 2, 1]

        logits = torch.sum(logit_all * attn_last, 1) # [N_node, 1]

        if ret_feat:
            node_emb = torch.sum(q * attn_last, 1).cpu()
            return logits, node_emb

        if self.training:
            loss1, loss2 = self.loss_fn2(struct_h1, content_h1, pos, idx)
            loss_struct = self.loss_fn(logit_struct[idx], labels[idx].unsqueeze(-1))
            loss_content = self.loss_fn(logit_content[idx], labels[idx].unsqueeze(-1))
            loss_all = self.loss_fn(logits[idx], labels[idx].unsqueeze(-1))
            loss = (1-self.loss_lambda) * loss_all + self.loss_lambda * (loss_struct + loss_content) / 2
            loss = self.loss_alpha * list_loss(logits[idx], labels[idx].unsqueeze(-1), self.list_num) + loss
            return logits, loss
        else:
            return logits


# mini batch version
class rgtn_b(nn.Module):
    def __init__(self,
                 args,
                 rel_num,
                 struct_in_dim,
                 content_in_dim,
                 loss_function1):
        super(rgtn_b, self).__init__()

        self.rel_emb = nn.Embedding(rel_num, args.pred_dim)
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        self.struct_gtran = GTRANBRel_feat(args.num_layers, rel_num, args.pred_dim, struct_in_dim,
                                  args.num_hidden, heads, args.in_drop, args.attn_drop, args.residual,
                                  args.norm, args.edge_mode, self.rel_emb)
        self.content_gtran = GTRANBRel_feat(args.num_layers, rel_num, args.pred_dim, content_in_dim,
                                  args.num_hidden, heads, args.in_drop, args.attn_drop, args.residual,
                                  args.norm, args.edge_mode, self.rel_emb)
        self.loss_fn1 = loss_function1


        self.feat_drop = args.feat_drop
        self.attn_struct = nn.Linear(args.num_hidden * heads[-2], 1)
        self.attn_content = nn.Linear(args.num_hidden * heads[-2], 1)
        self.scale = args.scale   # 是否中心性调整

        self.h_dim = args.num_hidden*heads[-2]
        self.loss_fn2 = Loss(self.h_dim)

        self.attention = ScaledDotProductAttention(temperature=self.h_dim ** 0.5)

        self.w_q = nn.Linear(self.h_dim, self.h_dim, bias=False)
        self.w_k = nn.Linear(self.h_dim, self.h_dim, bias=False)
        self.w_v = nn.Linear(self.h_dim, self.h_dim, bias=False)

        self.out_fc = PositionwiseFeedForward(self.h_dim, int(0.5 * self.h_dim), 0.1)

        self.attn_vec = nn.Parameter(torch.FloatTensor(size=(self.h_dim, 1)))    # 类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并其绑定到这个module里面
        nn.init.xavier_uniform_(self.attn_vec)  # 随机初始化

        # 结构重要性分数
        self.output_layer1 = nn.Sequential(
            nn.Linear(self.h_dim, 1),
            nn.LeakyReLU(inplace=True)
        )

        # 语义重要性分数
        self.output_layer2 = nn.Sequential(
            nn.Linear(self.h_dim, 1),
            nn.LeakyReLU(inplace=True)
        )

        if self.scale:    # 使用中心性调整
            self.gamma = nn.Parameter(torch.FloatTensor(size=(1,)))
            self.beta = nn.Parameter(torch.FloatTensor(size=(1,)))
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

        self.loss_lambda = args.loss_lambda   # 添加无监督的损失？ 0.5
        self.bn_s = nn.BatchNorm1d(self.h_dim)  # 结构的归一化
        self.bn_c = nn.BatchNorm1d(self.h_dim)  # 语义的归一化

        self.loss_alpha = args.loss_alpha     # 添加排序损失
        self.loss_beta = args.loss_beta
        self.list_num = args.list_num     # 排序的数量

    def forward(self, blocks, struct_input, content_input, pos, neg, labels=None):
        # 如果特征的dropout不为0，进行dropout，初始为0
        if self.feat_drop > 0:
            struct_input = F.dropout(struct_input, self.feat_drop, self.training)
            content_input = F.dropout(content_input, self.feat_drop, self.training)

        # 得到两种嵌入
        struct_h = self.struct_gtran(blocks, struct_input)
        content_h = self.content_gtran(blocks, content_input)

        # 将两种嵌入堆叠起来
        h = torch.stack([struct_h, content_h], 1)   # [N_node, 2, h_dim]，沿着一个新维度对输入张量序列进行连接，通常为了保留–[序列(先后)信息] 和 [张量的矩阵信息]

        # interact struct and content feature
        q = self.w_q(h)
        k = self.w_k(h)
        v = self.w_v(h)

        q, _ = self.attention(q, k, v)  # [N_node, 2, h_dim]
        q = self.out_fc(q) + h

        q = torch.stack(
            [self.bn_s(q[:, 0, :]), self.bn_c(q[:, 1, :])], 1
        )
        # print(q)

        # attention-based aggregation
        attn_last = torch.matmul(q, self.attn_vec)  # [N_node, 2, 1]
        attn_last = F.softmax(attn_last, 1)

        # 得到的结构的重要性
        logit_struct = self.output_layer1(q[:, 0, :])
        if self.scale:
            centrality = blocks[-1].dstdata['centrality']
            logit_struct = nn.functional.relu((centrality * self.gamma + self.beta).unsqueeze(-1) * logit_struct)    # 中心性调整

        # 得到的语义的重要性
        logit_content = self.output_layer2(q[:, 1, :])

        # 将结构和语义的重要性拼接起来
        logit_all = torch.stack([logit_struct, logit_content], 1)    # 沿着一个新维度对输入张量序列进行连接

        logits = torch.sum(logit_all * attn_last, 1)     # 两者混合的重要性

        # 最终的节点嵌入和语义嵌入
        struct_h = self.bn_s(q[:, 0, :])
        content_h = self.bn_c(q[:, 1, :])

        # 训练
        if self.training:
            # loss_con = self.loss_fn2(struct_h, content_h)
            loss1, loss2 = self.loss_fn2(struct_h, content_h, pos, neg)   # 对比损失
            loss_struct = self.loss_fn1(logit_struct, labels)       # 结构的MSE损失L1
            loss_content = self.loss_fn1(logit_content, labels)     # 语义的MSE损失L2
            loss_all = self.loss_fn1(logits, labels)                # 两者的MSE损失L0
            # 下游任务损失
            loss = (1-self.loss_lambda) * loss_all + self.loss_lambda * (loss_struct + loss_content) / 2

            # 只有无监督
            # loss = self.loss_alpha * list_loss(logits, labels, self.list_num) + loss + loss1  # 总损失   L0+(L1+L2)/2+Lj
            # 只有有监督
            # loss = self.loss_alpha * list_loss(logits, labels, self.list_num) + loss + loss2  # 总损失   L0+(L1+L2)/2+Lj

            loss = self.loss_alpha * list_loss(logits, labels, self.list_num) + 0.3 * loss + 0.7 * (loss1 + loss2 ) # 总损失   L0+(L1+L2)/2+Lj

            return logits, loss    # 如果在训练，返回最终的重要性和总损失
        else:
            return logits    # 返回最终的重要性

    # 推理,用学习好的模型做预测
    def inference(self, g, x_struct, x_content, batch_size, num_workers, device, ret_feat=False):
        # 自注意力层的个数
        layer_num = len(self.struct_gtran.gat_layers)

        # 每个注意力层的输出
        layer_output_s = self.struct_gtran.layer_output
        layer_output_c = self.content_gtran.layer_output

        # 初始化所有节点的标签为0
        logits = torch.zeros(g.number_of_nodes(), 1).to(device)

        if ret_feat:
            node_emb = torch.zeros(g.number_of_nodes(), layer_output_c[-1])

        # zip将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。l表示第几层
        # 对于每层：
        for l, (layer_s, layer_c) in enumerate(zip(self.struct_gtran.gat_layers, self.content_gtran.gat_layers)):

            # 结构和语义重要性
            y_struct = torch.zeros(g.number_of_nodes(), layer_output_s[l] if l!=layer_num-1 else 1)   # 最后一层只有1
            y_content = torch.zeros(g.number_of_nodes(), layer_output_c[l] if l!=layer_num-1 else 1)

            # 获取节点的所有邻居，生成需计算的节点在每一层计算时所需的依赖图。创建一个特定图的列表，这些图表示每层的计算依赖
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.number_of_nodes()),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers
            )

            # tqdm.tqdm进度条
            # input_nodes代表计算output_nodes的表示所需的节点。块包含了每个GNN层要计算哪些节点表示作为输出，要将哪些节点表示作为输入，以及来自输入节点的表示如何传播到输出节点。
            # 对于每层，每块节点：
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]
                block = block.int().to(device)
                edge_types = block.edata['etypes']

                # structure branch
                edge_feat_s = self.struct_gtran.rel_emb(edge_types)  # 关系嵌入
                h_struct = x_struct[input_nodes].to(device)   # 初始结构特征
                h_struct = layer_s(block, h_struct, h_struct, h_struct, edge_feat_s)  # 训练后的结构特征

                # content branch
                edge_feat_c = self.content_gtran.rel_emb(edge_types)
                h_content = x_content[input_nodes].to(device)    # 初始语义特征
                h_content = layer_c(block, h_content, h_content, h_content, edge_feat_c)   # 训练后的语义特征

                # 如果是最后一层，计算输出
                if l == (layer_num-1):
                    h = torch.stack([h_struct, h_content], 1)  # [N_node, 2, h_dim]

                    # interact struct and content feature
                    q = self.w_q(h)
                    k = self.w_k(h)
                    v = self.w_v(h)

                    q, _ = self.attention(q, k, v)  # [N_node, 2, h_dim]
                    q = self.out_fc(q) + h

                    q = torch.stack(
                        [self.bn_s(q[:, 0, :]), self.bn_c(q[:, 1, :])], 1
                    )

                    # attention-based aggregation
                    attn_last = torch.matmul(q, self.attn_vec)  # [N_node, 2, 1]
                    attn_last = F.softmax(attn_last, 1)

                    h_struct = self.output_layer1(q[:, 0, :])
                    if self.scale:
                        centrality = block.dstdata['centrality']
                        h_struct = nn.functional.relu(
                            (centrality * self.gamma + self.beta).unsqueeze(-1) * h_struct)

                    h_content = self.output_layer2(q[:, 1, :])
                    logit_all = torch.stack([h_struct, h_content], 1)

                    logits[output_nodes] = torch.sum(logit_all * attn_last, 1)
                    if ret_feat:
                        node_emb[output_nodes] = torch.sum(q * attn_last, 1).cpu()

                y_struct[output_nodes] = h_struct.cpu()
                y_content[output_nodes] = h_content.cpu()

            x_struct = y_struct
            x_content = y_content
        # print(logits)
        if ret_feat:
            return logits, node_emb
        return logits

