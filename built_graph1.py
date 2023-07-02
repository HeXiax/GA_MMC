import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
from torch.nn.parameter import Parameter

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class AttentionLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, Cross=False):
        super(AttentionLayer, self).__init__()
        self.Cross = Cross

        self.W = Parameter(torch.FloatTensor(in_dim, hidden_dim))
        torch.nn.init.xavier_uniform_(self.W)

        self.W1 = Parameter(torch.FloatTensor(in_dim, hidden_dim))
        torch.nn.init.xavier_uniform_(self.W1)


    def forward(self, feat_x, feat_y, k):
        # feat_x = F.normalize(feat_x, p=1, dim=1)
        # feat_y = F.normalize(feat_y, p=1, dim=1)
        # feat_x = torch.mm(feat_x, self.W1)
        # feat_y = torch.mm(feat_y, self.W1)

        assert feat_x.shape[1] == feat_y.shape[1]
        #e = torch.mm(torch.mm(feat_x, self.W), feat_y.transpose(0,1))

        norm1 = torch.norm(feat_x, p=2, dim=1).view(-1, 1)  # 范数
        norm2 = torch.norm(feat_y, p=2, dim=1).view(-1, 1)  # 范数
        e = torch.div(torch.mm(torch.mm(feat_x, self.W1), feat_y.t()), torch.mm(norm1, norm2.t()) + 1e-7)

        A = -9e15 * torch.ones_like(e).cuda()
        #A = torch.zeros_like(e).cuda()


        # for i in range(e.shape[0]):
        #     _, ind = torch.topk(e[i,:], k)
        #     b = e[i, ind]
        #     A[i, ind] = e[i, ind]
        a, ind = torch.topk(e, k)
        A.scatter_(1, ind, a)

        attention = F.softmax(A, dim=1)
        # rowsum = torch.sum(A, dim=1) ** (-1)
        # rowsum[torch.isinf(rowsum)] = 0.
        # D = torch.diag(rowsum)
        # attention = torch.mm(D, A)

        # out = torch.mm(attention, feat_y)
        return attention

class Multi_head_attention(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=8):
        super(Multi_head_attention, self).__init__()
        self.attention = [AttentionLayer(in_dim, hidden_dim) for _ in range(num_heads)]

        for i, attention in enumerate(self.attention):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, feat_x, feat_y, k):
        x = torch.cat([att(feat_x, feat_y, k).unsqueeze(0) for att in self.attention], dim=0)
        h = torch.mean(x, dim=0, keepdim=False)

        return h

class KNN_Att(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(KNN_Att, self).__init__()
        self.W = Parameter(torch.FloatTensor(in_feat, out_feat))
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, X, Y, k):
        X = torch.mm(X, self.W)
        Y = torch.mm(Y, self.W)
        num = X.shape[0]
        norm1 = torch.norm(X, p=2, dim=1).view(-1, 1)  # 范数
        norm2 = torch.norm(Y, p=2, dim=1).view(-1, 1)  # 范数
        cos = torch.div(torch.mm(X, Y.t()), torch.mm(norm1, norm2.t()) + 1e-7)
        cos1 = cos.transpose(0, 1)
        A = -9e15 * torch.ones_like(cos).cuda()
        #A = torch.zeros_like(cos).cuda()


        a, ind = torch.topk(cos, k)
        A.scatter_(1, ind, a)

        #B = torch.zeros_like(cos1).cuda()
        B = -9e15 * torch.ones_like(cos1).cuda()
        b, ind1 = torch.topk(cos1, k)
        B.scatter_(1, ind1, b)

        S1 = F.relu(A)
        rowsum = torch.sum(S1, dim=1) ** (-0.5)
        rowsum[torch.isinf(rowsum)] = 0.
        D = torch.diag(rowsum)
        S1 = torch.mm(D, S1)
        S1 = torch.mm(S1, D)


        S2 = B
        rowsum = torch.sum(S2, dim=1) ** (-1)
        rowsum[torch.isinf(rowsum)] = 0.
        D = torch.diag(rowsum)
        S2 = torch.mm(D, S2)
        #S2 = torch.mm(S2, D)


        return S1, S2




class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, adj, features, active=False):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output



def built_cos_knn(dataset, k):
    input = dataset
    num = input.shape[0]
    norm2 = torch.norm(input, p=2, dim=1).view(-1, 1)  # 范数
    cos = torch.div(torch.mm(input, input.t()), torch.mm(norm2, norm2.t()) + 1e-7)
    A = torch.zeros_like(cos).cuda()

    a, ind = torch.topk(cos, k)
    A.scatter_(1, ind, a)

    A = 0.5 * (A + A.transpose(0, 1))
    A = A.fill_diagonal_(0.0)
    A = A + torch.eye(num).cuda()
    rowsum = torch.sum(A, dim=1)**(-1)
    rowsum[torch.isinf(rowsum)] = 0.
    D = torch.diag(rowsum)
    A = torch.mm(D, A)
    #A = torch.mm(A, D)

    return A
