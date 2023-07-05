import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.sparse as sp
import math
import random


class Linear(nn.Module): 
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MLP_encoder(nn.Module):#
    def __init__(self, nfeat, nhid, dropout):
        super(MLP_encoder, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        return x


class LPGraph(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, epsilon, K, alpha):
        super(LPGraph, self).__init__()
        self.encoder = Linear(nfeat, nhid, dropout, bias=True)
        self.predictor = Linear(nhid, nclass, dropout, bias=True)
        self.epsilon = epsilon
        self.K = K 
        self.alpha = alpha
        self.number_class = nclass

    def forward(self, x): 
        x = torch.relu(self.encoder(x))
        return x, torch.log_softmax(self.predictor(x),dim=-1)

    def loss_function(self, y_hat, y_soft, epoch = 0):
        if self.training:
            y_hat_con = torch.detach(torch.exp(y_hat))
            exp = np.log(epoch / self.epsilon + 1)

            loss = - torch.sum(torch.mul(y_hat, torch.mul(y_soft, y_hat_con**exp))) / self.number_class

        else: 
            loss = - torch.sum(torch.mul(y_hat, y_soft)) / self.number_class
        return loss

    def inference(self, h, adj):
        y0 = torch.exp(h)
        y = y0
        for i in range(self.K):
            y = (1 - self.alpha) * torch.matmul(adj, y) + self.alpha * y0
        return y

    def get_edge_prob(self, x, adj):
        x = torch.relu(self.encoder(torch.matmul(adj, x)))
        h = self.predictor(torch.matmul(adj, x))
        A_pred = torch.sigmoid(h @ h.T)
        return A_pred


class AGST(nn.Module):

    def __init__(self, best_state, nfeat, nhid, nclass, dropout, epsilon, K, alpha, nshot, m):
        super(AGST, self).__init__()

        self.model = LPGraph(nfeat=nfeat,
                          nhid=nhid,
                          nclass=nclass,
                          dropout=dropout,
                          epsilon=epsilon,
                          K=K,
                          alpha=alpha)

        self.model_momt = MLP_encoder(nfeat=nfeat,
                    nhid=nhid,
                    dropout=dropout)

        for param_ori, param_momt in zip(self.model.encoder.parameters(), self.model_momt.parameters()):
            param_momt.data.copy_(param_ori.data)  # initialize
            param_momt.requires_grad = False  # not update by gradient

        self.proj_head1 = Linear(nhid, nhid, dropout, bias=True)

        self.nclass = nclass
        self.nshot = nshot

        # degree
        self.m = m

    @torch.no_grad()
    def _momentum_update_momt_encoder(self):
        """
        Momentum update
        """
        for param_ori, param_momt in zip(self.model.parameters(), self.model_momt.parameters()):
            param_momt.data = param_momt.data * self.m + param_ori.data * (1. - self.m)

    def forward(self, features, idx_train):
        with torch.no_grad():
            self._momentum_update_momt_encoder()

        query_features, output = self.model(features)

        query_features = self.proj_head1(query_features)

        k_embeddings = self.model_momt(features[idx_train])
        k_embeddings = self.proj_head1(k_embeddings)

        z_dim = k_embeddings.size()[1]
        # embedding lookup
        support_embeddings = k_embeddings
        support_embeddings = support_embeddings.view([self.nclass, self.nshot, z_dim])
        prototype = support_embeddings.sum(1) / self.nshot

        return output, query_features, prototype

    def get_proto_loss(self, query_features, label_momt, prototype, proto_norm_momt):

        query_features_norm = torch.norm(query_features, dim=-1)
        query_features = torch.div(query_features, query_features_norm.unsqueeze(1))

        prototype_norm = torch.norm(prototype, dim=-1)
        prototype = torch.div(prototype, prototype_norm.unsqueeze(1))

        sim_zc = torch.matmul(query_features, prototype.t())

        sim_zc_normalized = torch.div(sim_zc, proto_norm_momt)
        sim_zc_normalized = torch.exp(sim_zc_normalized)

        sim_2centroid = torch.gather(sim_zc_normalized, -1, label_momt)
        sim_sum = torch.sum(sim_zc_normalized, -1, keepdim=True)
        sim_2centroid = torch.div(sim_2centroid, sim_sum)
        proto_loss = torch.mean(sim_2centroid.log())
        proto_loss = -1 * proto_loss

        return proto_loss

