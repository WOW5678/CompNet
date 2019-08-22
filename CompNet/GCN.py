# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/21 0021 下午 3:22
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RGCN(nn.Module):
    def __init__(self,layer_sizes,total_ent,dev='cpu'):
        super(RGCN, self).__init__()
        # layer_size:list[16,4]
        self.layer_sizes = layer_sizes

        self.node_init = None
        self.layers = nn.ModuleList()
        self.device=dev


        _l=total_ent
        for l in self.layer_sizes:
            self.layers.append(GCNLayer(_l, l))
            _l = l
        self.ent_emb = None

    def forward(self,adj_mat_list):
        '''
        inp: (|E| x d)
        adj_mat_list: (R x |E| x |E|)
        '''
        out = self.node_init
        final_rep=[]
        for i, layer in enumerate(self.layers):
            if i != 0:
                out = F.relu(out)
            out = layer(out, adj_mat_list)
            final_rep.append(out)
            # out = F.normalize(out)
        self.ent_emb = out
        final_rep=torch.cat(final_rep,1)
        final_rep=torch.sum(final_rep,0).view(-1,final_rep.shape[1])
        return out,final_rep

class GCNLayer(nn.Module):
    def __init__(self,in_size,out_size,total_rel=2,n_basis=2,dev='cuda:1'):
        super(GCNLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.n_basis=n_basis
        self.total_rel=total_rel
        self.dev=dev
        self.basis_weights = nn.Parameter(torch.FloatTensor(self.n_basis, self.in_size, self.out_size))
        self.basis_coeff = nn.Parameter(torch.FloatTensor(self.total_rel, self.n_basis))

        self.register_parameter('bias', None)
        self.reset_parameters() #初始化所有的变量

    def forward(self,inp,adj_mat_list):
        '''
               inp: (|E| x in_size)
               adj_mat_list: (R x |E| x |E|)
               '''
        # Aggregation (no explicit separation of Concat step here since we are simply averaging over all)
        # self.basis_coeff:(95,2),self.basis_weights:(2,8285,16)
        # rel_weights:(95,8285,16)
        rel_weights = torch.einsum('ij,jmn->imn', [self.basis_coeff, self.basis_weights])
        # weights:(787075,16)
        weights = rel_weights.view(rel_weights.shape[0] * rel_weights.shape[1],
                                   rel_weights.shape[2]) # (in_size * R, out_size)

        emb_acc = []
        if inp is not None:
            for mat in adj_mat_list:
                # torch.mm是矩阵乘法相称
                emb_acc.append(torch.mm(mat, inp))  # (|E| x in_size)
            tmp = torch.cat(emb_acc, 1)
        else:
            tmp = torch.cat([item.to_dense() for item in adj_mat_list], 1)
            # tmp=torch.cat([tmp,adj_mat_list[2]],1)
            # print(adj_mat_list[0].to_dense().shape)
            # print(adj_mat_list[2].to_dense().shape)
            # print(tmp.shape)
        out = torch.matmul(tmp, weights)  # (|E| x out_size)

        if self.bias is not None:
            out += self.bias.unsqueeze(0)  # (|E| x out_size)
        return out  # (|E| x out_size)

    def reset_parameters(self):
        #使用xaview_uniform_方法初始化权重
        nn.init.xavier_uniform_(self.basis_weights.data) #(2,8285,16)
        nn.init.xavier_uniform_(self.basis_coeff.data) #(95,2)

        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias.data)
