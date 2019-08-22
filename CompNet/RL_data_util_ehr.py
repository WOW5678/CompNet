# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/20 0020 下午 8:33
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 按照GCN需要的格式 准备好邻接矩阵
"""
import numpy as np
import  scipy.sparse as sp
import os
import sys
import pickle as pkl
from collections import Counter
import csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
import dill
import gc
import pickle
import pickle  as pkl

np.random.seed(1001)

def init_ADJ(nodes_dict):
    adjacencies = []
    adj_shape = (len(nodes_dict), len(nodes_dict))

    edges = np.empty((len(nodes_dict), 2), dtype=np.int32)
    for j in range(len(nodes_dict)):
        edges[j] = np.array([j, j])
    row, col = np.transpose(edges)
    data = np.zeros(len(row))
    adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.uint8)
    #save_sparse_csr('/home/wangshanshan/GCN_RL/adjData/rel_combination.npz',adj)
    adjacencies.append(adj)
    adjacencies.append(adj)
    #ehr=get_ehr(nodes_dict)
    # print(ehr.to_dense().shape)
    #adjacencies.append(ehr)
    # 处理对抗关系 生成对抗元组的邻接矩阵(初始时也全部为0)
    #save_sparse_csr('/home/wangshanshan/GCN_RL/adjData/rel_adverse.npz', adj)
    return adjacencies

def getADJ(action,seletectedActions,nodes_dict,ddiIDS):
    adjacencies=[]
    adj_shape=(len(nodes_dict),len(nodes_dict))

    spoList_rel0,spoList_rel1=get_spo(action, seletectedActions, ddiIDS)
    edges = np.empty((len(spoList_rel0), 2), dtype=np.int32)
    for j,(s,p,o) in enumerate(spoList_rel0):
        #print('s-p-o:',(s,p,o))
        edges[j]=np.array([s,o])
    row,col=np.transpose(edges)
    data=np.ones(len(row))
    adj=sp.csr_matrix((data,(row,col)),shape=adj_shape,dtype=np.uint8)
    #save_sparse_csr('/home/wangshanshan/GCN_RL/adjData/rel_combination.npz',adj)
    adjacencies.append(adj)

    #处理对抗关系 生成对抗元组的邻接矩阵
    if len(spoList_rel1)==0:
        edges = np.empty((len(nodes_dict), 2), dtype=np.int32)
        for j in range(len(nodes_dict)):
            edges[j] = np.array([j, j])
        row, col = np.transpose(edges)
        data = np.zeros(len(row))
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.uint8)
    else:
        edges = np.empty((len(spoList_rel1), 2), dtype=np.int32)
        for j, (s, p, o) in enumerate(spoList_rel1):
            #print('s-p-o:',(s,p,o))
            edges[j] = np.array([s, o])
        row, col = np.transpose(edges)
        data = np.ones(len(row))
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.uint8)
    #save_sparse_csr('/home/wangshanshan/GCN_RL/adjData/rel_adverse.npz', adj)
    adjacencies.append(adj)
    del adj,edges,adj_shape,data,row,col,spoList_rel0,spoList_rel1
    gc.collect()
    return adjacencies

def get_spo(action,seletectedActions,ddiIDS):
    spoList0=[]
    #先添加组合药物组
    spoList0.append([action,'组合',action])
    if len(seletectedActions)!=0:
        for id in seletectedActions:
            spoList0.append([action,'组合',id])
            spoList0.append([id,'组合',action])

    spoList1=[]
    for row in ddiIDS:
        if action==row[0]:
            # print('action-对抗-row[2]:',[action,'对抗',row[1]])
            spoList1.append([action,'对抗',row[1]])
        if action==row[1]:
            spoList1.append([row[0],'对抗',action])
    return spoList0,spoList1

def save_sparse_csr(filename,array):
    np.savez(filename,data=array.data,indices=array.indices,indptr=array.indptr,shape=array.shape)

def load_data(filename):
    patients=[]
    drugs=[]
    drugSet=[]
    with open(filename,'r',encoding='utf-8') as f:
        reader=csv.reader(f)
        for row in reader:
            drugL=row[6].split(' ')
            #row[4]为病人描述
            row[4]=row[4].replace(' ','')
            tmpL=[item for item in drugL if item]
            if len(tmpL)>0:
                patients.append(' '.join(row[4]))
                drugs.append(tmpL)
                drugSet.extend(tmpL)
    return patients,drugs,list(set(drugSet))

def load_records(filename):
    records=dill.load(open(filename,'rb'))
    #对代码进行id化
    diagnosis=[row[0] for row in records]
    procedures=[row[1] for row in records]
    medicines=[row[2] for row in records]
    print('medicines:',medicines[:10])
    diagnosis_maxlen=max([len(line.split(' ')) for line in diagnosis])
    procedure_maxlen=max([len(line.split(' ')) for line in procedures])
    # 序列化诊断代码
    diagnosis_tokenizer = Tokenizer()
    diagnosis_tokenizer.fit_on_texts(diagnosis)
    sequences = diagnosis_tokenizer.texts_to_sequences(diagnosis)
    #填充变成等长的序列
    diagnosis_= pad_sequences(sequences, maxlen=diagnosis_maxlen, padding='post', truncating='post')
    #序列化过程代码
    procedure_tokenizer=Tokenizer()
    procedure_tokenizer.fit_on_texts(procedures)
    sequences=procedure_tokenizer.texts_to_sequences(procedures)
    #print(sequences)
    #从后端开始截断或padding
    #因为maxLength太大了 会带来很大的计算消耗，因此这里将maxLength设置为300
    #maxLength=300
    procedure_=pad_sequences(sequences,maxlen=procedure_maxlen,padding='post',truncating='post')

    # 序列化药物代码
    medicineSet=[]
    for row in medicines:
        for item in row.split(' '):
            if item not in medicineSet:
                medicineSet.append(item)
    #使药物字典的索引从0开始，，防止后续评估阶段出现错位
    drug2id={drug:id for id,drug in enumerate(medicineSet)}
    drug2id['END']=len(drug2id)
    drugIds=[]
    for line in medicines:
        #print('line:',line)
        line=line+' '+'END'
        drugIds.append([drug2id[item] for item in line.split(' ')])
    #规整数据集
    X,Y=[],[]
    for x,y,z in zip(diagnosis_,procedure_,drugIds):
        X.append([x,y])
        Y.append(z)
    print('drugIds:',drugIds[:5])
    return diagnosis_maxlen,procedure_maxlen,diagnosis_tokenizer,procedure_tokenizer,X,Y,drugIds,drug2id

def load_drugDDI(ddi_file,TOPK,med2id):
    from collections import defaultdict
    import pandas as pd

    atc3_atc4_dic = defaultdict(set) #key为atc3，value为atc4，其中atc3是atc4仅保留前4个字符
    for item in med2id.keys():
        atc3_atc4_dic[item[:4]].add(item)  # 为什么只保留前4个字符呢？不明白这里

    # cid_act文件中是药物CID编码与atc3编码的对应
    cid2atc_dic=defaultdict(set)

    cid_atc='MIMIC-III/drug-atc.csv'
    with open(cid_atc, 'r') as f:
        for line in f:
            line_ls = line[:-1].split(',')
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0:
                    cid2atc_dic[cid].add(atc[:4])
    #print(cid2atc_dic)
    # ddi load
    ddi_df = pd.read_csv(ddi_file)
    # fliter sever side effect
    ddi_most_pd = ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename(
        columns={0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
    ddi_most_pd = ddi_most_pd.iloc[-TOPK:, :]
    # ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
    fliter_ddi_df = ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
    ddi_df = fliter_ddi_df[['STITCH 1', 'STITCH 2']].drop_duplicates().reset_index(drop=True)
    #将ddi_df对应到atc4编码形式
    ddi_df['STITCH 1']=ddi_df['STITCH 1'].map(cid2atc_dic)
    ddi_df['STITCH 2']=ddi_df['STITCH 2'].map(cid2atc_dic)
    drugDDI=[]
    for drug1,drug2 in zip(ddi_df['STITCH 1'],ddi_df['STITCH 1']):
        for item in drug1:
            for item2 in drug2:
                if item!=item2 and [med2id.get(item),med2id.get(item2)] not in drugDDI:
                    drugDDI.append([med2id.get(item),med2id.get(item2)])
    return drugDDI

def get_torch_sparse_matrix(A,dev):
    '''
    A : list of sparse adjacency matrices
    '''
    #idx:(2,2586)
    # print('A.tocoo().row:',A.tocoo().row)
    # print('A.tocoo().col:',A.tocoo().col)
    newA=[]
    for row in A:
        idx = torch.LongTensor([row.tocoo().row, row.tocoo().col])
        #dat :tensor，shape:(2586,)
        #print('A:',A.tocoo().data)
        dat = torch.FloatTensor(row.tocoo().data)
        newA.append(torch.sparse.FloatTensor(idx, dat, torch.Size([row.shape[0], row.shape[1]])).to(dev))
    del idx,dat
    # gc.collect()
    return newA




