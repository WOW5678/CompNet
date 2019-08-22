# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/26 0026 下午 2:30
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 对MIMIC数据集进行预处理工作，选择出住院次数在2次以上的病人，目的减少药物空间，并且方便与很多方法进行对比
"""
import csv
import pandas as pd
from collections import defaultdict
import dill
import numpy as np


def process_procedure(procedure_file):
    pro_pd=pd.read_csv(procedure_file,dtype={'ICD9_CODE':'category'})
    print(pro_pd.head(5))
    #删除ROW_ID列
    pro_pd.drop(columns=['ROW_ID'],inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'],inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'],inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True,inplace=True)
    print(pro_pd.head(5))
    return pro_pd

def process_med(med_file):
    med_pd=pd.read_csv(med_file,dtype={'NDC':'category'})
    #filter
    med_pd.drop(columns=['ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
                     'FORMULARY_DRUG_CD','GSN','PROD_STRENGTH','DOSE_VAL_RX',
                     'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP','FORM_UNIT_DISP',
                      'ROUTE','ENDDATE','DRUG'],axis=1,inplace=True)
    print('med_pd.head(5):',med_pd.head(5))
    med_pd.drop(index=med_pd[med_pd['NDC']=='0'].index,axis=0,inplace=True)
    med_pd.dropna(inplace=True) #删除空行
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    def filter_first24hour_med(med_pd):
        med_pd_new = med_pd.drop(columns=['NDC'])
        med_pd_new = med_pd_new.groupby(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']).head([1]).reset_index(drop=True)
        print(med_pd_new.head(10))
        med_pd_new = pd.merge(med_pd_new, med_pd, on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'])
        print('merge:',med_pd_new.head(10))
        med_pd_new = med_pd_new.drop(columns=['STARTDATE'])
        return med_pd_new

    med_pd=filter_first24hour_med(med_pd)
    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    def process_vist_lg2(med_pd):
        a=med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
        a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x: len(x))
        a = a[a['HADM_ID_Len'] > 1]
        return a

    med_pd_lg2 = process_vist_lg2(med_pd).reset_index(drop=True)
    med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')
    print(med_pd.head(5))
    print('len(med_pd):',len(med_pd))
    return med_pd

def process_diag(diag_file):
    diag_pd=pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM', 'ROW_ID'], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    return diag_pd.reset_index(drop=True)
#
def ndc2atc4(med_pd):
    with open(ndc_rxnorm_file, 'r') as f:

        ndc2rxnorm=eval(f.read())
        print(ndc2rxnorm)
    # ndc2rxnorm是个字典，key为NDC，value为RXCUI
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)
    print(med_pd.head(5))

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index=med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)

    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC4':'NDC'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    print('transoform:', len(med_pd))
    return med_pd


#得到前1000个最频繁的procedure代码
def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['ICD9_CODE'].isin(pro_count.loc[:1000, 'ICD9_CODE'])]

    return pro_pd.reset_index(drop=True)

#过滤得到2000个最频繁的诊断
def filter_2000_most_diag(diag_pd):
    diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]
    return diag_pd.reset_index(drop=True)


# 过滤得到前300个最频繁的药物
def filter_300_most_med(med_pd):
    med_count = med_pd.groupby(by=['NDC']).size().reset_index().rename(columns={0: 'count'}).sort_values(by=['count'],                                                                                                         ascending=False).reset_index(
        drop=True)
    med_pd = med_pd[med_pd['NDC'].isin(med_count.loc[:299, 'NDC'])]

    return med_pd.reset_index(drop=True)

def process_all():
    #get med and diag(vist>=2)
    med_pd = process_med(med_file)
    med_pd = ndc2atc4(med_pd)
    diag_pd = process_diag(diag_file)
    #diag_pd = filter_2000_most_diag(diag_pd)
    pro_pd = process_procedure(procedure_file)

    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')


    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index()
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'PRO_CODE'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
#     data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data['NDC_Len'] = data['NDC'].map(lambda x: len(x))
    print('data:',len(data)) #13727
    return data


def statistics():
    print('#patients ', data['SUBJECT_ID'].unique().shape)
    print('#clinical events ', len(data))

    diag = data['ICD9_CODE'].values
    med = data['NDC'].values
    pro = data['PRO_CODE'].values

    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])

    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))
    print('#procedure', len(unique_pro))

    avg_diag = 0
    avg_med = 0
    avg_pro = 0
    max_diag = 0
    max_med = 0
    max_pro = 0
    cnt = 0
    max_visit = 0
    avg_visit = 0

    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]
        x = []
        y = []
        z = []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['ICD9_CODE']))
            y.extend(list(row['NDC']))
            z.extend(list(row['PRO_CODE']))
        x = set(x)
        y = set(y)
        z = set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y)
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt
    print('#avg of diagnoses ', avg_diag / cnt)
    print('#avg of medicines ', avg_med / cnt)
    print('#avg of procedures ', avg_pro / cnt)
    print('#avg of vists ', avg_visit / len(data['SUBJECT_ID'].unique()))

    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of procedures ', max_pro)
    print('#max of visit ', max_visit)

#################################################################
#Create Vocaboray for Medical Codes & Save Patient Record in pickle file
import dill
import pandas as pd
class Voc(object):
    def __init__(self):
        self.id2word={}
        self.word2id={}
    def add_sentence(self,sentence):
        for word in sentence:
            if word not in self.word2id:
                self.id2word[len(self.word2id)]=word
                self.word2id[word]=len(self.word2id)

def create_str_token_mapping(df):
    diag_voc=Voc()
    med_voc=Voc()
    pro_voc=Voc()

    for index,row in df.iterrows():
        diag_voc.add_sentence(row['ICD9_CODE'])
        med_voc.add_sentence(row['NDC'])
        pro_voc.add_sentence(row['PRO_CODE'])

    dill.dump(obj={'diag_voc':diag_voc, 'med_voc':med_voc ,'pro_voc':pro_voc},file=open('MIMIC-III/voc_final.pkl','wb'))
    return diag_voc,med_voc,pro_voc

def create_patient_records_for_my(df,dig_voc,med_voc,pro_voc):
    records=[]
    for index,row in df.iterrows():
        admission=[]
        admission.append( ' '.join(row['ICD9_CODE']).strip())
        admission.append(' '.join(row['PRO_CODE']).strip())
        admission.append(' '.join(row['NDC']).strip())
        records.append(admission)
    print(records[:3])
    print(len(records))
    dill.dump(obj=records,file=open('MIMIC-III/records_final_for_my.pkl','wb'))
    return records


if __name__ == '__main__':
    med_file = 'MIMIC-III/PRESCRIPTIONS.csv'
    diag_file = 'MIMIC-III/DIAGNOSES_ICD.csv'
    procedure_file = 'MIMIC-III/PROCEDURES_ICD.csv'

    #因为drug-ddi中药物的编码是atc4格式的，所以需要将med_file中的NDC编码转换为atc4格式
    ndc2atc_file = 'MIMIC-III/rxnorm2atc4.csv'
    cid_atc = 'MIMIC-III/drug-atc.csv' #？？？？
    #药物代码映射文件
    ndc_rxnorm_file = 'MIMIC-III/ndc2rxnorm_mapping.txt' # NDC代码转换成xrnorm，xrnorm再转换成atc4格式

    data = process_all()
    statistics()
    data.to_pickle('MIMIC-III/data_final.pkl')
    data.head()

    ##处理记录
    path = 'MIMIC-III/data_final.pkl'
    df = pd.read_pickle(path)
    diag_voc, med_voc, pro_voc = create_str_token_mapping(df)
    #records = create_patient_record(df, diag_voc, med_voc, pro_voc)
    records=create_patient_records_for_my(df,diag_voc,med_voc,pro_voc)

    #处理对抗药物数据
    ddi_file='MIMIC-III/drug-DDI.csv'
    cid_atc='MIMIC-III/drug-atc.csv'
    voca_file='MIMIC-III/voc_final.pkl'
    data_file='MIMIC-III/records_final_for_my.pkl'
    TOPK=40

    #读取电子病历记录
    records=dill.load(open(data_file,'rb'))
    cid2atc3_dic=defaultdict(set) #value 为一个set
    med_voc=dill.load(open(voca_file,'rb'))['med_voc']
    med_voc_size=len(med_voc.id2word)
    med_unique_word=[med_voc.id2word[i] for i in range(med_voc_size)]
    atc3_atc4_dic=defaultdict(set)
    for item in med_unique_word:
        atc3_atc4_dic[item[:4]].add(item) #为什么只保留前4个字符呢？不明白这里

    with open(cid_atc, 'r') as f:
        for line in f:
            line_ls = line[:-1].split(',')
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0:
                    cid2atc3_dic[cid].add(atc[:4])
    # ddi load
    ddi_df = pd.read_csv(ddi_file)
    # fliter sever side effect
    ddi_most_pd = ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename(
        columns={0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
    ddi_most_pd = ddi_most_pd.iloc[-TOPK:, :]
    # ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
    fliter_ddi_df = ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
    ddi_df = fliter_ddi_df[['STITCH 1', 'STITCH 2']].drop_duplicates().reset_index(drop=True)
    print('=================')
    print(len(ddi_df))
    print('complete!')
