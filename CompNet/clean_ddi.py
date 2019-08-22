# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/20 0020 下午 3:37
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 修正DDI数据集，使得DDI中的药物名称保持与EHR中的药物名称一致  同时只保存EHR中出现过的DDI药物对
"""
import csv

def load_ddi(filename):
    ddiDrugSet=[]
    ddiDrugPairs=[]
    with open(filename,'r',encoding='utf-8') as f:
        reader=csv.reader(f)
        for row in reader:
            if len(row)>0:
                ddiDrugSet.extend([row[0].replace('\ufeff',''),row[1]])
                ddiDrugPairs.append([row[0].replace('\ufeff',''),row[1]])
    return ddiDrugSet,ddiDrugPairs

def load_EHR(filename):
    drugs=[]
    with open(filename,'r',encoding='utf-8') as f:
        reader=csv.reader(f)
        for row in reader:
            drugs.extend(row[6].split(' '))
    return set(drugs)

def linkDrugs(drugset,ddiDrugSet, ddiDrugPairs):
    ddiDrug2EhrDrug={} # ddi中的药物名称到EHR中药物名称的对应，每个ddi药物可能对应对个EHR药物，因此value为一个list
    for drug in drugset:
        #对drug 进行变形 判断变形后的drug是否存在ddiDrugSet中
        transformD=drugMatch(drug,ddiDrugSet)
        print(transformD)
        if transformD!=None and transformD not in ddiDrug2EhrDrug:
            ddiDrug2EhrDrug[transformD]=[]
            ddiDrug2EhrDrug[transformD].append(drug)
        elif transformD!=None and transformD in ddiDrug2EhrDrug:
            ddiDrug2EhrDrug[transformD].append(drug)
    #print(ddiDrug2EhrDrug)
    #重新整理DDI数据(将DDI中的药名 替换成EHR中的药物名称)
    newDDIDrugPairs=[]
    for pair in ddiDrugPairs:

        if len(pair)>0:
            drug1,drug2=None,None
            if pair[0] in ddiDrug2EhrDrug:
                drug1=ddiDrug2EhrDrug.get(pair[0])
            if pair[1] in ddiDrug2EhrDrug:
                drug2=ddiDrug2EhrDrug.get(pair[1])
            if drug1!=None and drug2!=None:
                for d1 in drug1:
                    for d2 in drug2:
                        if [d1,d2] not in newDDIDrugPairs:
                            newDDIDrugPairs.append([d1,d2])
    #print('newDDIDrugPairs:',newDDIDrugPairs)
    #将规整好之后的对抗药物对 写入到新的文件中
    with open('data/newDDIData.csv','w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerows(newDDIDrugPairs)

    #将两个drug都存在于EHR的写入到一个新文件中，后面使用更加方便 即对newDDIDrugPairs进行过滤
    with open('data/newDDIData_filter.csv','w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        for row in newDDIDrugPairs:
            if row[0] in drugset and row[1] in drugset:
                writer.writerow(row)




def drugMatch(drug,ddiDrugSet):
    for index_1 in range(len(drug), -1, -1):
        for index_2 in range(len(drug)):
            # 先判断药物是否在drugSet中 若不在的话 不可能有DDI 直接不用后续的判断
            # 若在的话 直接返回匹配结果
            drug2 = drug[index_2:index_1]
            if drug2 in ddiDrugSet:
                return drug2
    return None


if __name__ == '__main__':
    ddi_path='data/药物不良作用.csv'
    ehr_path='data/patientInformation_0103_onetime_current.csv'
    drugset=load_EHR(ehr_path)
    ddiDrugSet, ddiDrugPairs=load_ddi(ddi_path)
    print('len(ddiDrugSet):',len(drugset))
    # print(ddiDrugSet)
    # print('ddiDrugPairs:',ddiDrugPairs)
    linkDrugs(drugset,ddiDrugSet, ddiDrugPairs)