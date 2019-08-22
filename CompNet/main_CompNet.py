# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/20 0020 下午 7:35
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function:利用RL结合GCN 进行多种药物的预测
 使用诊断数据对病人进行建模
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import dill
import collections
import random
from sklearn.model_selection import train_test_split
from RL_data_util_ehr import *
from GCN import *
import gc
from sklearn.metrics import roc_auc_score

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device='cpu'
print('device:',device)

class CNN(nn.Module):
    def __init__(self,vocab_size,emb_size,num_channels,hidden_dim,dropout):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(embedding_dim=emb_size,num_embeddings=vocab_size)
        self.conv=nn.Sequential(
            nn.Conv1d(           #input shape (1, 28, 28)
                in_channels=emb_size,   #input height
                out_channels=num_channels, # n_filters
                kernel_size=3,   # filter size
                stride=2,        # filter movement/step
                                 # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),              # output shape (16, 28, 28)
            nn.Tanh(),
            #nn.MaxPool2d(kernel_size=2),# choose max value in 2x2 area, output shape (16, 14, 14)
            nn.Conv1d(  # input shape (1, 28, 28)
                in_channels=num_channels,  # input height
                out_channels=num_channels,  # n_filters
                kernel_size=3,  # filter size
                stride=2,  # filter movement/step
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),
            nn.Tanh(),
            # output shape (16, 28, 28)
            #nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
            nn.Conv1d(  # input shape (1, 28, 28)
                in_channels=num_channels,  # input height
                out_channels=num_channels,  # n_filters
                kernel_size=3,  # filter size
                stride=2,  # filter movement/step
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),
        )
        self.dropout=dropout
        self.out=nn.Linear(num_channels,hidden_dim,bias=True)
        nn.init.kaiming_normal_(self.out.weight)
    def forward(self,x):
        #print('x:',type(x))
        x_emb=self.embedding(x).unsqueeze(0).permute(0,2,1)
        #print('x_emb:',x_emb.shape) #[1, 100, 30]
        x = self.conv(x_emb)
        # average and remove the extra dimension
        remaining_size = x.size(dim=2)
        features = F.max_pool1d(x, remaining_size).squeeze(dim=2)
        features = F.dropout(features, p=self.dropout)
        output = self.out(features)
        return output


class DQN(nn.Module):
    def __init__(self,state_size,action_size):
        super(DQN, self).__init__()
        # 一个简单的三层的感知器网络用来根据状态做决策
        self.W=nn.Parameter(torch.FloatTensor(state_size,state_size))
        # 使用xaview_uniform_方法初始化权重
        nn.init.xavier_uniform_(self.W.data)  # (2,8285,16)
        self.U=nn.Parameter(torch.FloatTensor(state_size,state_size))
        #nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_uniform_(self.U.data)  # (95,2)

        self.fc1=nn.Linear(state_size,512)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2=nn.Linear(512,action_size)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.learn_step_counter=0 #用于判断何时更新target网络

    def forward(self,x_):
        x_t,h_t_1=x_[0],x_[1]
        #x_t=torch.FloatTensor(x_t.cpu())
        #print('x_t:',x_t.shape) #1, 100
        #print('h_t_1:',type(h_t_1))

        h_t_1=h_t_1.to(device)
        #h_t_1=torch.from_numpy(h_t_1)
        #print('h_t_l:',h_t_1.shape)
        #print('h_t_1:',h_t_1[0][:5])
        state=F.sigmoid(torch.mm(self.W,x_t.t())+torch.mm(self.U,h_t_1.t()))
        #print('state:',state.t()[0][:5])
        fc1=F.relu(self.fc1(state.t()))
        output=self.fc2(fc1)
        #print('state-h:{}'.format(state.t()[0][:5]))
        return state.t(),output

class Agent(object):
    def __init__(self,state_size,action_size,layer_sizes):
        super(Agent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        # self.memory = collections.deque(maxlen=3000)
        self.gamma = 0.9  # 计算未来奖励的折扣率
        self.epsilon = 0.9  # agent最初探索环境选择action的探索率
        self.epsilon_min = 0.05  # agent控制随机探索的阈值
        self.epsilon_decay = 0.995  # 随着agent做选择越来越好，降低探索率
        self.cnn_diagnosis = CNN(len(diagnosis_token.word_index) + 2, EMB_SIZE, 128, EMB_SIZE, 0.5).to(device)
        self.cnn_procedure = CNN(len(procedure_token.word_index) + 2, EMB_SIZE, 128, EMB_SIZE, 0.5).to(device)
        self.rgcn= RGCN(layer_sizes,drug_vocab_size,dev=device).to(device)
        self.model=DQN(state_size,action_size).to(device)
        self.target_model =DQN(state_size,action_size).to(device)
        self.model_params = list(self.cnn_diagnosis.parameters())+list(self.cnn_procedure.parameters()) +list(self.rgcn.parameters())+ list(self.model.parameters())
        self.optimizier=torch.optim.Adam(self.model_params,lr=LR,betas=(0.9,0.999),weight_decay=5.0)
        self.loss=nn.MSELoss()
        self.load_params()
        self.update_target_model()

    def load_params(self):
        if os.path.exists('MIMIC-III/agent.pkl'):
            #reload the params
            trainedModel=torch.load('MIMIC-III/agent.pkl')
            print('load trained model....')
            self.cnn_diagnosis.load_state_dict(trainedModel.cnn_diagnosis.state_dict())
            self.cnn_procedure.load_state_dict(trainedModel.cnn_procedure.state_dict())
            self.rgcn.load_state_dict(trainedModel.rgcn.state_dict())
            self.model.load_state_dict(trainedModel.model.state_dict())
            self.target_model.load_state_dict(trainedModel.target_model.state_dict())

    def reset(self,x):
        #得到每个电子病历数据的表示
        x0=torch.LongTensor(x[0]).to(device)
        x1=torch.LongTensor(x[1]).to(device)
        diagnosis_f = self.cnn_diagnosis(x0)
        procedure_f=self.cnn_procedure(x1)
        f=torch.cat((diagnosis_f,procedure_f),0) #按行拼接 f:(batch,t1+t2,hidden_size) 即f:(batch,diagnosis_maxlen+procedure_maxlen,100)
        #print('g:',f.shape)
        return f # f.shape:(2,100)

    def act(self,x,h,selectedAction):
        #根据state 选择action
        if np.random.rand()<self.epsilon:
            while True:
                action=random.randrange(self.action_size)
                if action not in selectedAction:
                    return action,h #直接使用上一步的隐状态作为当前的隐状态
        next_h,output=self.model((x,h))
        while True:
            with torch.no_grad():
                action = torch.max(output, 1)[1]
                if action not in selectedAction:
                    return action,next_h
                else:
                    output[0][action]=-999999

    def new_state(self,f,g):
        #print('g:',g[0][:5])
        # 这里并不是简单的相加或者拼接 而是使用了一种gate attention机制
        a = nn.functional.softmax(torch.mm(f, g.t()))
        f_ = torch.mm(a.t(),f)
        x = f_+ g
        return x #x shape(1,100)

    def step(self,action,selectedAction,y):
        # 首先判断该action是否为结束的标志：
        if action==action_size-1:
            #判断当前结束的步数是否对：
            if len(selectedAction)==len(y)-1:
                reward=2
                return reward,0
            else: #不该结束 结束了 有两种情况 超过结束个数或者不到结束个数
                reward=-2
                return reward,0
        else:
            #根据该action进行奖励
            if int(action) in y and action not in selectedAction :
                reward=1
            else:
                reward=-1

            #更新新的药物图谱
            adjacencies=getADJ(action,selectedAction,drug2id,ddi_df)
            adjacencies=get_torch_sparse_matrix(adjacencies,device)

            _, g = self.rgcn(adjacencies) # g shape(1,100)
            del adjacencies
            gc.collect()
            return reward,g

    def replay(self,BATCH_SIZE):
        print('learning_step_counter:{},self.epsilon:{}'.format(self.model.learn_step_counter,self.epsilon))
        # 没训练一次 都将模型learn_step_counter加一 并且判断是否需要更新target网络
        if self.model.learn_step_counter%TARGET_UPDATE_ITER==0:
            print('Update target model.')
            self.update_target_model()

            # 保存下训练过程中的损失函数值
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        self.model.learn_step_counter+=1
        batch_idx=np.random.choice(len(memory),BATCH_SIZE)
        #print(batch_idx)
        b_x=[]
        b_h=[]
        b_action = []
        b_reward=[]
        b_next_x=[]
        b_next_h=[]
        for id in batch_idx:
            #print('b_x:', memory[id][0][0][:5])
            b_x.append(memory[id][0])
            #print('b_h:', memory[id][1][0][:5])
            b_h.append(memory[id][1])

            #print('b_action:', memory[id][2])
            b_action.append(memory[id][2])
            #print('b_reward:', memory[id][3])
            b_reward.append(memory[id][3])
            #print('b_next_x:', memory[id][4][0][:5])
            b_next_x.append(memory[id][4])
            #print('b_next_h:', memory[id][5][0][:5])
            b_next_h.append(memory[id][5])
        #将这些数据转变成tensor对象
        b_x=torch.cat(b_x,0).to(device)
        b_h = torch.cat([item.to(device) for item in b_h], 0)
        b_reward=torch.FloatTensor(b_reward).to(device)
        b_action=torch.LongTensor(b_action).to(device)
        b_next_x=torch.cat(b_next_x,0)
        b_next_h = torch.cat([item.to(device) for item in b_next_h], 0)

        b_action=b_action.unsqueeze(1).to(device)
        _,q_eval=self.model((b_x,b_h))
        q_eval=q_eval.gather(1,b_action)
        #print('q_eval:',q_eval[0][:5])
        _,q_next=self.target_model((b_next_x,b_next_h))
        q_next=q_next.detach()

        q_target=(b_reward+GAMMA*q_next.max(1)[0]).unsqueeze(1)
        #print('q_target:', q_target[0][:5])
        loss=self.loss(q_eval,q_target)

        self.optimizier.zero_grad()
        loss.backward(retain_graph=True)
        # for param in self.model_params:
        #     param.grad.data.clamp_(-5,5)
        self.optimizier.step()

        return loss

    def update_target_model(self):
        #加载self.model的参数到target_model中
        self.target_model.load_state_dict(self.model.state_dict())


class Evaluate(object):
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    def evaluate(self):
        # 评估在数据集上的模型表现
        # 其他统计指标
        Jaccard_list = []
        Recall_list = []
        Reward_list = []
        Precision_list=[]
        F_list=[]
        D_DList=[]

        for x,y in zip(self.X,self.Y):
            if len(y)==0:
                break
            sampleReward = 0
            y = set(y)  # 因为y中有重复的药物 因此这里转变为集合 去掉重复的药物
            # 得到初始的状态
            # 使用GCN对adjaccies进行处理 得到图表示
            # if test:
            _, g = agent.rgcn(init_adj)
            #print('g:',g.shape)
            f= agent.reset(x)
            selectedAction=[]
            h = np.zeros((1, state_size))  # （1,100）
            h=torch.FloatTensor(h)
            #按理说应该不知道len(y)
            for step in range(max_step):
                x = agent.new_state(f, g)  # x shape:(100,1)
                #print('step:{},x:{},h:{}'.format(step,x[0][:5],h[0][:5]))

                next_h, output =agent.model((x,h))
                #print('output:',output[0][:5])
                next_h=next_h.detach()
                output=output.detach()
                prob,action=torch.max(output,1)
                # print('output:{}'.format(output[0][:5]))
                # print('prob:{},action:{}'.format(prob,action))
                if action == action_size - 1 and step == 0:
                    output[0][action_size-1]=-999999
                while True:
                    action = torch.max(output, 1)[1]
                    if int(action) not in selectedAction:
                        break
                    else:
                        output[0][action] = -999999
                # 执行该action 得到reward 并更新状态
                reward, _= agent.step(action, selectedAction, y)
                if type(_) != int:  # 说明预测的不是结束符
                    g = _
                    # 将选择的action加入到selectedAction中
                    selectedAction.append(int(action))
                    sampleReward += int(reward)
                    next_x = agent.new_state(f, g)  # 得到新时刻的输入xt
                    # 用新时刻的状态替代原先的状态
                    x = next_x
                    h = next_h

                else:  # 预测出了结束符
                    selectedAction.append(int(action))
                    sampleReward += int(reward)
                    break

            jaccard, recall, precision, f_measure = self.evaluate_sample(selectedAction, y)

            Jaccard_list.append(jaccard)
            Recall_list.append(recall)
            Reward_list.append(sampleReward)
            Precision_list.append(precision)
            F_list.append(f_measure)
            # 判断生成的药物中是否有DDI药物对
            d_d,ddRate =self.evaluate_ddi(y_pred=selectedAction)
            #print('d_d:',d_d)
            D_DList.append(ddRate)
        avg_jaccard = sum(Jaccard_list) * 1.0 / len(Jaccard_list)
        avg_recall = sum(Recall_list) * 1.0 / len(Recall_list)
        avg_reward = sum(Reward_list) * 1.0 / len(Reward_list)
        avg_precision=sum(Precision_list)*1.0/len(Precision_list)
        avg_f=sum(F_list)*1.0/len(F_list)
        avg_ddr=sum(D_DList)*1.0/len(D_DList)
        print('avg_jaccard:{},avg_recall:{},avg_precision:{},avg_f:{},avg_reward:{},avg_ddr:{}'.format(avg_jaccard, avg_recall,avg_precision,avg_f, avg_reward,avg_ddr))
        del Jaccard_list,Recall_list,Reward_list,Precision_list,F_list

        return avg_reward,avg_jaccard,avg_recall,avg_precision,avg_f,avg_ddr

    def evaluate_sample(self,y_pred,y_true): #针对单个样本的三个指标的评估结果
        print('y_pred:',y_pred)
        print('y_true:',y_true)
        jiao_1 = [item for item in y_pred if item in y_true]
        bing_1 = [item for item in y_pred] + [item for item in y_true]
        bing_1 = list(set(bing_1))
        # print('jiao:',jiao_1)
        # print('bing:',bing_1)
        recall = len(jiao_1) * 1.0 / len(y_true)
        precision = len(jiao_1) * 1.0 / len(y_pred)
        jaccard = len(jiao_1) * 1.0 / len(bing_1)

        if recall + precision == 0:
            f_measure = 0
        else:
            f_measure = 2 * recall * precision * 1.0 / (recall + precision)
        print('jaccard:%.3f,recall:%.3f,precision:%.3f,f_measure:%.3f' % (jaccard, recall, precision, f_measure))
        del jiao_1,bing_1
        return jaccard,recall,precision,f_measure

    def evaluate_ddi(self,y_pred):
        y_pred=list(set(y_pred))
        #根据药物id找到对应的药物名称
        # pred_drugs=[drug2id.get(id) for id in y_pred]
        #判断这些药物中是否存在着对抗的药物对

        #对生成的药物进行两两组合
        D_D=[]
        for i in range(len(y_pred) - 1):
            for j in range(i + 1, len(y_pred)):
                key1 = [y_pred[i],y_pred[j]]
                key2 = [y_pred[j],y_pred[i]]

                if key1 in ddi_df or key2 in ddi_df:
                    # 记录下来该DDI数据  以便论文中的case Study部分分析
                    D_D.append(key1)
        allNum=len(y_pred)*(len(y_pred)-1)/2
        if allNum>0:
            return D_D,len(D_D)*1.0/allNum
        else:
            return D_D,0

    def plot_result(self,total_reward,total_recall,total_jaccard):
        # 画图
        import matplotlib.pyplot as plt
        import matplotlib
        # 开始画图
        plt.figure()
        ax = plt.gca()

        epochs = np.arange(len(total_reward))

        plt.subplot(1, 2, 1)
        # 设置横坐标的显示刻度为50的倍数
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        plt.plot(epochs, total_reward, label='Reward')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, total_recall, label='Recall', color='red')
        plt.plot(epochs, total_jaccard, label='Jaccard')
        # 设置横坐标的显示刻度为50的倍数
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')

        plt.show()  # 展示


def testModel():

    # 评估一下模型在测试集上的表现
    test_eva = Evaluate(X_test, Y_test)
    avg_reward, avg_jaccard, avg_recall, avg_precision, avg_f, ddiNum = test_eva.evaluate()
    # 将结果写入到文件中
    print('The result on test set.....')
    #print('avg_reward:{},\tavg_jaccard:{},\tavg_recall:{},\tavg_precision:{},\tavg_f:{},\tddiNum:{}'.format(
    #    avg_reward,
    #    avg_jaccard, avg_recall, avg_precision, avg_f, ddiNum))
    del avg_reward, avg_jaccard, ddiNum, avg_precision, avg_f
    gc.collect()


if __name__ == '__main__':
    # 1. 加载数据集
    records='MIMIC-III/records_final_for_my.pkl'
    diagnosis_maxlen, procedure_maxlen, diagnosis_token,procedure_token,X,Y, medicine_sequences, drug2id=load_records(records)
    drug_vocab_size = len(drug2id)
    #print(Y[:50])
    # 2.准备好DDI药物对
    drugDDIFile='MIMIC-III/drug-DDI.csv'
    ddi_df=load_drugDDI(drugDDIFile,TOPK=40,med2id=drug2id)

    # 3. 分割训练集和验证集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(1/3), random_state=1)
    # 对训练集再次进行划分 得到训练集和验证集
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=1)
    print('train,val,test:', len(X_train), len(X_val), len(X_test))
    del records,X,Y

    with open('MIMIC-III/testData.pkl','wb') as f:
        pickle.dump([X_test,Y_test],f)
    with open('MIMIC-III/ddi-drug2id.pkl','wb') as f:
        pickle.dump([ddi_df,drug2id],f)
    with open('MIMIC-III/drug2id.pkl','wb') as f:
        pickle.dump(drug2id,f)


    # 4. 强化学习部分
    # 超参数参数
    LR=1e-5
    EMB_SIZE=100
    Layer_sizes=[50,50]
    state_size = EMB_SIZE
    action_size = drug_vocab_size
    print('action_size:',action_size)
    BATCH_SIZE = 64
    EPISODES = 500  # 让agent玩游戏的次数
    TARGET_UPDATE_ITER=10
    GAMMA=1
    max_step=15
    min_jaccard = 0.0
    memory=collections.deque(maxlen=1000)
    Test=False
    # #因为每个样本的初始时有一样的图谱，所以放在循环外面，可以节省计算资源和时间
    # 初始化图
    init_adj= init_ADJ(drug2id)
    #print(len(drug2id))
    init_adj=get_torch_sparse_matrix(init_adj,device)
    if os.path.exists('MIMIC-III/drug2id.pkl'):
        with open('MIMIC-III/drug2id.pkl', 'rb') as f:
            drug2id = pickle.load(f)
    print('drug2id:', drug2id)
    if Test:
        agent=torch.load('MIMIC-III/agent.pkl')
        agent.cnn_diagnosis.eval()
        agent.cnn_procedure.eval()
        agent.rgcn.eval()
        agent.model.eval()
        print(agent.model_params[1])
        testModel()
    else:
        agent = Agent(state_size, action_size, layer_sizes=Layer_sizes)
        # 使用GCN对adjaccies进行处理 得到图表示
        _, init_g = agent.rgcn(init_adj) #init_g shape:(1,100)
        for e in range(EPISODES):
            print('epoch:%d'%e)
            epochLoss=[]
            #针对每个EHR
            for x,y in zip(X_train,Y_train):
                g=init_g
                sampleReward=0
                y=set(y) #因为y中有重复的药物 因此这里转变为集合 去掉重复的药物
                if len(y)==0:
                    break
                # 使用RNN得到EHR的表示f
                f=agent.reset(x) # shape (2,100)
                h=np.zeros((1,state_size)) #（1,100）
                h=torch.FloatTensor(h)
                selectedAction =[]
                # 得到初始的状态
                for step in range(len(y)):
                    x=agent.new_state(f,g) # x shape:(100,1)
                    #print('step:{},h:{}'.format(step, h[0][:5]))
                    if step==len(y)-1: #说明是最后一步 不再让其预测 而是直接给出结束标志
                        #这是因为模型很难选择出结束标志 模型得到的都是惩罚 所以它几乎不可能预测出end标志
                        selectedAction.append(action_size-1)
                        reward=1
                        sampleReward+=int(reward) #直接给奖励值
                        memory.append((x,h,action_size-1,reward,x,h)) #因为预测出了结束 即没有增加新的节点 因为下一个状态还是state
                    else:
                        #根据状态选择action
                        action,next_h=agent.act(x,h,selectedAction)
                        if action==action_size-1 and step==0:
                            #第一个就预测出了结束 则不能结束
                            while True:
                                action,next_h=agent.act(x,h,selectedAction)
                                if action!=action_size-1:
                                    break
                        #执行该action 得到reward 并更新状态
                        reward,_=agent.step(action,selectedAction,y)
                        if type(_)!=int: #说明预测的不是结束符
                            g=_
                            #将选择的action加入到selectedAction中
                            selectedAction.append(int(action))
                            sampleReward+=int(reward)
                            next_x=agent.new_state(f,g) #得到新时刻的输入xt
                            #将经验放入经验池
                            # 记忆这次transition
                            memory.append((x,h, action, reward, next_x,next_h))
                            #用新时刻的状态替代原先的状态
                            x=next_x
                            h=next_h

                        else: #预测出了结束符
                            selectedAction.append(int(action))
                            sampleReward+=int(reward)
                            memory.append((x,h,action,reward,x,h))

                # 用之前的经验训练agent
                if len(memory) >BATCH_SIZE:
                    epochLoss.append(float(agent.replay(BATCH_SIZE)))
                gc.collect()
                # params = agent.model.state_dict()
                # for k, v in params.items():
                #     print(k)  # 打印网络中的变量名
                #     print(v[:5])  # 打印conv1的weight
                #     # print(params['fc2.bias'])  # 打印conv1的bias
            if e%1==0:
                #每10轮评估一下模型在测试集上和验证集上的表现
                val_eva=Evaluate(X_val,Y_val)
                avg_reward, avg_jaccard, avg_recall,avg_precision,avg_f,ddiRate=val_eva.evaluate()
                #将结果写入到文件中
                # 写结果文件
                file = open('MIMIC-III/result.txt', 'a+')

                file.write('{},\t{},\t{},\t{},\t{},\t{}\t{},\t{}'.format(e,float(sum(epochLoss)*1.0/len(epochLoss)),avg_reward,avg_jaccard,avg_recall,avg_precision,avg_f,ddiRate))
                file.write('\n')
                if avg_jaccard>min_jaccard:
                    min_jacard=avg_jaccard
                    #保存当前模型
                    torch.save(agent,'MIMIC-III/agent.pkl')
                    #print(agent.model_params[1])
                    #每当结果提升了以后保存模型 并且在得到测试集上的结果
                    testModel()
                del avg_reward,avg_jaccard,ddiRate,epochLoss,avg_precision,avg_f
                gc.collect()
                file.close()





