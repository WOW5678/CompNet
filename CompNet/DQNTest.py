# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/1 0001 上午 10:03
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 使用pytorch框架实现DQN 网络模型
"""
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim  as optim
import torch.nn.functional as F
import torchvision.transforms as T

env=gym.make('CartPole-v0').unwrapped

#set up matplotlib
is_ipython='inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import  display
plt.ion()
#if gup is to be used
device=torch.device('cuda' if torch.cuda.is_available() else "cpu")

#回放记忆
Transition=namedtuple('Transition',('state','action','next_state','reward'))

class ReplayMemory(object):
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory=[]
        self.position=0
    def push(self,*args):
        "save a transition"
        if len(self.memory)<self.capacity:
            self.memory.append(None)
        self.memory[self.position]=Transition(*args)
        #self.position指明要放入transition的位置
        self.position=(self.position+1)%self.capacity
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    def __len__(self):
        return len(self.memory)

# Q NET
#使用CNN网络预测每个行为的Q值，返回两个值 Q(s,left)和Q(s,right) 输入是s
class DQN(nn.Module):
    def __init__(self,h,w):
        super(DQN, self).__init__()
        self.conv1=nn.Conv2d(3,16,kernel_size=5,stride=2)
        self.b1=nn.BatchNorm2d(16)
        self.conv2=nn.Conv2d(16,32,kernel_size=5,stride=2)
        self.b2=nn.BatchNorm2d(32)
        self.conv3=nn.Conv2d(32,32,kernel_size=5,stride=2)
        self.b3=nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size,kernel_size=5,stride=2):
            return (size-(kernel_size-1)-1)//stride +1
        convw=conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh=conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size=convh*convw*32
        self.head=nn.Linear(linear_input_size,2) #448 or 512
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self,x):
        x=F.relu(self.b1(self.conv1(x)))
        x=F.relu(self.b2(self.conv2(x)))
        x=F.relu(self.b3(self.conv3(x)))
        print('x.size:',x.size())
        #view的作用是将一个多行的tensor拼接成一个单行的tensor
        return self.head(x.view(x.size(0),-1))

#input extraction
#The code below are utilities for extracting and processing rendered images from the environment.
# It uses the torchvision package, which makes it easy to compose image transforms.
# Once you run the cell it will display an example patch that it extracted.
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()

# Training Process
BATCH_SIZE=128
GAMMA=0.999
EPS_START=0.9
EPS_END=0.05
EPS_DECAY=200
TARGET_UPDATE=10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen=get_screen()
_, _, screen_height, screen_width = init_screen.shape

policy_net=DQN(screen_height,screen_width).to(device)
target_net=DQN(screen_height,screen_width).to(device)
#pytorch 中的 state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系.
# (如model的每一层的weights及偏置等等)
target_net.load_state_dict(policy_net.state_dict())
#eval函数就是实现list、dict、tuple与str之间的转化
#str函数把list，dict，tuple转为为字符串
target_net.eval()

optimizer=optim.RMSprop(policy_net.parameters())
memory=ReplayMemory(10000)

steps_done=0

def selec_action(state):
    global steps_done
    sample=random.random()
    eps_threshold=EPS_END+(EPS_START-EPS_END)*\
                math.exp(-1*steps_done/EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold: #通过模型选择出action
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            print('values:',policy_net(state)[0][:5])
            return policy_net(state).max(1)[1].view(1, 1)
    else:#随机选择出一个action
        return torch.tensor([[random.randrange(0,2)]],device=device,dtype=torch.long)

episode_durations=[]

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t=torch.tensor(episode_durations,dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Durations')
    plt.plot(durations_t.numpy())

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

# Training
def optimize_model():
    if len(memory)<BATCH_SIZE:
        return
    transtions=memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch=Transition(*zip(*transtions))

    non_final_mask=torch.tensor(tuple(map(lambda s:s is not None,batch.next_state)),device=device,dtype=torch.uint8)
    non_final_next_states=torch.cat([s for s in batch.next_state if s is not None])
    state_batch=torch.cat(batch.state)
    action_batch=torch.cat(batch.action)
    reward_batch=torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values=policy_net(state_batch).gather(1,action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values=torch.zeros(BATCH_SIZE,device=device)
    next_state_values[non_final_mask]=target_net(non_final_next_states).max(1)[0].detach()
    # compute the expected Q value
    expected_state_action_values=(next_state_values*GAMMA)+reward_batch

    # Compute huber loss
    loss=F.smooth_l1_loss(state_action_values,expected_state_action_values.unsqueeze(1))

    #optimze the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters(): #将梯度固定到到一个指定的范围
        param.grad.data.clamp_(-1,1)
    optimizer.step() #将计算出的grad 应用到参数中


num_episodes=30
for i_episode in range(num_episodes):
    #初始化环境和状态
    env.reset()
    last_screen=get_screen()
    current_screent=get_screen()
    state=current_screent-last_screen

    for t in count():
        #select and perform the action
        action=selec_action(state)
        _,reward,done,_=env.step(action.item())
        reward=torch.tensor([reward],device=device)

        # 观察新的状态
        last_screen=current_screent
        current_screent=get_screen()
        if not done:
            next_state=current_screent-last_screen
        else:
            next_state=None

        #存储transition
        memory.push(state,action,next_state,reward)

        #move to the next state
        state=next_state

        # 执行一步优化
        optimize_model()
        if done:
            episode_durations.append(t+1)
            plot_durations()
            break
    # 更新target 网络
    if i_episode %TARGET_UPDATE==0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()