import gym
import numpy as np 
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import cv2


class FruitEnv:
    def __init__(self):
        self.reset()
        
    def reset(self):
        game = np.zeros((10,8))
        game[9,3:6] = 1.
        self.state = game
        self.t = 0
        self.done = False
        return self.state[np.newaxis,:].copy()
    
    def step(self, action):
        reward = 0.
        game = self.state
        
        if self.done:
            print('Call step after env is done')
            
        if self.t==200:
            self.done = True
            return game[np.newaxis,:].copy(),10,self.done
        
        # according to action, move the plate
        if action==0 and game[9][0]!=1:
            game[9][0:7] = game[9][1:8].copy()
            game[9][7] = 0
        elif action==1 and game[9][7]!=1:
            game[9][1:8] = game[9][0:7].copy()
            game[9][0] = 0
        #judge whether fruit land on the ground or the plate
        if 1 in game[8]:
            fruit = np.where(game[8]==1)
            if game[9][fruit] != 1:
                reward = -1.
                self.done = True
            else:
                reward = 1.
            game[8][fruit] = 0.
        game[1:9] = game[0:8].copy()
        game[0] = 0
        
        if self.t%8==0:
            idx = random.randint(a = 0, b = 7)
            game[0][idx] = 1.
        self.t += 1
        
        return game[np.newaxis,:].copy(),reward,self.done


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        # 16*10*8
        self.maxpool = nn.MaxPool2d(2,2)
        # 16*5*4
        self.fc = nn.Sequential(
            nn.Linear(16*5*4, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DQNAgent():
    def __init__(self, network, eps, gamma, lr):
        self.network = network
        self.eps = eps
        self.gamma = gamma
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def learn(self, batch):
        s0, a0, r1, s1, done = zip(*batch)
        
        n = len(s0)
        
        s0 = torch.FloatTensor(s0)
        s1 = torch.FloatTensor(s1)
        r1 = torch.FloatTensor(r1)
        a0 = torch.LongTensor(a0)
        done = torch.BoolTensor(done)
        
        increment = self.gamma * torch.max(self.network(s1).detach(), dim=1)[0]
        
        y_true = r1+increment
        y_pred = self.network(s0)[range(n),a0]
        
        loss = F.mse_loss(y_pred, y_true)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def sample(self, state):
        '''
        epsilon explore to choose the next action
        '''
        state = state[np.newaxis,:]
        action_value = self.network(torch.FloatTensor(state))
        
        if random.random()<self.eps:
            return random.randint(0, 2)
        else:
            max_action = torch.argmax(action_value,dim=1)
            return max_action.item()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input:nx1x10x8 tensor
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        # nx16x10x8 tensor
        self.maxpool = nn.MaxPool2d(2,2)
        # nx16x5x4 tensor
        self.fc = nn.Sequential(
            nn.Linear(16*5*4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


gamma = 0.9
eps_high = 0.9
eps_low = 0.1
num_episodes = 500
LR = 0.001
batch_size = 5
decay = 200
env = FruitEnv()
net = DQN()
agent = DQNAgent(net,1,gamma,LR)

replay_lab = deque(maxlen=5000)

state = env.reset()

for episode in range(num_episodes):
    agent.eps = eps_low + (eps_high-eps_low) *\
    (np.exp(-1.0 * episode/decay))
    
    s0 = env.reset()
    while True:
        a0 = agent.sample(s0)
        s1, r1, done = env.step(a0)
        
        replay_lab.append((s0.copy(),a0,r1,s1.copy(),done))
        
        if done:
            break
        
        s0 = s1
    
        if replay_lab.__len__()>=batch_size:
            batch = random.sample(replay_lab,k=batch_size)
            loss = agent.learn(batch)
        
    if (episode+1)%50==0:
        print("Episode: %d, loss: %.3f"%(episode+1,loss))
        #score = evaluate(agent,env,10)
        #print("Score: %.1f"%(score))


from matplotlib.colors import ListedColormap
 
cmap_light = ListedColormap(['white','red'])
from IPython import display

agent.eps = 0
s = env.reset()
i = 0
img_path = "D:\\Users\\22063\\Desktop\\images\\"
while True:
    a = agent.sample(s)
    s, r, done = env.step(a)
    
    if done:
        break
        
    img = s.squeeze()
    plt.savefig(img_path + 'graph'+ str(i) + '.png')
    plt.imshow(img, cmap=cmap_light)
    i += 1
    print(i)
    #plt.show()
    #display.clear_output(wait=True)


#extract a picture
size = (640, 480)
#complete the creation of write into object,The first parameter is the name of the composite video, the second is the encoder that can be used, the third is the frame rate, i.e. how many images are displayed per second, and the fourth is the image size information.
videowrite = cv2.VideoWriter('D:\\Users\\22063\\Desktop\\video_output\\test.mp4',-1,10,size)#20 is frame number,size is the size of picture
img_array=[]
for filename in ['D:\\Users\\22063\\Desktop\\images\\graph{0}.png'.format(i) for i in range(200)]:
    img = cv2.imread(filename)
    if img is None:
        print(filename + " is error!")
        continue
    img_array.append(img)
for i in range(200):
    videowrite.write(img_array[i])
print('end!')