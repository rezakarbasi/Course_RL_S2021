#%% hyperparameter
gamma = 1
alpha = 0.1
MAX_EPISODE = 4000

#%%

import torch
torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

sigmas = []

class Policy(nn.Module):
    def __init__(self, s_size=3, h_size=20):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)

        self.mu = nn.Linear(h_size, 1)
        self.sigma = nn.Linear(h_size, 1)

        self.distribution = torch.distributions.Normal
        
        torch.nn.init.constant_(self.sigma.bias,0.1)
        
        
    # output : (mean , var) 
    # note : var>0
    def forward(self, state):
        state = torch.Tensor(state)
        x = F.relu(self.fc1(state))
        # x = self.fc2(x)

        mu = self.mu(x)[0]
        sigma = F.softplus(self.sigma(x)[0]) + 0.025
        m = self.distribution(mu, sigma)
        
        # print(sigma.item())
        sigmas.append(sigma.item())

        a = m.sample()
        log_prob = m.log_prob(a)

        return a,log_prob

policy = Policy(h_size=50)
optimizer = optim.Adam(policy.parameters(), lr=1e-4)
#%%
import gym

env = gym.make("Pendulum-v0")

plot_rewards = []
action_log = []
sigma_log = []

for episode in range(MAX_EPISODE+1):
# for episode in range(1):
    if episode%10==0:
        print('\repisode : {}'.format(episode),end='')
    
    state = env.reset()
    done = False
    
    rewards = []
    saved_log_probs = []
    
    while done==False:
        action, log_prob = policy(state)
        saved_log_probs.append(log_prob)
        obs, reward, done, info = env.step([action])
        rewards.append(reward)
        
        action_log.append(action.item())
        
        state = obs
        # env.render()
        
        if episode%200==0:
            env.render()
            

    plot_rewards.append(sum(rewards))    
    rewards.reverse()
    saved_log_probs.reverse()
    
    G = 0
    policy_loss = torch.Tensor([0])
    for r,l in zip(rewards,saved_log_probs):
        G = G*gamma + r
        policy_loss+=-l * G
    
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    if episode%200==0:
        filtered=[plot_rewards[0]]
        for r in plot_rewards:
            filtered.append(filtered[-1]+0.01*(r-filtered[-1]))
            
        plt.plot(filtered)
        plt.savefig('testBadam_50N-4.png')
        plt.close()
        
        sigma_log.append(np.mean(sigmas))
        # print(sigma_log)
        sigmas = []

        plt.plot(sigma_log)
        plt.savefig('tetsBadam_50N-4.png')
        plt.close()
        
        import time
        time.sleep(1)


    