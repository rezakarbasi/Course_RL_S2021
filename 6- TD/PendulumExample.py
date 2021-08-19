#%%%%%%%%%%
import gym
env = gym.make("Pendulum-v0")
env.reset()
env.observation_space
env.action_space


#%%%%%%%%%%%%%
import numpy as np

# hyper parameters
max_episode = 8000
alpha = 0.1
gamma = 0.95

epsilon = 1
epsDecay = 0.999

state_min = [-1,-1,-8]
state_max = [1,1,8]
state_interval = [0.1,0.1,0.5]

action_min = -2
action_max = 2
action_interval = 0.2


#
action_dim = int((action_max-action_min)/action_interval)
state_dims = []
for minn,maxx,interval in zip(state_min,state_max,state_interval):
    state_dims.append(int((maxx-minn)/interval)+1)

# defines
def discretize_state(state):
    out = []
    for idx , s in enumerate(state):
        o = int((s-state_min[idx])/state_interval[idx])
        out.append(o)
    return out

def inverse_action(action_idx):
    out = action_min + action_interval*action_idx + action_interval/2
    return out

def choose_action(Q,state,eps):
    q = Q[state[0],state[1],state[2]]

    if np.random.rand()>eps:
        return np.argmax(q)
    else:
        return np.random.choice(range(len(q)))


def do_greedy(env,Q):
    obs = env.reset()
    done = False
    s = discretize_state(obs)
    
    while done==False:
        action_idx = choose_action(Q,s,0)
        action = inverse_action(action_idx)
        
        obs, reward, done, info = env.step([action])
        env.render()
        
        sp = discretize_state(obs)
        s=sp
            

Q = np.zeros((state_dims[0],state_dims[1],state_dims[2],action_dim))
rewards = []

for e in range(max_episode):
    if e%3==0:
        print('\repisode {:4d} \t epsilon : {:1.4f}'.format(e,epsilon),end='')
        epsilon*=epsDecay
    
    obs = env.reset()
    done = False
    s = discretize_state(obs)
    
    rewards.append(0)
    
    while done==False:
        action_idx = choose_action(Q,s,epsilon)
        action = inverse_action(action_idx)
        
        obs, reward, done, info = env.step([action])
        
        rewards[-1]+=reward
        
        sp = discretize_state(obs)
        
        td_error = reward + gamma*max(Q[sp[0],sp[1],sp[2]])- Q[s[0],s[1],s[2],action_idx]
        
        Q[s[0],s[1],s[2],action_idx] =  Q[s[0],s[1],s[2],action_idx] \
                                        + alpha * td_error
        
        
        s=sp
        
        if e%500==0:
            env.render()


print('\n\n-----------------\n\t greedy')
do_greedy(env,Q)
env.close()

import matplotlib.pyplot as plt
plt.plot(rewards)

filterd=[-1200]
for r in rewards:
    filterd.append(filterd[-1]+0.001*(r-filterd[-1]))
plt.plot(filterd)
