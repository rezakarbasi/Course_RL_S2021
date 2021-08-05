import gym
import numpy as np

env = gym.make('MountainCar-v0')
env.close()
env.reset()

maxPos = 0.61
minPos = -1.3

maxVel = 0.1
minVel = -0.1

precisionPos=0.05
precisionVel=0.01

gamma = 0.95
epsilon = 1
alpha = 0.1


Q = np.zeros((int((maxPos-minPos)//precisionPos),int((maxVel-minVel)//precisionVel),3))

def EpsilonGreedy(s):
    if np.random.rand()<epsilon:
        return np.random.choice(range(3))

    s = GetState(s)
    return np.argmax(Q[s[0],s[1]])

def GetState(s):
    pos = s[0]
    vel = s[1]
    
    posIdx = (pos-minPos)//precisionPos
    velIdx = (vel-minVel)//precisionVel
    
    return int(posIdx),int(velIdx)

def SARSA(s,a,r,sp,ap):
    s = GetState(s)
    sp = GetState(sp)
    Q[s[0],s[1],a] = Q[s[0],s[1],a] + alpha * (r+gamma*Q[sp[0],sp[1],ap]-Q[s[0],s[1],a])

def DoGreedy():
    env.close()

    state = env.reset()
    done = False

    while done==False:
        s = GetState(state)
        action = np.argmax(Q[s[0],s[1]])

        newS,reward,done,info = env.step(action)
        
        env.render()
        
        state = newS    
 
success = 0
rewards = []
for i in range(6000):
    done = False
    
    
    rewards.append(0)

    if i%50 ==49:
        epsilon*=0.97
    

    print('\r{:4} epsiodes - success = {:3} - eps= {:1.4f} - min Q = {:4.2f} - max Q = {:4.2f}'.format(i,success,epsilon,np.min(Q),np.max(Q)),end='')

    if i%1000 == 0:
        success = 0
        
    state = env.reset()
    newAction = EpsilonGreedy(state)
    while done==False:
        
        newS,reward,done,info = env.step(newAction)

        action = newAction
        newAction = EpsilonGreedy(state)
        SARSA(state, action, reward, newS, newAction)
        state = newS
        
        rewards[-1]+=reward
        
        if i%1000 == 0:
            env.render()
        
        
        if done and rewards[-1]>-190:
            success+=1
  
env.close()
  
import matplotlib.pyplot as plt
plt.plot(rewards)
plt.show()
# plt.close()

DoGreedy()
