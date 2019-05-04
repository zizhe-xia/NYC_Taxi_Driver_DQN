# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:41:17 2019

@author: xiazizhe
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pandas as pd
import random

from deepqmodel import DeepQNetwork

class RandomAgent(object):
    
    def __init__(self, maxMemorySize, action_space=[0,1,2], agentName='Agent'):
        self.action_space = action_space
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCntr = 0
        self.agentName = agentName

    def storeTransition(self, observation, action, reward, observation_):
        if self.memCntr < self.memSize:
            self.memory.append([observation, action, reward, observation_])
        else:            
            self.memory[self.memCntr%self.memSize] = [observation, action, reward, observation_]
        self.memCntr += 1 
        
    def chooseAction(self, observation):
        action = np.random.choice(self.action_space)
        self.steps += 1
        return action
    
    def clearMemory(self):
        self.memory = []
        self.memCntr = 0
    
    def resetSteps(self):
        self.steps = 0
        self.learn_step_counter = 0
    
    def resetAgent(self):
        self.clearMemory()
        self.resetSteps()
        
    def saveModelState(self, output_dir='./../output/'):
        torch.save(self.Q_eval.state_dict(), output_dir + self.agentName + '.ckpt')
    
    def loadModelState(self, file):
        self.Q_eval = DeepQNetwork(self.ALPHA).double()
        self.Q_eval.load_state_dict(torch.load(file))
        self.Q_eval.eval()
        self.Q_next = DeepQNetwork(self.ALPHA).double()
        self.Q_next.load_state_dict(torch.load(file))
        self.Q_next.eval()
    
    def saveModelCheckpoint(self, output_dir='./../output/'):
        torch.save(self.Q_eval, output_dir + self.agentName + 'entire_deepq.ckpt')
        
    def loadModelCheckpoint(self, file):
        model = torch.load(file)
        model.eval()


class LearningAgent(RandomAgent):
    def __init__(self, gamma, epsilon, alpha, 
                 maxMemorySize, epsEnd=0.05, 
                 action_space=[0,1,2], replace=500, agentName='LearningAgent'):
        ### actions: 0-accept, 1-reject&wait, 2-reject&back
        self.GAMMA = gamma # discount factor
        self.EPSILON = epsilon # 1- epsilon: probability of choosing the optimal
        self.EPS_END = epsEnd
        self.ALPHA = alpha
        self.replace_target_cnt = replace
        super(LearningAgent, self).__init__(maxMemorySize, action_space, agentName)
        ### implemented in super class
#        self.action_space = action_space
#        self.memSize = maxMemorySize
#        self.steps = 0
#        self.learn_step_counter = 0
#        self.memory = []
#        self.memCntr = 0
        self.Q_eval = DeepQNetwork(alpha).double()
        self.Q_next = DeepQNetwork(alpha).double()

    ### implemented in super class
#    def storeTransition(self, observation, action, reward, observation_):
#        if self.memCntr < self.memSize:
#            self.memory.append([observation, action, reward, observation_])
#        else:            
#            self.memory[self.memCntr%self.memSize] = [observation, action, reward, observation_]
#        self.memCntr += 1
        
    def chooseEpsilonAction(self, observation):
        '''
        Use epsilon exploration
        '''
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        if rand < 1 - self.EPSILON:
            action = torch.argmax(actions).item() ### need to change here
        else:
            action = np.random.choice(self.action_space)
        self.steps += 1
        return action
    
    def chooseAction(self, observation):
        actions = self.Q_eval.forward(observation)
        action = torch.argmax(actions).item()
        self.steps += 1
        return action
    
    def learn(self, batch_size, num_steps=1):
        self.Q_eval.optimizer.zero_grad()
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

#        if self.memCntr+batch_size < self.memSize:            
#            memStart = int(np.random.choice(range(self.memCntr)))
#        else:
#            memStart = int(np.random.choice(range(self.memSize-batch_size-1)))
        
        for i in range(num_steps):
            miniBatch = random.sample(self.memory, batch_size) # instead of [memStart:memStart+batch_size]
            miniBatch = pd.DataFrame(miniBatch, columns=['observation', 'action', 'reward', 'observation_']).values
    
            # convert to list because memory is an array of numpy objects
            Qpred = self.Q_eval.forward(torch.stack(list(miniBatch[:,0][:])).double()).to(self.Q_eval.device)
            Qnext = self.Q_next.forward(torch.stack(list(miniBatch[:,3][:])).double()).to(self.Q_eval.device)       
            
#            maxA = torch.argmax(Qnext, dim=1).to(self.Q_eval.device)
            rewards = torch.Tensor(list(miniBatch[:,2])).double().to(self.Q_eval.device)
#            Qtarget = Qpred.clone()
            Qpred_max = Qpred.max(dim=1)[0]
#            Qpred_max = Variable(Qpred_max, requires_grad=True).to(self.Q_eval.device)
            print(Qpred_max.requires_grad)
            Qtarget_max = rewards.squeeze() + self.GAMMA * Qnext.max(dim=1)[0]
            Qtarget_max = Variable(Qtarget_max, requires_grad=False).to(self.Q_eval.device)
            print(Qtarget_max.requires_grad)
#            for i in range(len(rewards)):
#                m = maxA[i].item()
#    #            print(Qtarget[i][m])
#    #            print(rewards[i] + torch.max(Qnext[i]))
#                Qtarget[i][m] = rewards[i] + self.GAMMA * torch.max(Qnext[i])
#    #        Qtarget[:,maxA] = rewards + self.GAMMA*torch.max(Qnext[1]) ##### Change here
            
            if self.steps > 200:
                if self.EPSILON - 1e-3 > self.EPS_END:
                    self.EPSILON -= 1e-3
                else:
                    self.EPSILON = self.EPS_END
    
            #Qpred.requires_grad_()        
            loss = self.Q_eval.loss(Qpred_max, Qtarget_max).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()
            self.learn_step_counter += 1
            print(list(self.Q_eval.parameters())[0])
        


