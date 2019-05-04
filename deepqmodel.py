# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:28:12 2019

@author: XIA Zizhe
"""
import torch
import torch.nn as nn
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, ALPHA):
        # ALPHA is the learning rate of the deep q network
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 3)
        self.activate = nn.ReLU()
        #self.optimizer = optim.SGD(self.parameters(), lr=self.ALPHA, momentum=0.9)
        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss = nn.SmoothL1Loss() #Huber loss
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     
        self.to(self.device)

    def forward(self, observation):
        actions = torch.DoubleTensor(observation).to(self.device)
        actions = self.activate(self.fc1(actions))
        actions = self.activate(self.fc2(actions))
        actions = self.activate(self.fc3(actions))
        actions = self.fc4(actions)
        return actions

