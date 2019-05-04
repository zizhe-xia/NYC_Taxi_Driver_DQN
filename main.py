# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:28:41 2019

@author: XIA Zizhe
"""

from utils import plot_learning
import pandas as pd
import numpy as np
import random
import pickle
import time

from deepqmodel import DeepQNetwork
from agent import LearningAgent, RandomAgent
import environment

def initialize(env, brain):
    while brain.memCntr < brain.memSize:
        observation = env.reset()
        done = False
        while not done:
            # 0-accept ride, 1-reject and wait, 2-reject and back
            action = random.choice(env.get_action_space())
            observation_, reward, done = env.step(action)
            brain.storeTransition(observation, action, reward, observation_)
            observation = observation_
    print('done initializing memory')

def train_agent(env, brain, numGames, batch_size=64, num_steps=2):
    scores = []
    steps = []
    epsHistory = []
    for i in range(numGames):
        start = time.time()
        print('starting learning game ', i+1, 'epsilon: %.4f' % brain.EPSILON)
        epsHistory.append(brain.EPSILON)        
        done = False
        observation = env.reset()
        step_begin = agent.steps
        score = 0
#        lastAction = 0
        while not done:
#            if len(frames) == 3:
#                action = brain.chooseEpsilonAction(observation)
#                frames = []
#            else:
#                action = lastAction
            action = brain.chooseEpsilonAction(observation)
            observation_, reward, done = env.step(action)
            score += reward
            brain.storeTransition(observation, action, reward, observation_)
            observation = observation_
            brain.learn(batch_size, num_steps)
#            lastAction = action
        step_end = agent.steps
        scores.append(score)
        steps.append(step_end - step_begin)
        end = time.time()
        print('score:',score)
        print('%d seconds for the current episode' % (end-start))
    return scores,steps,epsHistory

def test_agent(env, brain, numGames):
    scores = []
    steps = []
    for i in range(numGames):
        print('starting testing game ', i+1)     
        done = False
        observation = env.reset()
        step_begin = agent.steps
        score = 0
#        lastAction = 0
        while not done:
            action = brain.chooseAction(observation)
            observation_, reward, done = env.step(action)
            score += reward
#            brain.storeTransition(observation, action, reward, observation_)
            observation = observation_
#            brain.learn(batch_size)
#            lastAction = action
        step_end = agent.steps
        scores.append(score)
        steps.append(step_end - step_begin)
        print('score:',score)
    return scores, steps

if __name__ == '__main__':
    ##### Set parameters
    ### Agent and Env params
    output_dir = './../output/'
    data_months = 3
    gamma = 1
    epsilon = 1.0
    alpha = 0.005
    maxMemorySize = 200
    epsEnd = 0.05
    action_space = [0,1,2]
    replace = 500
    ### Training params
    numGames = 100
    batch_size = 32
    num_steps = 2
#    ##### Initialization for learning Agent
#    env = environment.NYCTaxiEnv(data_months=data_months)
#    brain = LearningAgent(gamma, epsilon, alpha, maxMemorySize, epsEnd,
#                          action_space, replace)
#    start = time.time()
#    initialize(env, brain)
#    end = time.time()
#    print('%d seconds passed for initialization' %(end - start))
#    pickle.dump(brain, open('brain_init.pickle','wb'))
#    ##### Training Learning Agent
#
#    scores,steps,epsHistory = train_agent(env, brain, numGames, batch_size, num_steps)
#    brain.saveModelCheckpoint('./../output/')
#    brain.saveModelState('./../output/')
#    filename = str(numGames) + 'Games' + 'Gamma' + str(brain.GAMMA) + \
#               'Alpha' + str(brain.ALPHA) + 'Memory' + str(brain.memSize)
#    df = pd.DataFrame({'scores':scores,'steps':steps,'epsHistory':epsHistory})
#    df.to_csv(output_dir + 'LearningAgent' + filename + '.csv', index=False)
    
    ##### Random agent
    env = environment.NYCTaxiEnv(data_months=data_months)
    randombrain = RandomAgent(maxMemorySize=1,action_space=action_space)   
    numGames = 100
    scores,steps = test_agent(env, randombrain, numGames)
    fileName = str(numGames) + 'Games'
    df = pd.DataFrame({'scores':scores,'steps':steps})
    df.to_csv(output_dir + 'RandomAgent' + fileName + '.csv', index=False)
    
    