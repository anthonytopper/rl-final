#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        self.action = env.get_action_space()

        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        self.model = DQN().to(self.device)
        self.model_target = DQN().to(self.device)
        self.episode = 100000
        self.max_steps_per_episode = 14000
        self.update_target_network = 10000
        self.epsilon = 1.0
        self.min_epsilon = 0.1
        self.step_epsilon =  (self.epsilon-self.min_epsilon)/(1E6)
        self.env=env
        self.history = []
        self.buffer_size = min(args.history_size//5,2000)
        self.history_size = args.history_size
        self.learning_rate = 1e-4
        self.name = args.name
        self.batch_size=32
        self.gamma = 0.99
        self.priority=[]
        self.w = 144
        self.h = 256
        self.mode = args.mode
        self.delay = args.delay
        self.epoch = args.continue_epoch
        if args.test_dqn or self.epoch>0:
            #you can load your model here
            print('loading trained model')
            ###########################
            self.model.load_state_dict(torch.load(self.name+'.pth', map_location=self.device))
            self.model_target.load_state_dict(torch.load(self.name+'.pth', map_location=self.device))
            # YOUR IMPLEMENTATION HERE #
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.model.eval()
        with torch.no_grad():
            if test == False:
                if np.random.random()<self.epsilon or len(self.history) < self.buffer_size:
                    action = int(np.random.choice([0,1],1)[0])
                else:
                    obs = torch.from_numpy(observation).to(self.device).float()
                    action_prob = self.model(obs.view(1,12,self.h,self.w))
                    action = torch.argmax(action_prob).detach().item() 
                return action

            else:
                observation = np.swapaxes(observation,0,2)/255.
                obs = torch.from_numpy(observation).to(self.device).float()
                action_prob = self.model(obs.view(1,12,self.h,self.w))
                action = torch.argmax(action_prob).detach().item()

                return self.action[action]
        ###########################

        
    
    def push(self,state,action,reward,done,state_next,smooth=None):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.history.append(np.array([state,action,reward,done,state_next,smooth]))

        
        if len(self.history) > self.history_size:
            self.history.pop(0)


        ###########################
        
        
    def replay_buffer(self,refresh=False):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if 'prioritized' in self.mode.split('_'):
            if refresh :
                self.priority = np.zeros(len(self.history))
                for i in range(len(self.history)):
                    max_reward, _ = torch.max(self.model_target(torch.from_numpy(self.history[i][4]).to(self.device).float().view(1,12,self.h,self.w)), axis=1)
                    max_reward = max_reward.detach().item() 
                    Q = self.model(torch.from_numpy(self.history[i][0]).to(self.device).float().view(1,12,self.h,self.w))[0,self.history[i][1]].detach().item()
                    self.priority[i] = abs((self.history[i][2] + self.gamma * max_reward - Q)) 
                self.priority = self.priority / sum(self.priority)
                return 0
            priority = np.zeros(len(self.history))
            priority[:len(self.priority)] = self.priority
            if sum(priority)==0:
                indices = np.random.choice(range(len(self.history)), size=self.batch_size)
            else:
                indices = np.random.choice(range(len(self.history)), size=self.batch_size,p=priority)

            ###########################
            return indices
        else:
            return np.random.choice(range(len(self.history)), size=self.batch_size)
        

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        

        episode_reward_history = []
        best_reward = -10
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,momentum=0.5)
        loss_fn = torch.nn.SmoothL1Loss()
        frame_count = 0
        if self.epoch>0:
            f = open(self.name+'.txt', "a")
        else:
            f = open(self.name+'.txt', "w")
        done=False
        for ep in range(self.epoch,self.episode):
            state = self.env.reset()
            state = np.swapaxes(state,0,2)/255.
            episode_reward = 0
            pre_action=[0,0,0,0,0,0,0,0,0,0]
            smooth=0
            for timestep in range(0, self.max_steps_per_episode):
                frame_count += 1
                action = self.make_action(state,test=False)
                if done:
                    action=1

                # Decay
                self.epsilon -= self.step_epsilon
                self.epsilon = max(self.epsilon, self.min_epsilon)

                # next frame
                state_next, reward, done, _ = self.env.step(self.action[action])
                state_next = np.swapaxes(state_next,0,2)/255.
                episode_reward += reward
                # print(reward)
                #normalize reward
                # reward = np.sign(reward)
                # Save actions and states in replay buffer
                

                state = state_next
                if 'smooth1' in self.mode.split('_'):
                    pre_action.pop(0)
                    pre_action.append(action)
                    smooth = float(np.mean(pre_action)-0.5)
               
                self.push(state,action,reward,done,state_next,smooth)
                        
                
                if frame_count % 8 == 0 and len(self.history) >= self.buffer_size:
                    if frame_count%self.history_size//10==0 and 'prioritized' in self.mode.split('_'):
                        #update priority vector
                        self.replay_buffer(refresh = True)
                    indice = self.replay_buffer()
                    self.model.train()
                    # data_batch = torch.from_numpy(np.array(self.history)[indice]).to(self.device).float()
                    state_sample = torch.from_numpy(np.array([self.history[i][0] for i in indice])).to(self.device).float()
                    action_sample = torch.from_numpy(np.array([self.history[i][1] for i in indice])).to(self.device).float()
                    rewards_sample = torch.from_numpy(np.array([self.history[i][2] for i in indice])).to(self.device).float()
                    done_sample = torch.from_numpy(np.array([self.history[i][3] for i in indice])).to(self.device).float()
                    next_state_sample = torch.from_numpy(np.array([self.history[i][4] for i in indice])).to(self.device).float()
                    smooth_sample = torch.from_numpy(np.array([self.history[i][5] for i in indice])).to(self.device).float()
                    future_rewards = self.model_target(next_state_sample)

                    max_reward, _ = torch.max(future_rewards, axis=1)
                    updated_q_values = rewards_sample + self.gamma * max_reward
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample
                    mask = F.one_hot(action_sample.long(), 2).to(self.device).float()

                    q_values = self.model(state_sample)
                    q_action = torch.sum(q_values * mask, axis=1)
                    loss = loss_fn(q_action,updated_q_values)
                    
                    if 'smooth1' in self.mode.split('_') and self.delay < ep:
                        penalty = torch.abs((ep-self.delay)/self.episode * torch.sum(smooth_sample))
                        loss +=  penalty
                        
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
                    optimizer.step()
                    

                if frame_count % self.update_target_network == 0:
                    self.model_target.load_state_dict(self.model.state_dict())

                if done:
                    break
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > 30:
                del episode_reward_history[:1]
            running_reward = np.mean(episode_reward_history)
#             if ep%500==0:
#                 print("Episode:\t{},\t Avereged reward: {:.2f}\n".format(ep,running_reward))
            f.write("Episode:\t{},\t Avereged reward: {:.2f}\n".format(ep,running_reward))
            if running_reward> best_reward:
                best_reward= running_reward
                torch.save(self.model.state_dict(), self.name+'.pth')
        f.close()

        ###########################
