from environment import Environment
from preprocessor import Preprocessor

import math
import random
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# from dqn import DQN
# from sarsa import DQN
# from reinforce import DQN
# from a2c import DQN
from one_step_actor_critic import DQN 



# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

#python -m http.server
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def select_action(state):
    policy_net.eval()
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.

        # MC methods
        # dist =  torch.distributions.Categorical(policy_net(state)[0].squeeze())
        # action = dist.sample()
        # return action
        return policy_net(state)[0].max(1)[1].view(1, 1).long()

        # TD methods
        # return policy_net(state).max(1)[1].view(1, 1).long()
    
    
if __name__=='__main__':
    env = Environment("127.0.0.1", 9090)
    BATCH_SIZE = 32
    

    width = 80
    height = 80
    preprocessor = Preprocessor(width, height)
    
    # change fname here
    fname = 'results/1step.pt'

    n_actions = len(env.actions.keys())
    policy_net = DQN(height, width, n_actions).float().to(device)
    policy_net.load_state_dict(torch.load(fname, map_location=device))
    
    
    

    episode_rewards = []
    episode_durations = []
    episode_losses = []
    steps_done = 0

    
    
    
    num_episodes = 1500
    

    while True:
        # Initialize the environment and state
        frame, _, done = env.start_game()
        frame = preprocessor.process(frame)
        state = preprocessor.get_initial_state(frame)
        state = torch.tensor(state).unsqueeze(0).float().to(device)
        cum_steps = 0
        step = 0
        global cum_loss
        cum_loss = 0
        cum_reward = 0
        

        
        while not done:
            # Select and perform an action
            action = select_action(state)
            action_str = Environment.actions[action.item()]
#            print("action: ", action_str)
            frame, reward, done = env.do_action(action.item())
            frame = preprocessor.process(frame)
            next_state = preprocessor.get_updated_state(frame)
            next_state = torch.tensor(next_state).unsqueeze(0).float().to(device)
            
            reward = torch.tensor([reward], device=device).float()
            cum_reward += reward
            cum_steps += 1

            


            # Move to the next state
            state = next_state

            
            
            step+=1
