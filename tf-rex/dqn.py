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

class DQN(nn.Module):
    
    def __init__(self, h, w, output):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(96)
        

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2))
        
        linear_input_size = convw * convh * 96
        fc1_output_size = (linear_input_size+output)//3
        fc2_output_size = (linear_input_size+output)//3
        self.fc1 = nn.Linear(linear_input_size, fc1_output_size)
        self.fc2 = nn.Linear(fc1_output_size,fc2_output_size)
        self.fc3 = nn.Linear(fc2_output_size,output)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.float()
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.fc1(x.view(x.size(0), -1))
        x = F.leaky_relu(x)
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x).float()
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        policy_net.eval()
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1).long()
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def plot_rewards():
    # Plot Duration
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.show()
    # Plot cum rewards
    plt.figure(2)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.show()
    
    # Plot losses
    plt.figure(2)
    losses_t = torch.tensor(episode_losses, dtype=torch.float)
    
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(losses_t.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated

def optimize_model(memory,is_short_term=True):
    if len(memory) < BATCH_SIZE and is_short_term:
        return
    if not is_short_term:
        samples = np.array([len(item) for item in memory.values()],dtype=float)
        samples /= np.sum(samples)/BATCH_SIZE
        samples += 1
        transitions = []
        for idx in range(samples.size):
            transitions.extend( memory[idx].sample(int(samples[idx])) )
        
    else:
        transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    policy_net.train()
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(reward_batch.shape[0], device=device).float()

    # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Use double Q learning here
    policy_net.eval()
    target_net.eval()
    with torch.no_grad():
        next_state_action_index = policy_net(non_final_next_states).argmax(1).detach()
    
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_state_action_index.unsqueeze(1)).squeeze()
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
#    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    global cum_loss
    cum_loss += loss.item()

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    

if __name__ == "__main__":
    env = Environment("127.0.0.1", 9090)
    BATCH_SIZE = 32

#    GAMMA = 0.999
    GAMMA = 0.95
    EPS_START = 0.9
    EPS_END = 0.005
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    

    width = 80
    height = 80
    preprocessor = Preprocessor(width, height)

    n_actions = len(env.actions.keys())
    policy_net = DQN(height, width, n_actions).float().to(device)
    target_net = DQN(height, width, n_actions).float().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    

    episode_rewards = []
    episode_durations = []
    episode_losses = []
    steps_done = 0

    lr = 1e-4
    
    
    optimizer = optim.Adam(policy_net.parameters(), lr)
    
    
    
    short_term_memory = ReplayMemory(2000)
    long_term_memory_dic = {}
    long_term_memory_len = 500
    num_episodes = 1000
    is_long_term = False

    for i_episode in range(num_episodes):
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
            # Store the transition in memory
            short_term_memory.push(state, action, next_state, reward)
            try:
                long_term_memory = long_term_memory_dic[step//100]
            except KeyError:
                long_term_memory = ReplayMemory(long_term_memory_len)
                long_term_memory_dic[step//100] = long_term_memory
            long_term_memory.push(state, action, next_state, reward)
            
            if not is_long_term and step>200:
                is_long_term = True

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            if not is_long_term:
                optimize_model(short_term_memory)
            else:
                optimize_model(long_term_memory_dic,False)
            
            step+=1
        if cum_steps>0:
            episode_durations.append(cum_steps)
            episode_rewards.append(cum_reward)
            episode_losses.append(cum_loss/cum_steps)
            plot_rewards()
        

            
        
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        # Save weights
        if i_episode%10==0:
            torch.save(policy_net.state_dict(), 'meta/dino_dqn_reward_%d_ep_%d.pt'%(cum_steps,i_episode))
    
    print('Complete')
    savename = "final"

    # env.render()
    # env.close()
    plt.ioff()
    plt.savefig("dino_dqn_final.png")
    torch.save(policy_net.state_dict(), 'dino_dqn_final.pt')