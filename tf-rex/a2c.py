from environment import Environment
from preprocessor import Preprocessor

import math
import random
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


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    
    def __init__(self, h, w, output):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 24, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2))
        linear_input_size = convw * convh * 32
        head_output_size = (linear_input_size+output)//2
        self.head = nn.Linear(linear_input_size, head_output_size)
        self.tail = nn.Linear(head_output_size,output)
        self.valuefc = nn.Linear(head_output_size, 1)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        x = F.leaky_relu(x)
        return F.softmax(self.tail(x).float(), dim=-1), self.valuefc(x).float()

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            dist =  torch.distributions.Categorical(policy_net(state)[0].squeeze())
            action = dist.sample()
            return action
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def plot_rewards():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model(state_list, reward_list, action_list):
    R = 0
    n = 5
    rewards = []

    states = torch.stack(state_list).to(device).squeeze()
    actions = torch.Tensor(action_list).long().to(device)
    logits, q = policy_net(states)
    log_prob = torch.log(logits.squeeze().gather(1,actions[:,None]).squeeze())

    length = len(reward_list)
    for r in reward_list[::-1]:
        R = r + GAMMA * R
        rewards.insert(0,R)

    loss2 = F.smooth_l1_loss(torch.Tensor(rewards).to(device), q)

    for i in range(len(rewards)):
        if length - i + n < length:
            rewards[length-i-1] += (GAMMA**n)*q.detach().squeeze()[length-i-1+n]

    rewards = torch.Tensor(rewards).to(device)
    # rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    loss1 = -torch.sum(torch.mul(log_prob, (rewards - q.detach().squeeze())))
    loss = loss1 + loss2

    # without baseline
    # loss = -torch.sum(torch.mul(log_prob, (rewards)))
    
    # with baseline
    # loss1 = -torch.mean(torch.mul(log_prob, (rewards - q.detach().squeeze())))
    # loss2 = F.smooth_l1_loss(rewards, q)
    # loss = loss1 + loss2

    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


if __name__ == "__main__":
    env = Environment("127.0.0.1", 9090)
    BATCH_SIZE = 512

    GAMMA = 0.999
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
    steps_done = 0

    lr = 1e-4
    optimizer = optim.Adam(policy_net.parameters(), lr)
    num_episodes = 1000
    

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        frame, _, done = env.start_game()
        frame = preprocessor.process(frame)
        # print (frame)
        state = preprocessor.get_initial_state(frame)
        state = torch.tensor(state).unsqueeze(0).float().to(device)
        cum_rewards = 0

        # Initialize the environment and state
        state_list = []
        reward_list = []
        action_list = []

        while not done:

            # Select and perform an action
            action = select_action(state)
            action_str = Environment.actions[action.item()]
            print("action: ", action_str)
            frame, reward, done = env.do_action(action.item())
            frame = preprocessor.process(frame)
            next_state = preprocessor.get_updated_state(frame)
            next_state = torch.tensor(next_state).unsqueeze(0).float().to(device)
            
            reward = torch.tensor([reward], device=device).float()
            cum_rewards += reward
            
            # Store the transition in memory
            state_list.append(state)
            reward_list.append(reward)
            action_list.append(action)

            
            # Move to the next state
            state = next_state

        # Model is only optimized after entire policy is ran
        optimize_model(state_list, reward_list, action_list)

        
        episode_rewards.append(cum_rewards)
        plot_rewards()
            
        # Save weights
        if i_episode%50==0:
            torch.save(policy_net.state_dict(), 'reinforce_reward_%.1f_ep_%d.pt'%(cum_rewards,i_episode))
    
    print('Complete')
    savename = "final"

    # env.render()
    # env.close()
    plt.ioff()
    plt.savefig("dinosaur" + savename + ".png")
    torch.save(policy_net.state_dict(), 'dino'+savename+'.pt')