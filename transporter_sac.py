'''
Soft Actor-Critic version 1
using state value function: 1 V net, 1 target V net, 2 Q net, 1 policy net
'''


import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
from reacher import Reacher
from transporter import Transporter

torch.manual_seed(1234)  #Reproducibility


GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


# intialization
# NUM_JOINTS=4
# LINK_LENGTH=[200, 140, 80, 50]
# INI_JOING_ANGLES=[0.1, 0.1, 0.1, 0.1]
NUM_JOINTS=2
LINK_LENGTH=[200, 140]
INI_JOING_ANGLES=[0.1, 0.1]
SCREEN_SIZE=1024
SPARSE_REWARD=False
SCREEN_SHOT=True
DETERMINISTIC=False
X_dim = 128 # dim of the input X_dim*X_dim
X_channel = 1 # input channel


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action

def plot(frame_idx, rewards, predict_qs):
    clear_output(True)
    plt.figure(figsize=(20,5))
    # plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    # plt.plot(predict_qs)
    plt.savefig('transporter_sac.png')
    # plt.show()

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, activation=F.relu, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.activation = activation
        
    def forward(self, state):
        x = self.activation(self.linear1(state))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, activation=F.relu, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.activation = activation
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, activation=F.relu, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = 10.
        self.num_actions = num_actions
        self.activation = activation

        
    def forward(self, state):
        x = self.activation(self.linear1(state))
        # x = self.activation(self.linear2(x))
        # x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))

        mean    = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        # print(log_std)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        deterministic evaluation provides better performance according to the original paper;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z      = normal.sample() 
        action_0 = torch.tanh(mean + std*z.to(device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range*action_0
        # action = F.leaky_relu(mean+ std*z.to(device))
        # print('action: ', action)
        # log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon)
        ''' stochastic evaluation '''
        log_prob = Normal(mean, std).log_prob(mean + std*z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        ''' deterministic evaluation '''
        # log_prob = Normal(mean, std).log_prob(mean) - torch.log(1. - torch.tanh(mean).pow(2) + epsilon) -  np.log(self.action_range)
        '''
         both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
         the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
         needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
         '''
        # print('log_prob: ', log_prob[0])
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample().to(device)
        action = self.action_range* torch.tanh(mean + std*z)
        # action = F.leaky_relu(mean+ std*z.to(device))
        
        action = mean.detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]
        
        return action


    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return (self.action_range*a).numpy()




def update(batch_size, reward_scale, gamma=0.99,soft_tau=1e-2):
    alpha = 1.0  # trade-off between exploration (max entropy) and exploitation (max Q)
    
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    # print('sample:', state, action,  reward, done)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_value    = value_net(state)
    new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)

    reward = reward_scale*(reward - reward.mean(dim=0)) /reward.std(dim=0) # normalize with batch mean and std
# Training Q Function
    target_value = target_value_net(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value # if done==1, only reward
    # print('r: ', reward[0])
    # print('tar v: ', target_value[0])
    # print('pre q 1: ', predicted_q_value1[0] )
    # print('t q : ', target_q_value[0])
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())


    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()  

# Training Value Function
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action),soft_q_net2(state, new_action))
    target_value_func = predicted_new_q_value - alpha * log_prob # for stochastic training, it equals to expectation over action
    # print('pre v: ', predicted_value[0])
    # print('pre n q: ', predicted_new_q_value)
    # print('log p: ', log_prob)
    # print('t v: ', target_value_func[0])
    value_loss = value_criterion(predicted_value, target_value_func.detach())

    
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

# Training Policy Function
    policy_loss = (alpha * log_prob - predicted_new_q_value).mean()
    # policy_loss = (alpha * log_prob - soft_q_net1(state, new_action)).mean()  # Openai Spinning Up implementation
    # policy_loss = (alpha * log_prob - (predicted_new_q_value - predicted_value.detach())).mean() # max Advantage instead of Q to prevent the Q-value drifted high

    ## version of github/higgsfield
    # log_prob_target=predicted_new_q_value - predicted_value
    # policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
    # mean_lambda=1e-3
    # std_lambda=1e-3
    # mean_loss = mean_lambda * mean.pow(2).mean()
    # std_loss = std_lambda * log_std.pow(2).mean()
    # policy_loss += mean_loss + std_loss


    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    print('value_loss: ', value_loss.item())
    print('q loss: ', q_value_loss1.item(), q_value_loss2.item())
    print('policy loss: ', policy_loss.item() )


# Soft update the target value net
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(  # copy data value into target parameters
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )
    return predicted_new_q_value.mean()

img_dim=128
img_channel=3
num_landmarks=5
heatmap_size=16  # width and height of heatmap
transporter = Transporter(img_dim, img_channel, num_landmarks, heatmap_size)

ENV = ['Pendulum', 'Reacher'][1]
if ENV == 'Reacher':
    env=Reacher(screen_size=SCREEN_SIZE, num_joints=NUM_JOINTS, link_lengths = LINK_LENGTH, \
    ini_joint_angles=INI_JOING_ANGLES, target_pos = [369,730], render=True)  # [369,430]
    action_dim = env.num_actions
    state_dim  = env.num_observations
elif ENV == 'Pendulum':
    env = NormalizedActions(gym.make("Pendulum-v0"))
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
hidden_dim = 512
if SCREEN_SHOT:
    state_dim = num_landmarks*2  # the dim of coding of screenshot

value_net        = ValueNetwork(state_dim, hidden_dim, activation=F.relu).to(device)
target_value_net = ValueNetwork(state_dim, hidden_dim, activation=F.relu).to(device)

soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim, activation=F.relu).to(device)
soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim, activation=F.relu).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, activation=F.relu).to(device)


print('(Target) Value Network: ', value_net)
print('Soft Q Network (1,2): ', soft_q_net1)
print('Policy Network: ', policy_net)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
    

value_criterion  = nn.MSELoss()
soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()

value_lr  = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)


replay_buffer_size = int(1e6)
replay_buffer = ReplayBuffer(replay_buffer_size)


# hyper-parameters
max_episodes  = 1000
max_steps   = 20
frame_idx   = 0
batch_size  = 128
explore_steps = 0
rewards     = []
predict_qs  = []
reward_scale=10.0


transporter.load_model()

# training loop
list_s=[]
list_s_=[]
for epi in range(max_episodes):
    if ENV == 'Reacher':
        state = env.reset(SCREEN_SHOT)
    elif ENV == 'Pendulum':
        state =  env.reset()
    state = np.transpose(state[0], (2,0,1))  # (128, 128, 3) -> (3, 128, 128)
    compressed_state=transporter.compress_state(state)
    episode_reward = 0
    predict_q = 0
    print('Episode: ', epi)
    
    for step in range(max_steps):
        if frame_idx >= explore_steps:
            action = policy_net.get_action(compressed_state, deterministic=DETERMINISTIC)
        # action = policy_net.get_action(state)
        else:
            action = policy_net.sample_action()
        # print('action: ', action)
        if ENV ==  'Reacher':
            next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
        elif ENV ==  'Pendulum':
            next_state, reward, done, _ = env.step(action)
        next_state = np.transpose(next_state[0], (2,0,1))
        compressed_next_state = transporter.compress_state(next_state)

        replay_buffer.push(compressed_state, action, reward, compressed_next_state, done)
        list_s.append(state)
        list_s_.append(next_state)
        
        state = next_state
        episode_reward += reward
        frame_idx += 1

        
        if len(replay_buffer) > batch_size:
            predict_q=update(batch_size, reward_scale)
            # print('update')

        if len(list_s)%batch_size==0 and len(list_s)>0:  # train the transporter during RL training
            source_batch=torch.Tensor(list_s).to(device)
            target_batch=torch.Tensor(list_s_).to(device)
            trans_loss = transporter.update(source_batch, target_batch)

            list_s=[]
            list_s_=[]
        
        if frame_idx % 500 == 0:
            plot(frame_idx, rewards, predict_qs)
        
        if done:
            break
    if epi==0:
        rewards.append(episode_reward)
    else:
        rewards.append(rewards[-1]*0.9+0.1*episode_reward)  # running mean
    predict_qs.append(predict_q)
