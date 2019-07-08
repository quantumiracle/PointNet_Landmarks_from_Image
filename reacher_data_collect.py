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
import pickle
import gzip


f1=gzip.open('./pointnet_data/s.gzip', 'wb')
f2=gzip.open('./pointnet_data/s_.gzip', 'wb')
NUM_JOINTS=2
LINK_LENGTH=[200, 140]
INI_JOING_ANGLES=[0.1, 0.1]
# NUM_JOINTS=4
# LINK_LENGTH=[200, 140, 80, 50]
# INI_JOING_ANGLES=[0.1, 0.1, 0.1, 0.1]
SCREEN_SIZE=1024
SPARSE_REWARD=False
SCREEN_SHOT=True
action_range = 10.0

env=Reacher(screen_size=SCREEN_SIZE, num_joints=NUM_JOINTS, link_lengths = LINK_LENGTH, \
ini_joint_angles=INI_JOING_ANGLES, target_pos = [369,430], render=True)
action_dim = env.action_space.shape[0]


s=env.reset(SCREEN_SHOT)
s=s[0]
list_s=[]
list_s_=[]
total=1024
save_interval=256
for i in range(total):
    print(i)
    a=action_range*np.random.uniform(-1,1,action_dim)
    s_,_,_,_=env.step(a, SPARSE_REWARD, SCREEN_SHOT)  # s_: shape (1,200,200,3), value [0.0,1.0]
    target_pos=np.random.uniform(200,800, 2)  # random set target position
    env.set_target_position(target_pos)
    s_=s_[0]
    s=s_
    list_s.append(s)
    list_s_.append(s_)
    if (i+1)%save_interval==0 and i>0:
        pickle.dump(list_s, f1)
        pickle.dump(list_s_, f2)
        list_s=[]
        list_s_=[]

f1.close()
f2.close()


    # plt.imshow(s)
    # plt.show()
    # print(s, s.shape, np.max(s))


# load test

# f1=open('./pointnet_data/s.gzip', 'rb')
# d=pickle.load(f1)
# print(np.array(d).shape)
# plt.imshow(d[1])
# plt.show()