import pygame
import numpy as np
import math
import time
from gym.spaces.box import Box
import matplotlib.pyplot as plt

class Reacher:
    def __init__(self, screen_size=1000, num_joints=2, link_lengths = [200, 140], ini_joint_angles=[0.1, 0.1], target_pos = [669,430], render=False):
        # Global variables
        self.screen_size = screen_size
        self.num_joints = num_joints # not counting the initial one
        self.link_lengths = link_lengths
        self.ini_joint_angles = ini_joint_angles
        self.joint_angles = ini_joint_angles
        self.num_actions = self.num_joints
        self.num_observations= 2*(self.num_actions+2) # initial joint +1, target position +1
        self.L = 8 # distance from target to get reward 2
        self.action_space=Box(-100,100, [self.num_actions])
        self.observation_space=Box(-1000,1000, [2*(self.num_actions+2)])
        self.target_pos=target_pos

        self.render=render
        if self.render == True:
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Reacher")
        else:
            pass

        self.is_running = 1
        self.steps=0
        self.max_episode_steps=50
        self.near_goal_range=0.5
        self.change_goal=0
        self.change_goal_episodes=10

    # Function to compute the transformation matrix between two frames
    def compute_trans_mat(self, angle, length):
        cos_theta = math.cos(math.radians(angle))
        sin_theta = math.sin(math.radians(angle))
        dx = -length * sin_theta
        dy = length * cos_theta
        T = np.array([[cos_theta, -sin_theta, dx], [sin_theta, cos_theta, dy], [0, 0, 1]])
        return T

    # Function to draw the current state of the world
    def draw_current_state(self, ):
        T = np.zeros((self.num_joints, 3, 3))  # transition matrix
        origin = np.zeros((self.num_joints, 3)) # transformed coordinates - 3 values
        p = np.zeros((self.num_joints+1, 2))  # joint coordinates in world
        p[0] = [0, 0] # initial joint coordinates

        for i in range(self.num_joints):
            T[i] = self.compute_trans_mat(self.joint_angles[i], self.link_lengths[i])
            multiplier = np.array([0, 0, 1])
            for j in range(i):
                multiplier=np.dot(T[i-j], multiplier)
            origin[i] = np.dot(T[0], multiplier)
            p[i+1] = [origin[i][0], -1.*origin[i][1]]  # the - is because the y-axis is opposite in world and image coordinates
        
        int_coordinates = [[0 for i in range(2)] for j in range(self.num_joints+1)]
        for i in range (self.num_joints+1):
            int_coordinates[i][0] = int(0.5 * self.screen_size + p[i][0])
            int_coordinates[i][1] = int(0.5 * self.screen_size + p[i][1])


        if self.render == True:
            self.screen.fill((0, 0, 0))
            for i in range (self.num_joints+1):
                if i < self.num_joints:
                    pygame.draw.line(self.screen, (255, 255, 255), [int_coordinates[i][0], int_coordinates[i][1]], [int_coordinates[i+1][0], int_coordinates[i+1][1]], 10) # draw link
                pygame.draw.circle(self.screen, (0, 255, 0), [int_coordinates[i][0], int_coordinates[i][1]], 20)  # draw joint
            pygame.draw.circle(self.screen, (255, 255, 0), np.array(self.target_pos).astype(int), 30) # draw target
            # Flip the display buffers to show the current rendering
            pygame.display.flip()
            # time.sleep(0.5)

            ''' screenshot the image '''
            # pygame.image.save(self.screen, './screen.png')
            array_screen = pygame.surfarray.array3d(self.screen) # 3d array pygame.surface (self.screen)
            red_array_screen=pygame.surfarray.pixels_red(self.screen) # 2d array from red pixel of pygame.surface (self.screen)
            downsampling_rate=8 # downsmaple the screen shot, origin 1000*1000*3
            downsampled_array_screen=array_screen[::downsampling_rate,::downsampling_rate,]
            # plt.imshow(array_screen[::downsampling_rate,::downsampling_rate,])
            # plt.show()
            
        else:
            pass
        return np.array(int_coordinates).reshape(-1), np.array([downsampled_array_screen])
    
    def reset(self, screen_shot):
        ''' reset the environment '''
        self.steps=0
        self.joint_angles = np.array(self.ini_joint_angles)*180.0/np.pi
        if self.render == True:
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Reacher")
        else:
            pass
        self.is_running = 1

        ''' reset the target position for learning across tasks '''
        # self.change_goal+=1
        # if self.change_goal > self.change_goal_episodes:
        #     self.change_goal=0
        #     range_pose=0.3
        #     target_pos=range_pose*np.random.rand(2) + [0.5,0.5]
        #     self.target_pos=target_pos*self.screen_size

        pos_set, screenshot=self.draw_current_state()
        
        if screen_shot:
            return screenshot/255.  # normalize to range 0-1
        else:
            return np.array(np.concatenate((pos_set,self.target_pos)))/self.screen_size

    def step(self, action, sparse_reward, screen_shot):    
        # Get events and check if the user has closed the window
        if self.render == True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = 0
                    break
        else:
            pass
        # Change the joint angles (the increment is in degrees)
        for i in range (self.num_joints):  
            self.joint_angles[i] += action[i]

        pos_set, screenshot=self.draw_current_state()
        distance2goal = np.sqrt((pos_set[-2]-self.target_pos[0])**2+(pos_set[-1]-self.target_pos[1])**2)
        # print(distance2goal, pos_set[-2], pos_set[-1], self.target_pos[0], self.target_pos[1])
        if sparse_reward:
            if distance2goal < self.L:
                reward = 20
            else:
                reward = -1
        
        else:  # dense reward  
            reward_0=100.0
            reward = reward_0 / (np.sqrt((pos_set[-2]-self.target_pos[0])**2+(pos_set[-1]-self.target_pos[1])**2)+1)
            # reward = -np.sqrt((pos_set[-2]-self.target_pos[0])**2+(pos_set[-1]-self.target_pos[1])**2)
        if screen_shot: 
            return screenshot/255., reward, 0, distance2goal
        else: 
            return np.array(np.concatenate((pos_set,self.target_pos)))/self.screen_size, reward, 0, distance2goal
    
    def reward(self, state):
        ''' an interface for deriving reward from state '''
        state = state*self.screen_size  # *self.screen_size is because the output in step() has divided it
        reward_0=100.0
        reward = reward_0 / (np.sqrt((state[-4]-self.target_pos[0])**2+(state[-3]-self.target_pos[1])**2)+1)
        return reward

    def set_target_position(self, pos):
        self.target_pos=pos




if __name__ == "__main__":


    num_episodes=500
    num_steps=20
    action_range=20.0
    # NUM_JOINTS=4
    # LINK_LENGTH=[200, 140, 80, 50]
    # INI_JOING_ANGLES=[0.1, 0.1, 0.1, 0.1]
    NUM_JOINTS=2
    LINK_LENGTH=[200, 140]
    INI_JOING_ANGLES=[0.1, 0.1]
    SPARSE_REWARD=False
    SCREEN_SHOT=False

    # reacher=Reacher(render=True)  # 2-joint reacher
    reacher=Reacher(screen_size=1000, num_joints=NUM_JOINTS, link_lengths = LINK_LENGTH, \
    ini_joint_angles=INI_JOING_ANGLES, target_pos = [669,430], render=True)


    epi=0
    while epi<num_episodes:
        print(epi)
        epi+=1
        step=0
        reacher.reset(SCREEN_SHOT)
        while step<num_steps:
            step+=1
            # action=np.random.uniform(-action_range,action_range,size=NUM_JOINTS)
            action=[9,0]
            state, re, _, _ =reacher.step(action, SPARSE_REWARD, SCREEN_SHOT)
            print(state, re)



    
