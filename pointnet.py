import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import argparse

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
import pickle
import gzip



parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)

args = parser.parse_args()

save_path='./pointnet_model/'

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


class PointNet(nn.Module):
    def __init__(self, img_dim, img_channel, num_landmarks, heatmap_size, gaussian_std=2., learning_rate= 2e-4, weight_decay=5e-4):  # square image, width=height=img_dim
        super(PointNet, self).__init__()
        self.CONV_NUM_FEATURE_MAP=32
        self.CONV_KERNEL_SIZE=4
        self.CONV_STRIDE=2
        self.CONV_PADDING=1
        self.IMG_DIM=img_dim
        self.HEATMAP_SIZE=heatmap_size
        self.IMG_CHANNEL=img_channel
        self.STD = gaussian_std

        ''' generate landmarks from target images '''
        self.landmark_detector=nn.Sequential(
            nn.Conv2d(img_channel, self.CONV_NUM_FEATURE_MAP, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),  # in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.CONV_NUM_FEATURE_MAP, self.CONV_NUM_FEATURE_MAP * 2, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
            nn.BatchNorm2d(self.CONV_NUM_FEATURE_MAP * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.CONV_NUM_FEATURE_MAP*2, self.CONV_NUM_FEATURE_MAP * 4, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
            nn.BatchNorm2d(self.CONV_NUM_FEATURE_MAP * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.CONV_NUM_FEATURE_MAP*4, num_landmarks, 1, 1, 0, bias=False)
            )

        ''' extract features (same shape as heatmap) in original images '''
        self.feature_extractor=nn.Sequential(
            nn.Conv2d(img_channel, self.CONV_NUM_FEATURE_MAP, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),  # in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.CONV_NUM_FEATURE_MAP, self.CONV_NUM_FEATURE_MAP * 2, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
            nn.BatchNorm2d(self.CONV_NUM_FEATURE_MAP * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.CONV_NUM_FEATURE_MAP*2, self.CONV_NUM_FEATURE_MAP * 4, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
            nn.BatchNorm2d(self.CONV_NUM_FEATURE_MAP * 4),
            nn.LeakyReLU(0.2, inplace=True),
            )

        ''' image generation '''
        self.regressor= nn.Sequential(
            nn.ConvTranspose2d( self.CONV_NUM_FEATURE_MAP * 4 + num_landmarks, self.CONV_NUM_FEATURE_MAP * 2, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
            nn.BatchNorm2d(self.CONV_NUM_FEATURE_MAP * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( self.CONV_NUM_FEATURE_MAP * 2, self.CONV_NUM_FEATURE_MAP, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
            nn.BatchNorm2d(self.CONV_NUM_FEATURE_MAP),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.CONV_NUM_FEATURE_MAP, self.IMG_CHANNEL, self.CONV_KERNEL_SIZE, self.CONV_STRIDE, self.CONV_PADDING, bias=False),
            nn.BatchNorm2d(self.IMG_CHANNEL),
            nn.Sigmoid()
            )

        self.landmark_detector=self.landmark_detector.to(device)
        self.feature_extractor=self.feature_extractor.to(device)
        self.regressor=self.regressor.to(device)

        parameters_list=list(self.landmark_detector.parameters())+list(self.feature_extractor.parameters())+list(self.regressor.parameters())
        self.optimizer = torch.optim.Adam(parameters_list, lr=learning_rate, weight_decay=weight_decay)


        # output_size = (i-k+2p)//2+1
        h_dim2 = int(num_landmarks*(img_dim/(self.CONV_STRIDE*self.CONV_STRIDE*self.CONV_STRIDE))**2)


        # generate heatmap meshgrid
        x=np.linspace(0, self.HEATMAP_SIZE-1, self.HEATMAP_SIZE)
        y=np.linspace(0, self.HEATMAP_SIZE-1, self.HEATMAP_SIZE)
        xx,yy=np.meshgrid(x,y)
        self.xy=np.concatenate((xx[:, :,np.newaxis],yy[:, :,np.newaxis]), axis=2)

    def forward(self, x, x_):
        # get landmarks -> gaussian heatmaps from target images
        landmarks_col, landmarks_row = self.generate_landmarks(x_)
        gaussian_heatmaps=self.gaussian_heatmap(landmarks_col, landmarks_row)  # (256, 5, 16, 16)

        # extract features from source images
        target_features=self.feature_extractor(x)  # (256, 128, 16, 16)

        # generate new target images with gaussian heatmaps from target images and features from original images
        generation_input=torch.cat((target_features, gaussian_heatmaps), dim=1)  # concatenate the features and gaussian heatmaps along the channel dimension, shape: (256, 133, 16, 16)
        generated_images = self.regressor(generation_input)  # (256, 3, 128, 128)

        return generated_images

    def update(self, x, x_):
        ''' update the pointnet, including 3 models '''
        criterion = nn.MSELoss()
        generated_x_=self.forward(x, x_)
        recon_loss = criterion(generated_x_, x_.detach())

        self.optimizer.zero_grad()
        recon_loss.backward()
        self.optimizer.step()

        return recon_loss.item()

    def generate_landmarks(self, x):
        ''' generate landmarks from images '''

        heatmaps=self.generate_heatmaps(x)  # (256, 5, 16, 16)
        landmarks_col, landmarks_row=self.condense(heatmaps)  # (256, 5, 1)

        return landmarks_col, landmarks_row


    def generate_heatmaps(self, x):
        ''' step 1 '''
        x=x.view(-1, self.IMG_CHANNEL, self.IMG_DIM, self.IMG_DIM)
        x=self.landmark_detector(x)

        return x

    def condense(self, heatmaps):
        ''' step 2 
        heatmaps: (Batch, Landmarks, Width, Height)
        '''
        col=heatmaps.sum(dim=-1)
        row=heatmaps.sum(dim=-2)
        softmax=torch.nn.Softmax(dim=-1) 
        prob_col=softmax(col)
        prob_row=softmax(row)
        col_grid=torch.linspace(0, self.HEATMAP_SIZE-1, self.HEATMAP_SIZE).unsqueeze(1)
        row_grid=torch.linspace(0, self.HEATMAP_SIZE-1, self.HEATMAP_SIZE).unsqueeze(1)
        landmarks_col=torch.matmul(prob_col, col_grid)
        landmarks_row=torch.matmul(prob_row, row_grid) 

        return landmarks_col, landmarks_row


    def gaussian_heatmap(self, landmarks_col, landmarks_row):
        ''' step 3: fit a gaussian for each landmark '''
        assert landmarks_col.shape == landmarks_row.shape
        heatmap_batch=[]
        for i in range(landmarks_col.shape[0]): # per image
            heatmap_list=[]
            for j in range(landmarks_col.shape[1]): # per landmark
                mean=[landmarks_col[i][j], landmarks_row[i][j]]
                std=2*[self.STD]  # fixed standard deviation
                dis=Normal(torch.FloatTensor(mean), torch.FloatTensor(std))

                heatmap=torch.exp(dis.log_prob(torch.FloatTensor(self.xy)).sum(dim=-1))
                heatmap_list.append(heatmap)
                
            heatmap_list=torch.stack(heatmap_list)  # transfer list to torch tensor
            heatmap_batch.append(heatmap_list)

        return torch.stack(heatmap_batch)

    def save_model(self):
        torch.save(self.landmark_detector.state_dict(), save_path+'landmark_detector.pth')
        torch.save(self.feature_extractor.state_dict(), save_path+'feature_extractor.pth')
        torch.save(self.regressor.state_dict(),save_path+'regressor.pth')

    def load_model(self):
        self.landmark_detector.load_state_dict(torch.load(save_path+'landmark_detector.pth'))
        self.feature_extractor.load_state_dict(torch.load(save_path+'feature_extractor.pth'))
        self.regressor.load_state_dict(torch.load(save_path+'regressor.pth'))


def plot(x):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(x)
    plt.savefig('pointnet.png')
    # plt.show()


img_dim=128
img_channel=3
num_landmarks=5
heatmap_size=16  # width and height of heatmap
pointnet = PointNet(img_dim, img_channel, num_landmarks, heatmap_size)
total_data=1024
batch_data=256  # size of pickle batch
num_epochs=100
batch_size=256 # size of training batch

if args.train:
    f1 =gzip.open('./pointnet_data/s.gzip','rb')
    f2 =gzip.open('./pointnet_data/s_.gzip','rb')
    source_batch=[]
    target_batch=[]
    loss_list=[]
    ''' get the data '''
    for i in range(int(total_data/batch_data)):
        source_samples=pickle.load(f1) # image value 0-1, size:(128,128,3)
        target_samples=pickle.load(f2)
        for idx in range(len(source_samples)):
            source_sample=np.transpose(source_samples[idx], (2,0,1))
            target_sample=np.transpose(target_samples[idx], (2,0,1))
            source_batch.append(source_sample)
            target_batch.append(target_sample)
    source_batch=np.array(source_batch)  # (batch_size, img_channel, img_dim, img_dim )

    # y=pointnet(torch.Tensor(source_batch[:batch_size]), torch.Tensor(target_batch[:batch_size]))
    ''' train the model '''
    for epoch in range(num_epochs):
        train_loss=0.
        for i in range(int(total_data/batch_data)):
            train_source_batch=source_batch[i*batch_size: (i+1)*batch_size]
            train_target_batch=target_batch[i*batch_size: (i+1)*batch_size]
            train_source_batch=torch.Tensor(train_source_batch).to(device)
            train_target_batch=torch.Tensor(train_target_batch).to(device)
            loss = pointnet.update(train_source_batch, train_target_batch)
            train_loss+=loss

        print('Epoch: {}  | Loss: {:.4f}'.format(epoch, loss))
        
        loss_list.append(train_loss)
        if epoch%10:
            pointnet.save_model()
            plot(loss_list)
    
    pointnet.save_model()        
    f1.close()
    f2.close()

if args.test:
    pointnet.load_model()
    f1 =gzip.open('./pointnet_data/s.gzip','rb')
    f2 =gzip.open('./pointnet_data/s_.gzip','rb')
    source_samples=pickle.load(f1) # image value 0-1, size:(128,128,3)
    target_samples=pickle.load(f2)

    idx=18
    my_dpi=96  # tested at: https://www.infobyip.com/detectmonitordpi.php
    source_sample=np.transpose(source_samples[idx], (2,0,1))
    target_sample=np.transpose(target_samples[idx], (2,0,1))

    plt.figure(figsize=(128/my_dpi, 128/my_dpi), dpi=my_dpi) # plot image with exact pixels
    plt.imshow(target_samples[idx])
    plt.savefig('./pointnet_data/'+'target.png')
    plt.imshow(source_samples[idx])
    plt.savefig('./pointnet_data/'+'original.png')
    source_sample=torch.Tensor(source_sample).unsqueeze(0).to(device)
    target_sample=torch.Tensor(target_sample).unsqueeze(0).to(device)

    xs, ys=pointnet.generate_landmarks(source_sample)  # generate landmarks from source image
    xs=xs.detach().cpu().numpy()
    ys=ys.detach().cpu().numpy()
    print(xs, ys)
    plt.scatter(xs, ys, c='r', s=40)  # plot landmarks on original image
    plt.savefig('./pointnet_data/'+'landmark.png')

    generated_image=pointnet(source_sample, target_sample)  # generate image from source image to mimic the target image
    generated_image = np.transpose(generated_image.detach().cpu().numpy()[0], (1,2,0)) # (128, 128, 3)
    plt.imshow(generated_image)
    plt.savefig('./pointnet_data/'+'generated.png')
    # plt.show()

        



