# PointNet and Transporter: Landmarks Generation from Images

Pytorch Implementation of [Unsupervised Learning of Object Landmarks through Conditional Image Generation](https://papers.nips.cc/paper/7657-unsupervised-learning-of-object-landmarks-through-conditional-image-generation) and [Unsupervised Learning of Object Keypoints for Perception and Control](https://arxiv.org/abs/1906.11883)
for landmarks generation from images with unsupervised learning.

## To Run:
* **PointNet** in [Unsupervised Learning of Object Landmarks through Conditional Image Generation](https://papers.nips.cc/paper/7657-unsupervised-learning-of-object-landmarks-through-conditional-image-generation):

  `python pointnet.py --train/test`

* **Transporter** in [Unsupervised Learning of Object Keypoints for Perception and Control](https://arxiv.org/abs/1906.11883):

   `python transporter.py --train/test`

* **Transporter for RL** in [Unsupervised Learning of Object Keypoints for Perception and Control](https://arxiv.org/abs/1906.11883), using soft actor-critic (SAC) instead of neural fitted Q-iteration (NFQ):

  `python transporter_sac.py` for training a reinforcement learning algorithm SAC with landmarks coordinates extracted by the Transporter.
  
  
## Results: 
On the test dateset with screenshot of Reacher environment.

* **PointNet**

Source image:
<p align="center">
<img src="https://github.com/quantumiracle/PointNet_Landmarks_from_Image/blob/master/image_pointnet/original.png" width="20%">
  
Source iamge with landmarks (red rectangles):
<p align="center">
<img src="https://github.com/quantumiracle/PointNet_Landmarks_from_Image/blob/master/image_pointnet/landmark.png" width="20%">


Target image:
<p align="center">
<img src="https://github.com/quantumiracle/PointNet_Landmarks_from_Image/blob/master/image_pointnet/target.png" width="20%">


Generated target image with landmarks (same as on source image):

<p align="center">
<img src="https://github.com/quantumiracle/PointNet_Landmarks_from_Image/blob/master/image_pointnet/generated.png" width="20%">



* **Transporter:**

Source image:
<p align="center">
<img src="https://github.com/quantumiracle/PointNet_Landmarks_from_Image/blob/master/image/original.png" width="20%">
  
Source iamge with landmarks (red rectangles):
<p align="center">
<img src="https://github.com/quantumiracle/PointNet_Landmarks_from_Image/blob/master/image/landmark.png" width="20%">


Target image:
<p align="center">
<img src="https://github.com/quantumiracle/PointNet_Landmarks_from_Image/blob/master/image/target.png" width="20%">


Generated target image with landmarks (same as on source image):

<p align="center">
<img src="https://github.com/quantumiracle/PointNet_Landmarks_from_Image/blob/master/image/generated.png" width="20%">


* **Transporter for RL**:

Some RL training curves with Transporter:

<p align="center">
<img src="https://github.com/quantumiracle/PointNet_Landmarks_from_Image/blob/master/image_transporter_rl/transporter_sac_keeptraining.png" width="80%">
  
  <p align="center">
<img src="https://github.com/quantumiracle/PointNet_Landmarks_from_Image/blob/master/image_transporter_rl/transporter_sac_notraining.png" width="80%">
  
  <p align="center">
<img src="https://github.com/quantumiracle/PointNet_Landmarks_from_Image/blob/master/image_transporter_rl/transporter_sac_notraining2.png" width="80%">

Experiments show that, with the landmarks extraced from the pre-trained Transporter as observation, RL algorithms like SAC can  learn a relatively good policy (near optimal, but shaking around it) very quickly, but not the optimal policy. However, as the landmark positions from the Transporter are not very accurately describing the structure of the agent (e.g. joints of the 'reacher'), it cannot learn a very good policy in the end no matter the Transporter is keep being trained or not during the RL learning process.
