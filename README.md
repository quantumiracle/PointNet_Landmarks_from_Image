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
