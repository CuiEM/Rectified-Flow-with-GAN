# Rectified Flow + GAN
![Python](https://img.shields.io/badge/-Python-05122A?style=flat&logo=python)&nbsp;
![Pytorch](https://img.shields.io/badge/-pytorch-05122A?logo=pytorch)&nbsp;
![Static Badge](https://img.shields.io/badge/-Rectified%20FLow-05122A)
![Static Badge](https://img.shields.io/badge/-WGAN-05122A)

# Ideas
My basic idea is to build a time-dependent WGAN with rectified flow. First, I replace the generator in WGAN with Rectified Flow. Then, I modify the critic in WGAN to be time-dependent so I can critic both the real and fake images in the random time steps between 0 and 1. After about months, I find that the result is always bad. In the night of 2024/11/28, I found out that GAN model actually cannot help Rectified Flow to generate better images because training rectified flow (or any flow matching) is not basde on the log-likelihood of the data distribution.
## Brief Introduction
This is a repo for Rectified Flow model building with Time-dependent WGAN. As you can see, I upload all the codes and configs here to show my work in this unrealistic idea.

## Structures
In this section, I will introduce the structures of this repo.

### Train
There are 4 training scripts here:
- 'TRAIN_CELEBA.py': Training on CelebA dataset without time-dependent WGAN.
- 'TRAIN_CELEBA_TIME.py': Training on CelebA dataset with time-dependent WGAN.
- 'TRAIN_CIFAR10.py': Training on CIFAR10 dataset without time-dependent WGAN.
- 'TRAIN_CIFAR10_TIME.py': Training on CIFAR10 dataset with time-dependent WGAN.

### Basic
In this folder, those 8 files are the basic module or functions for this project:
- 'critic.py': Critic model.
    - 'Discriminator_Time': Critic model for time-dependent WGAN.
    - 'Discriminator_Noise': Critic model for WGAN with noise on the basis of 'Discriminator_Time' (Check the last section for more details).
- 'infer.py': Inference/Generate script.
- 'rectified_flow.py': Rectified Flow model.
- 'unet.py': UNet model (But I just import Unet from ['rectified_flow_pytorch'](https://github.com/lucidrains/rectified-flow-pytorch))
- 'utils.py': Some utils functions.

### Configs
There are two config files here:
- 'config.yml': The config file for training.
- 'acc_config.yml': The config file for accelerator.(But I don't use it actually)

### Eval
There is one eval python file and one jupyter notebook file here:
- 'fid.py': FID score calculation.
- 'plot.ipynb': Plot the generated images.

# Supplement Materials
Although the idea is not working, I still learnt a lot from this project. Here is an interesting point during training WGAN: **Gradient penalty can be replaced by instance noise** according to the paper [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406) and discussion in Reddit [here](https://www.reddit.com/r/MachineLearning/comments/oif5dp/d_adding_guassian_noise_to_discriminator_layers/). By doing this, we don't need gradient penalty, thus sufficiently decreasing the training time and the number of parameters.