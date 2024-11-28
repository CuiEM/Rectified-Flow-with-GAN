import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import sys
import yaml 
import geoopt
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
sys.path.append("./basic")
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from basic.utils import save_checkpoint, gradient_penalty_time, load_dataset
from basic.critic import Discriminator_Noise, initialize_weights
from basic.rectified_flow import RectifiedFlow
from rectified_flow_pytorch import Unet
from eval.fid import Calculate_FID
from accelerate import Accelerator

# 读取配置文件
with open('configs/config.yml', 'r') as file:
    config = yaml.safe_load(file)

# 使用配置参数
BATCH_SIZE = config['batch_size']
IMAGE_SIZE = config['image_size']
CHANNELS_IMG = config['channels_img']
NUM_EPOCHS = config['num_epochs']
FEATURES_CRITIC = config['features_critic']
FEATURES_GEN = config['features_gen']
CRITIC_ITERATIONS = config['critic_iterations']
LAMBDA_GP = config['lambda_gp']
WEIGHT_CLIP = config['weight_clip']
LOAD_RF_PRETRAIN = config['load_rf_pretrain']
LOAD_CRITIC_PRETRAIN = config['load_critic_pretrain']
eps = 0.1
BASE_CHANNELS = config['base_channels']
STEPS = config['steps']
using_accelerator = config['using_accelerator']
device = config['device']

# 使用Accelerator
if using_accelerator:
    accelerator = Accelerator()
    device = accelerator.device
else:
    device = device

loader = load_dataset(dataset_name='CELEBA', image_size=IMAGE_SIZE, channels_img=CHANNELS_IMG, batch_size=BATCH_SIZE)

model = Unet(BASE_CHANNELS, channels=CHANNELS_IMG).to(device)
rf = RectifiedFlow()
critic = Discriminator_Noise().to(device)

CHECKPOINT = torch.load("checkpoints/CELEBA_CHECKPOINT_TIME.pth.tar")

# initializate optimizer
opt_gen = optim.Adam(model.parameters(), lr=1e-4)
opt_critic = optim.SGD(critic.parameters(), lr=1e-4)

# 学习率衰减
scheduler = StepLR(opt_gen, step_size=25, gamma=0.1)
# scheduler_critic = StepLR(opt_critic, step_size=20, gamma=0.1)

if using_accelerator:
    model, critic, opt_gen, opt_critic, scheduler = accelerator.prepare(
        model, 
        critic, 
        opt_gen, 
        opt_critic, 
        scheduler
        )

# 加载RF预训练模型与否
if LOAD_RF_PRETRAIN:
    print("=> Loading checkpoint")
    model.load_state_dict(CHECKPOINT['model'])

# 加载critic预训练模型与否
if LOAD_CRITIC_PRETRAIN:
    print("=> Loading checkpoint")
    critic.load_state_dict(CHECKPOINT['critic'])
else:
    initialize_weights(critic)
    
# for tensorboard plotting
writer_real = SummaryWriter(f"logs/CELEBA/real")
writer_fake = SummaryWriter(f"logs/CELEBA/fake")

step = 0

model.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    # 不需要标签，无监督
    for batch_idx, (real, _) in enumerate(tqdm(loader)):
        real = real.to(device)

        # rf自训练
        t_0 = torch.rand(real.size(0)).to(device)
        pre_t = t_0*(1-eps)
        t = pre_t + eps

        x_0 = torch.randn_like(real).to(device)
        x_t_real = rf.create_flow(real, x_0, t).to(device)
        x_t_pre_real = rf.create_flow(real, x_0, pre_t).to(device)
        pre_t = pre_t.squeeze()
        v_pre = model(x_t_pre_real, pre_t)

        x_t_fake = x_t_pre_real + v_pre * eps
        loss_rf = rf.LOSS(v_pre, real, x_0)

        # 训练critic: max E[critic(real)] - E[critic(fake)]
        for _ in range(CRITIC_ITERATIONS):
            # t_expanded = t.expand(x_t_real.size(0)).to(device)
            critic_real = critic(x_t_real, t).reshape(-1)
            critic_fake = critic(x_t_fake.detach(), t).reshape(-1)
            # gp = gradient_penalty_time(critic, x_t_real, x_t_fake, t_expanded, device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))

            opt_critic.zero_grad()
            if using_accelerator:
                accelerator.backward(loss_critic, retain_graph=True)
            else:
                loss_critic.backward(retain_graph=True)
            opt_critic.step()

        loss_gen_critic = -torch.mean(critic(x_t_fake, t).reshape(-1))
        # loss_gen_critic = torch.mean((critic(v_pred).reshape(-1) - label_real) ** 2)
        loss_gen = loss_gen_critic + 4 * loss_rf

        opt_gen.zero_grad()
        if using_accelerator:
            accelerator.backward(loss_gen)
        else:
            loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally; save checkpoint; print to tensorboard
        if batch_idx % 100 == 0 or batch_idx == 0:

            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} ---> Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            # save checkpoint
            checkpoint = {
                "model": model.state_dict(),
                "opt_gen": opt_gen.state_dict(),
                "critic": critic.state_dict(),
                "opt_critic": opt_critic.state_dict(),
            }
            save_checkpoint(checkpoint, filename="checkpoints/CELEBA_CHECKPOINT_TIME.pth.tar")

            with torch.no_grad():
                fake = torch.randn_like(real).to(device)
                fake = rf.generator(model, fake, STEPS, device)

            # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:25], nrow=5, normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:25], nrow=5, normalize=True)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1

    scheduler.step()

Calculate_FID(expert = True, inferfake = True, data_type = 'CELEBA', num_samples = 10000)