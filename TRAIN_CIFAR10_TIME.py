import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import geoopt
import sys
import yaml
import torch.nn.functional as F
sys.path.append("./basic")

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from basic.utils import save_checkpoint, gradient_penalty
from basic.critic_time import Discriminator64, initialize_weights
from basic.unet import Unet
from basic.rectified_flow import RectifiedFlow
from rectified_flow_pytorch import Unet


# 读取配置文件
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# 使用配置参数
device = config['device'] if torch.cuda.is_available() else "cpu"
LEARNING_RATE = config['learning_rate']
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
BASE_CHANNELS = config['base_channels']
STEPS = config['steps']

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize( [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)

dataset = datasets.CIFAR10(root="datasets/", transform=transforms, download=True)

def create_small_dataset(full_dataset, num_samples=320):
    small_dataset = torch.utils.data.Subset(full_dataset, range(num_samples))
    return small_dataset

# 使用小规模数据集
small_dataset = create_small_dataset(dataset, num_samples=320)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

model = Unet(BASE_CHANNELS, channels=CHANNELS_IMG).to(device)
rf = RectifiedFlow()
critic = Discriminator64(CHANNELS_IMG, FEATURES_CRITIC).to(device)

CHECKPOINT = torch.load("checkpoints/CIFAR10_CHECKPOINT_TIME.pth.tar")

if LOAD_RF_PRETRAIN:
    pretrain_rf_checkpoint = torch.load("pretrain/rectified-flow/pretrain_rf_CIFAR10_checkpoint_bn.pth.tar")
    print("=> Loading pretrain checkpoint")
    model.load_state_dict(pretrain_rf_checkpoint['model'])
    # opt_gen.load_state_dict(pretrain_rf_checkpoint['optimizer'])
else:
    print("=> Loading checkpoint")
    model.load_state_dict(CHECKPOINT['model'])

# 加载critic预训练模型与否
if LOAD_CRITIC_PRETRAIN:
    # pretrain_critic_checkpoint = torch.load("pretrain/wgan/gan_pretrain.pth.tar")
    # print("=> Loading pretrain checkpoint")
    # critic.load_state_dict(pretrain_critic_checkpoint['critic'])
    initialize_weights(critic)
else:
    print("=> Loading checkpoint")
    critic.load_state_dict(CHECKPOINT['critic'])
    # opt_critic.load_state_dict(CHECKPOINT['opt_critic'])


# 修改优化器配置
opt_gen = geoopt.optim.RiemannianAdam(model.parameters(), lr=1e-4)
scheduler = StepLR(opt_gen, step_size=20, gamma=0.1)
opt_critic = optim.SGD(critic.parameters(), lr=1e-4, weight_decay=0.1)

# 用于tensorboard绘图
writer_real = SummaryWriter("logs/CIFAR10/real")
writer_fake = SummaryWriter("logs/CIFAR10/fake")
step = 0


model.train()
critic.train()

# 主训练循环的修改

for epoch in range(NUM_EPOCHS):
    # 不需要标签，无监督
    for batch_idx, (real, _) in enumerate(tqdm(loader)):
        real = real.to(device)

        # rf自训练
        t = torch.rand(real.size(0)).to(device)
        x_t_real, x_0 = rf.create_flow(real, t)
        x_t_real = x_t_real.to(device)
        x_0 = x_0.to(device)

        pre_t = (t - 0.02).to(device)
        pre_t = pre_t[:, None, None, None]
        x_t_pre_real = pre_t * real + (1-pre_t) * x_0
        pre_t = pre_t.squeeze()
        v_pre = model(x_t_pre_real, pre_t)

        x_t_fake = x_t_pre_real + v_pre * 0.02
        loss_rf = rf.mse_loss(v_pre, real, x_0)
        loss_2 = - F.cosine_similarity(v_pre, real-x_0).mean() # 余弦相似度
        loss_rf = loss_rf + loss_2
        # fake = x_0.clone().to(device)
        # fake = rf.generator(model, fake, STEPS, device)

        # 训练critic: max E[critic(real)] - E[critic(fake)]
        # 等价于最小化其负值

        for _ in range(CRITIC_ITERATIONS):
            t_expanded = t.expand(x_t_real.size(0))
            critic_real = critic(x_t_real, t_expanded).reshape(-1)
            critic_fake = critic(x_t_fake.detach(), t_expanded).reshape(-1)
            gp = gradient_penalty(critic, x_t_real, x_t_fake, t_expanded, device)

            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp

            # loss_real = torch.mean((critic_real - label_real) ** 2)
            # loss_fake = torch.mean((critic_fake - label_fake) ** 2)
            # loss_critic = loss_real + loss_fake + LAMBDA_GP * gp

            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            # for p in critic.parameters():
            #     p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        # print(f"loss_real: {loss_real:.4f}, \nloss_fake: {loss_fake:.4f}, \ngp: {gp:.4f}")

        loss_gen_critic = -torch.mean(critic(x_t_fake, t_expanded).reshape(-1))
        # loss_gen_critic = torch.mean((critic(v_pred).reshape(-1) - label_real) ** 2)
        loss_gen = loss_gen_critic + 4 * loss_rf

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally; save checkpoint; print to tensorboard
        if batch_idx % 100 == 0 or batch_idx == 0:

            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            # save checkpoint
            checkpoint = {
                "model": model.state_dict(),
                "opt_gen": opt_gen.state_dict(),
                "critic": critic.state_dict(),
                "opt_critic": opt_critic.state_dict(),
            }
            save_checkpoint(checkpoint, filename="checkpoints/CIFAR10_CHECKPOINT_TIME.pth.tar")

            with torch.no_grad():
                fake = torch.randn_like(real).to(device)
                fake = rf.generator(model, x_0, STEPS, device)

            # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:25], nrow=5, normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:25], nrow=5, normalize=True)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1