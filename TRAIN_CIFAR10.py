import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys
import yaml
import torch.nn.functional as F
sys.path.append("./basic")

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from basic.utils import save_checkpoint, load_checkpoint, gradient_penalty
from basic.critic import Discriminator64, initialize_weights
from basic.miniunet import MiniUnet
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
eps = 1e-3

# rf params
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

CHECKPOINT = torch.load("checkpoints/CIFAR10_CHECKPOINT_NO_TIME.pth.tar")

# initializate optimizer
opt_gen = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = StepLR(opt_gen, step_size=20, gamma=0.1)
opt_critic = optim.SGD(critic.parameters(), lr=1e-4)
# scheduler_critic = StepLR(opt_critic, step_size=20, gamma=0.1)

# 加载RF预训练模型与否
if LOAD_RF_PRETRAIN:
    print("=> Loading checkpoint")
    model.load_state_dict(CHECKPOINT['model'])

# 加载critic预训练模型与否
if LOAD_CRITIC_PRETRAIN:
    initialize_weights(critic)
else:
    print("=> Loading checkpoint")
    critic.load_state_dict(CHECKPOINT['critic'])


# for tensorboard plotting
writer_real = SummaryWriter(f"logs/CIFAR10/real")
writer_fake = SummaryWriter(f"logs/CIFAR10/fake")

step = 0

model.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    # 不需要标签，无监督
    for batch_idx, (real, _) in enumerate(tqdm(loader)):
        real = real.to(device)

        # rf自训练
        t = torch.rand(real.size(0)).to(device)
        x_0 = torch.randn_like(real).to(device)
        x_t_real = rf.create_flow(real, x_0, t)
        x_t_real = x_t_real.to(device)
        x_0 = x_0.to(device)
     
        v_pred = model(x_t_real, t)
        loss_rf = rf.LOSS(v_pred, real, x_0)
        t = t[:, None, None, None]
        fake = x_t_real + v_pred * (1 - t)

        # 训练critic: max E[critic(real)] - E[critic(fake)]
        # 等价于最小化其负值
        for _ in range(CRITIC_ITERATIONS):
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake.detach()).reshape(-1)
            # gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        loss_gen_critic = -torch.mean(critic(fake).reshape(-1))
        loss_gen = loss_rf + 7 *loss_gen_critic
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
                "critic": critic.state_dict(),
                "opt_gen": opt_gen.state_dict(),
                "opt_critic": opt_critic.state_dict()
            }
            save_checkpoint(checkpoint, filename="checkpoints/CIFAR10_CHECKPOINT_NO_TIME.pth.tar")

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