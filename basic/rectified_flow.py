import torch
import torch.nn.functional as F
import torchdiffeq
import torchvision.transforms as transforms
import os
from unet import Unet


class RectifiedFlow:
    
    def exists(v):
        return v is not None
    
    def euler(self, x, v, dt):
        return x + v * dt

    def create_flow(self, x_1, x_0, t):

        t = t[:, None, None, None]  # [B, 1, 1, 1]

        x_t = t * x_1 + (1 - t) * x_0

        return x_t

    # 司机
    def mse_loss(self, v, x_1, x_0):
        """ 计算RectifiedFlow的损失函数
        L = MSE(x_1 - x_0 - v(t))  匀速直线运动

        Args:
            v: 速度，维度为 [B, C, H, W]
            x_1: 原始图像，维度为 [B, C, H, W]
            x_0: 噪声图像，维度为 [B, C, H, W]
        """

        # 求loss函数，是一个MSE，最后维度是[B]

        loss = F.mse_loss(x_1 - x_0, v)
        # loss = torch.mean((x_1 - x_0 - v)**2)

        return loss
    
    def LOSS(self, v_pred, x_1, x_0):
        loss_rf = F.mse_loss(x_1 - x_0, v_pred)
        loss_2 = - F.cosine_similarity(v_pred, x_1-x_0).mean() # 余弦相似度
        loss_rf = loss_rf + loss_2
        return loss_rf
    
    @torch.no_grad()
    def sample(self, model, noise, steps, method = 'euler', device = 'cuda'):
        def ode_fn(t, x):
            t = t.view(1)
            v = model(x, t)
            return v

        times = torch.linspace(0, 1, steps, device = device)
        trajectory = torchdiffeq.odeint(ode_fn, noise, times, method = method)
        return trajectory[-1]

    def generator(self, model, fake, steps, device):
        # eps = 1e-3
        dt = 1.0 / steps
        for i in range(steps):
            t = i * dt
            t = torch.tensor([t]).to(device)
            v_pred = model(fake, t)
            fake = self.euler(fake, v_pred, dt)
        return fake
    
    def save_image(self, fake, save_path, i):
        fake = (fake + 1) / 2
        # fake = fake.clamp(0, 1)
        image = transforms.ToPILImage()(fake)
        image.save(os.path.join(save_path, f'{i}.png'))


if __name__ == '__main__':
    # 时间越大，越是接近原始图像

    rf = RectifiedFlow()

    model = Unet(64, channels=3).to('cuda')
    checkpoint_path = './checkpoints/CELEBA_CHECKPOINT_TIME.pth.tar'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    noise = torch.randn(1, 3, 64, 64).to('cuda')
    sample = rf.sample(model, noise, steps=50, method='euler', device='cuda')
    print(sample.shape)

