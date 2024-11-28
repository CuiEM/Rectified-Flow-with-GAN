import torch
import sys
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")
sys.path.append("./basic")
from rectified_flow_pytorch import Unet
from rectified_flow import RectifiedFlow
import os
from accelerate import Accelerator


def infer(
        checkpoint_path,
        base_channels=16,
        img_size=64,
        img_channels=3,
        step=50,  # 采样步数（Euler方法的迭代次数） 10步效果就很好 1步效果不好
        num_imgs=5,
        save_path='./pretrain/rectified-flow/results',
        device='cuda',
        data_type='MNIST'):
    
    if data_type == 'CELEBA':
        img_size = 64
    else:
        img_size = 32

    accelerator = Accelerator()
    device = accelerator.device
    

    os.makedirs(save_path, exist_ok=True)
    model = Unet(base_channels, channels=img_channels).to(device)
    rf = RectifiedFlow()
    model, rf = accelerator.prepare(model, rf)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        
        for i in range(num_imgs):
            print(f'Generating {i}th image...')
            
            noise = torch.randn(1, img_channels, img_size, img_size).to(device)

            # fake = rf.sample(model, noise, step, method='euler', device=device)
            fake = rf.generator(model, noise, step, device)
            fake = fake.squeeze(0)
            rf.save_image(fake, save_path, i)


if __name__ == '__main__':
    # 每个条件生成10张图像
    # label一个数字出现十次

    infer(checkpoint_path='./checkpoints/CELEBA_CHECKPOINT_TIME.pth.tar',
          base_channels=64,
          img_size=64,
          img_channels=3,
          step=10,
          num_imgs=3,
          save_path='./pretrain/rectified-flow',
          device='cuda',
          data_type='CELEBA')