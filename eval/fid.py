import torch
import os
import torchvision.transforms as transforms 
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append("./basic")
from pytorch_fid import fid_score
from infer import infer
from torchvision import datasets, transforms
from cleanfid import fid



# 将MNIST数据导出50张到fid/MNIST/real
def export_mnist(num_samples=500):

    # 定义数据变换
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 加载MNIST数据集
    mnist_dataset = datasets.MNIST(root='./datasets', train=True, download=True, transform=transform)
    
    # 导出指定数量的样本
    sampled_indices = torch.randperm(len(mnist_dataset))[:num_samples]
    sampled_data = [mnist_dataset[i] for i in sampled_indices]
    
    # 创建fid/real目录
    os.makedirs('./eval/fid/MNIST/real', exist_ok=True)
    
    # 保存样本到fid/real目录
    for i, sample in enumerate(sampled_data):
        sample_image = sample[0].squeeze()
        sample_image = (sample_image + 1) / 2  # 将像素值从[-1, 1]转换为[0, 1]
        sample_image = transforms.ToPILImage()(sample_image)
        sample_image.save(f'./eval/fid/MNIST/real/sample_{i}.png')

    print(f'导出 {num_samples} 张MNIST样本到 fid/MNIST/real 目录')


#将CIFAR10数据导出50张到fid/CIFAR10/real
def export_cifar10(num_samples=500):

    # 定义数据变换
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 加载CIFAR10数据集
    cifar10_dataset = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
    
    # 导出指定数量的样本
    sampled_indices = torch.randperm(len(cifar10_dataset))[:num_samples]
    sampled_data = [cifar10_dataset[i] for i in sampled_indices]
    
    # 创建fid/CIFAR10/real目录
    os.makedirs('./eval/fid/CIFAR10/real', exist_ok=True)
    
    # 保存样本到fid/CIFAR10/real目录
    for i, (image, _) in enumerate(sampled_data):
        # 将张量转换为PIL图像
        image = transforms.ToPILImage()(image * 0.5 + 0.5)
        # 保存图像        
        image.save(f'./eval/fid/CIFAR10/real/sample_{i}.png')

def export_celeba(num_samples=500):

    transform = transforms.Compose(
    [   
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize( [0.5 for _ in range(3)], [0.5 for _ in range(3)]),
    ]
    )

    dataset = datasets.CelebA(root="datasets/", transform=transform, download=False)

    sampled_indices = torch.randperm(len(dataset))[:num_samples]
    sampled_data = [dataset[i] for i in sampled_indices]

    os.makedirs('./eval/fid/CELEBA/real', exist_ok=True)

    for i, (image, _) in enumerate(sampled_data):
        image = transforms.ToPILImage()(image * 0.5 + 0.5)
        image.save(f'./eval/fid/CELEBA/real/sample_{i}.png')

# 计算FID
def Calculate_FID(expert = False, inferfake = False, data_type='CIFAR10', num_samples=500, step=100):
    CHECKPOINT_PATH = f'./checkpoints/{data_type}_CHECKPOINT_TIME.pth.tar'
    # CHECKPOINT_PRETRAIN_PATH = f'./pretrain/rectified-flow/pretrain_rf_{data_type}_checkpoint_bn.pth.tar'
    real_images_folder = f'./eval/fid/{data_type}/real'
    generated_images_folder = f'./eval/fid/{data_type}/fake'

    if inferfake is True:
        infer(
            checkpoint_path=CHECKPOINT_PATH,
            save_path=generated_images_folder,
            base_channels=64,
            step=step,
            num_imgs=num_samples,
            data_type=data_type,
            device='cuda'
        )

    if expert is True:
        if data_type == 'MNIST':
            export_mnist(num_samples=num_samples)
        elif data_type == 'CIFAR10':
            export_cifar10(num_samples=num_samples)
        elif data_type == 'CELEBA':
            export_celeba(num_samples=num_samples)

    # fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],batch_size=32, device='cuda', dims = 2048)
    fid_value = fid.compute_fid(real_images_folder, generated_images_folder)

    # fid_value = fid.compute_kid(generated_images_folder, dataset_name="cifar10", dataset_res=32, dataset_split="train")
    print('FID value:', fid_value)

NUM_SAMPLES = 10000

if __name__ == '__main__':
    Calculate_FID(
        expert = True,
        inferfake = True,
        step = 10,
        data_type = 'CELEBA',
        num_samples = NUM_SAMPLES,
    )