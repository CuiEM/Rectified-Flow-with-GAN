import torch
import torch.nn as nn
from torchsummary import summary
from collections import OrderedDict

class TimeEmbedding(nn.Module):
    def __init__(self, features_d):
        super(TimeEmbedding, self).__init__()
        self.features_d = features_d

    def forward(self, t):
        t = t * 1000
        freqs = torch.pow(10000, torch.linspace(0, 1, self.features_d // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)
    
class GaussianNoise(nn.Module):
    # std=0.1 for high resolution image 
    # # Try noise just for real or just for fake images.
    def __init__(self, std=0.25, decay_rate=0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.training:
            return x + torch.empty_like(x).normal_(std=self.std)
        else:
            return x

class Discriminator_Time(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator_Time, self).__init__()
        self.features_d = features_d
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, features_d)
        )
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img+features_d, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=2, stride=1, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, t):
        if t is not None:
            t_emb = TimeEmbedding(self.features_d)(t)
            t_emb = t_emb.view(-1, self.features_d, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, t_emb], dim=1)
        return self.disc(x)

class Discriminator_Noise(nn.Module):
    def __init__(self, activation=nn.LeakyReLU, std=0.1, std_decay_rate=0):
        super().__init__()
        self.std = std
        self.std_decay_rate = std_decay_rate
        self.activation = activation
        self.stacks = nn.Sequential(*[
            self.downsample(32, bn=False),
            self.downsample(64),
            self.downsample(128),
            self.downsample(256),
            self.downsample(512),
            self.downsample(1024),
        ])

        self.head = nn.Sequential(OrderedDict([
            ('gauss', GaussianNoise(self.std, self.std_decay_rate)),
            ('linear', nn.LazyLinear(1)),
            # ('act', nn.Sigmoid()),        # removed for BCEWithLogitsLoss
        ]))

    def downsample(self, num_filters, bn=True, stride=2):
        layers = [
            GaussianNoise(self.std, self.std_decay_rate),
            nn.LazyConv2d(num_filters, kernel_size=4, stride=stride, bias=not bn, padding=1)
        ]
        if bn:
            layers.append(nn.BatchNorm2d(num_filters))
        layers.append(self.activation(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, t):
        if t is not None:   
            features_d = x.shape[2]
            t_emb = TimeEmbedding(features_d)(t)
            t_emb = t_emb.view(-1, features_d, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, t_emb], dim=1)
        x = self.stacks(x)
        x = x.flatten(1)
        x = self.head(x)
        return x

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic = Discriminator_Noise().to(device)
    t = torch.rand(N).to(device)
    x = torch.randn(N, in_channels, H, W).to(device)
    print(critic(x, t).shape)

if __name__ == "__main__":
    test()