import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from tqdm import tqdm
from torch.nn import functional as F
# 导入 make_grid 和 save_image 函数
from torchvision.utils import make_grid, save_image

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_dim=256):
        super(UNet, self).__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.ReLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        self.down1 = self._block(in_channels, 64, (3, 3), 1, 1)
        self.down2 = self._block(64, 128, (3, 3), 2, 1)
        self.down3 = self._block(128, 256, (3, 3), 2, 1)

        self.up1 = self._block(256 + time_dim, 128, (3, 3), 1, 1)
        self.up2 = self._block(128 * 2, 64, (3, 3), 1, 1)
        self.up3 = self._block(64 * 2, out_channels, (3, 3), 1, 1)

    def _block(self, in_ch, out_ch, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x, t):
        t = self.time_mlp(t)
        t = t[:, :, None, None]

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        u1 = torch.cat((t.repeat(1, 1, d3.shape[-2], d3.shape[-1]), d3), dim=1)
        u1 = self.up1(u1)
        u2 = torch.cat((u1, d2), dim=1)
        u2 = self.up2(u2)
        u3 = torch.cat((u2, d1), dim=1)
        output = self.up3(u3)

        return output

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start, beta_end, timesteps)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, betas, device):
    noise = torch.randn_like(x_0).to(device)
    sqrt_alphas_cumprod_t = get_index_from_list(betas.sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        betas.sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
           + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

@torch.no_grad()
def p_sample(model, x, t, betas, device):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        betas.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(betas.sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t / sqrt_one_minus_alphas_cumprod_t * model(x, t)
    )
    posterior_variance_t = extract(betas.posterior_variance, t, x.shape)
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample(model, image_size, batch_size, channels, timesteps, betas, device='cuda'):
    model.eval()
    x = torch.randn(batch_size, channels, image_size, image_size).to(device)
    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, x, torch.full((batch_size,), i, device=device, dtype=torch.long), betas, device)
        x = img
    model.train()
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x

if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 50
    LR = 1e-3
    IMAGE_SIZE = 32
    CHANNELS_IMG = 3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    T = 300
    BETA_SCHEDULE = 'linear'

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    if BETA_SCHEDULE == 'cosine':
        betas = cosine_beta_schedule(T)
    elif BETA_SCHEDULE == 'linear':
        betas = linear_beta_schedule(T)

    betas = torch.tensor(betas, dtype=torch.float32)
    betas = betas.to(DEVICE)

    betas.sqrt_alphas_cumprod = torch.cumprod(1. - betas, dim=0)
    betas.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - betas)

    betas.alpha_bar = betas.sqrt_alphas_cumprod ** 2
    betas.beta_tilde = betas.betas * (1 - betas.sqrt_alphas_cumprod_prev) / (1 - betas.alpha_bar)
    betas.sqrt_beta_tilde = torch.sqrt(betas.beta_tilde)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for step, (images, _) in enumerate(dataloader):
            images = images.to(DEVICE)
            t = torch.randint(0, T, (BATCH_SIZE,), device=DEVICE).long()

            noisy_images, noise = forward_diffusion_sample(images, t, betas, DEVICE)
            predicted_noise = model(noisy_images, t)

            loss = F.mse_loss(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Step {step}: Loss: {loss.item()}")

        # Save the model after each epoch
        torch.save(model.state_dict(), f"./models/ddpm_unet_cifar10_epoch_{epoch}.pth")
        print("Model saved!")

    # Sample some images from the trained model
    sampled_images = sample(model, IMAGE_SIZE, BATCH_SIZE, CHANNELS_IMG, DEVICE)
    # 创建保存图像的目录
    os.makedirs("./samples", exist_ok=True)
    grid = make_grid(sampled_images, nrow=int(BATCH_SIZE**0.5))
    save_image(grid, "./samples/sample.png")
    print("Sample images generated and saved!")



