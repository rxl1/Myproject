import os
import time
from datetime import datetime

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn

from dldemos.ddpm.dataset import get_dataloader, get_img_shape
from dldemos.ddpm.ddpm_simple import DDPM
from dldemos.ddpm.network import (build_network, convnet_big_cfg,
                                  convnet_medium_cfg, convnet_small_cfg,
                                  unet_1_cfg, unet_res_cfg)

batch_size = 512 # 每个批次的样本数
n_epochs = 100  # 训练轮数


def train(ddpm: DDPM, net, device='cuda', ckpt_path='dldemos/ddpm/model.pth'):
    print('batch size:', batch_size)
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(batch_size)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-4)

    # 开始计时
    start_time = time.time()
    for e in range(n_epochs):
        total_loss = 0

        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)

            # 生成均匀分布的随机整数 t，范围在 [0, n_steps) 之间
            t = torch.randint(0, n_steps, (current_batch_size, )).to(device)
            
            # 生成服从标准正态分布的随机噪声 eps
            eps = torch.randn_like(x).to(device)

            # 使用 DDPM 模型生成带噪声的样本 x_t
            x_t = ddpm.sample_forward(x, t, eps)

            # 使用网络预测噪声 eps_theta
            eps_theta = net(x_t, t.reshape(current_batch_size, 1))

             # 计算损失
            loss = loss_fn(eps_theta, eps)  # 计算的是一个批次中所有数据的损失
    
            # 清空之前的梯度
            optimizer.zero_grad()
    
            # 反向传播计算梯度
            loss.backward()
    
            # 更新网络参数
            optimizer.step()
            
            # 累加损失
            total_loss += loss.item() * current_batch_size
        
        # 计算平均损失
        total_loss /= len(dataloader.dataset)

        # 结束计时
        end_time = time.time()

        # 保存模型参数
        torch.save(net.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(end_time - start_time):.2f}s')
    print('Done')


def sample_imgs(ddpm,
                net,
                output_path,
                n_sample=81,
                device='cuda',
                simple_var=True):
    """
    生成图像并保存到指定路径。

    :param ddpm: 扩散模型实例
    :param net: 预测噪声的神经网络实例
    :param output_path: 保存生成图像的路径
    :param n_sample: 生成的样本数量，默认为 81
    :param device: 设备类型（CPU 或 GPU），默认为 'cuda'
    :param simple_var: 是否使用简单的方差估计，默认为 True
    """
    net = net.to(device)
    net = net.eval()
    with torch.no_grad(): # 关闭梯度计算
        shape = (n_sample, *get_img_shape())  # 1, 3, 28, 28
        imgs = ddpm.sample_backward(shape,
                                    net,
                                    device=device,
                                    simple_var=simple_var).detach().cpu()
        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)
        imgs = einops.rearrange(imgs,
                                '(b1 b2) c h w -> (b1 h) (b2 w) c',
                                b1=int(n_sample**0.5))

        imgs = imgs.numpy().astype(np.uint8)

        cv2.imwrite(output_path, imgs)


configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg
]

if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)

    n_steps = 1000
    config_id = 2
    device = 'cuda'
    model_path = 'dldemos/ddpm/model_unet_res.pth'

    config = configs[config_id]
    net = build_network(config, n_steps)
    ddpm = DDPM(device, n_steps)

    train(ddpm, net, device=device, ckpt_path=model_path)

    net.load_state_dict(torch.load(model_path))

    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print("完成时间:" + current_time)
    save_path = "work_dirs/diffusion_" + current_time + ".jpg"
    sample_imgs(ddpm, net, save_path, device=device)
