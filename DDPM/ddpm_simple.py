'''
:Author       :error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
:Date         :2025-04-20 18:45:09
'''

import torch

class DDPM():

    def __init__(self,
                 device,
                 n_steps: int,  # 原文中的最大时间T
                 min_beta: float = 0.0001,  # β_min
                 max_beta: float = 0.02):   # β_max
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)  # 原文中β线性等分
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars


    def sample_forward(self, x, t, eps=None):    
        '''
        :param  self:
        :param  x:
        :param  t:
        :param  eps:
        :return: 
        '''
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res

    def sample_backward(self, img_shape, net, device, simple_var=True):
        '''
         desc  :
         param   self:
         param   img_shape:
         param   net:
         param   device:
         param   simple_var:
         return: 
        '''

        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var)
        return x

    def sample_backward_step(self, x_t, t, net, simple_var=True):
        '''
        description:
        param  [*] self:
        param  [*] x_t:当前时间步t的样本张量
        param  [*] t:当前的时间步
        param  [*] net:神经网络模型,用于预测噪声
        param  [*] simple_var:布尔值，决定是否使用简化的方差计算方式
        return [*] x_t 更新后的样本x_t-1
        '''

        n = x_t.shape[0]  # 获取输入张量 x_t 的批量大小n，即有多少个样本
        # 创建一个形状为(n,1)的张量t_tensor,其中每个元素都是当前时间步t.这个张量会被传递给神经网络以预测噪声ϵ
        t_tensor = torch.tensor([t] * n,
                                dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor) # 使用神经网络net预测当前时间步t下的噪声ϵ

        # 处理最后一个时间步
        if t == 0:
            noise = 0
        else:  # 没到最后一步
            if simple_var:  # 如果使用简化的方差计算方式
                var = self.betas[t]
            else:           # 如果不使用简化的方差计算方式
                var = (1 - self.alpha_bars[t - 1]) / (
                    1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t)  # 生成与 x_t 形状相同的随机噪声
            noise *= torch.sqrt(var)

        # 计算下一个时间步t-1的均值μ_t-1
        mean = (x_t - 
                (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                eps) / torch.sqrt(self.alphas[t])

        x_t = mean + noise # 下一个时间步t-1的样本x_t-1

        return x_t  # 返回更新后的样本x_t-1  
