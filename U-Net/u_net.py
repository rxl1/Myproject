'''
:Author       :renl 1113272056@qq.com
:Date         :2025-06-05 21:50:22
:reference    :https://github.com/qiaofengsheng/pytorch-UNet/tree/master
               bilibili: BV11341127iK
'''
from torch import nn
import torch
from torch.nn import functional as F

class ConvBlock(nn.Module):
    """连续两个卷积+ReLU的基本模块"""
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3,1,1, padding_mode = 'reflect', bias = False),
            nn.BatchNorm2d(out_channel), # 归一化后数据分布更利于激活函数
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(out_channel, out_channel, 3,1,1, padding_mode = 'reflect', bias = False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        return self.layer(x)

class DownSample(nn.Module):
    """下采样模块：MaxPool + 双卷积"""
    def __init__(self,in_channel, out_channel):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channel, out_channel)
            # nn.Conv2d(channel, channel, 3,2,1, padding_mode='reflect', bias=False),
            # nn.BatchNorm2d(channel), # 可加 可不加
            # nn.LeakyReLU()
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# 上采样通过转置卷积(卷积后出现很多空洞,对图像分割影响较大) 或 插值法(最邻近插值法)逐步增加特征图尺寸
class UpSample(nn.Module):
    """上采样模块：转置卷积 + 双卷积 + 跳跃连接"""
    def __init__(self, in_channel, skip_channel, out_channel, use_transpose=True):
        super().__init__()

        if use_transpose:
            # 方法1: 使用转置卷积进行上采样
            self.up = nn.ConvTranspose2d(in_channel, in_channel//2, 2, 2)
        else:
            # 方法2: 使用双线性插值 + 1x1卷积进行上采样
            self.up = nn.Sequential(  # 推荐使用 nn.Upsample + 1x1卷积 组合
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 上采样扩大尺寸
                nn.Conv2d(in_channel, in_channel//2, 1) # 减少通道数，保持与跳跃连接的通道匹配
            )

         # 拼接跳跃连接后的双卷积
        self.conv = ConvBlock(in_channel // 2 + skip_channel, out_channel)


    def forward(self, x, skip_features):
        # 上采样
        x = self.up(x)

        # 确保跳跃连接的特征尺寸与当前特征尺寸匹配（处理奇数尺寸）
        if skip_features.shape[2:] != x.shape[2:]:
            skip_features = F.interpolate(skip_features, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # 拼接特征
        x = torch.cat([x, skip_features], dim=1)

        # 双卷积处理拼接后的特征
        return self.conv(x)

# 构建UNet模型
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # 特征通道数设置
        features = [64, 128, 256, 512]

        # 下采样部分
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 输入卷积块
        self.input_conv = ConvBlock(in_channels, features[0])

        # 下采样卷积块
        for feature in features[1:]:
            self.downs.append(ConvBlock(feature//2, feature))

        # 瓶颈层
        self.bottleneck = ConvBlock(features[-1], features[-1]*2)

        # 上采样部分
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(UpSample(feature*2))
            self.ups.append(ConvBlock(feature*2, feature))

        # 输出卷积
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, t=None):
        # 保存跳跃连接
        skip_connections = []

        x = self.input_conv(x)
        skip_connections.append(x)
        x = self.pool(x)

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[-(idx//2 + 1)]

            # 处理尺寸不匹配（如果有）
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='nearest')

            x = torch.cat([x, skip_connection], dim=1)
            x = self.ups[idx+1](x)

        return self.output_conv(x)