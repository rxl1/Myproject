
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
    
class UNet_Encoder(nn.Module):
    """U-Net的下采样部分（编码器）"""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        # 初始卷积层
        self.initial_conv = ConvBlock(in_channels, base_channels)

        # 下采样路径
        self.down1 = DownSample(base_channels, base_channels * 2)  # 64 -> 128
        self.down2 = DownSample(base_channels * 2, base_channels * 4)  # 128 -> 256
        self.down3 = DownSample(base_channels * 4, base_channels * 8)  # 256 -> 512

        # 瓶颈层
        self.bottleneck = DownSample(base_channels * 8, base_channels * 16)  # 512 -> 1024

    def forward(self, x):
        # 保存每个下采样步骤的特征图，用于跳跃连接
        x1 = self.initial_conv(x)  # 原始尺寸
        x2 = self.down1(x1)        # 尺寸减半
        x3 = self.down2(x2)        # 尺寸再次减半
        x4 = self.down3(x3)        # 尺寸又减半
        x5 = self.bottleneck(x4)   # 瓶颈层，尺寸最小

         # 返回所有中间特征图
        return x5, [x1, x2, x3, x4]

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

class UNet_Decoder(nn.Module):
    """U-Net的上采样部分（解码器）"""
    def __init__(self, base_channels=64, num_classeses=1, use_transpose=True):
        super().__init__()

        # 上采样路径
        self.up1 = UpSample(base_channels * 16, base_channels * 8, base_channels * 8, use_transpose)
        self.up2 = UpSample(base_channels * 8, base_channels * 4, base_channels * 4, use_transpose)
        self.up3 = UpSample(base_channels * 4, base_channels * 2, base_channels * 2, use_transpose)
        self.up4 = UpSample(base_channels * 2, base_channels, base_channels, use_transpose)

        # 最终卷积层，生成预测掩码
        self.output_conv = nn.Conv2d(base_channels, num_classeses, kernel_size=1)

        def forward(self, x, skip_features):
            # 从下到上进行上采样，同时使用跳跃连接
            x = self.up1(x, skip_features[3])  # 使用down3的特征
            x = self.up2(x, skip_features[2])  # 使用down2的特征
            x = self.up3(x, skip_features[1])  # 使用down1的特征
            x = self.up4(x, skip_features[0])  # 使用initial_conv的特征

            # 最终输出
            return self.out_conv(x)




# 构建UNet模型
class UNet(nn.Module):
    # 创建一个3通道输入的U-Net编码器
    encoder = UNet_Encoder(in_channels=3, base_channels=64)
    
    # 测试输入（批次大小=1，通道=3，高度=256，宽度=256）
    x = torch.randn(1, 3, 256, 256)
    
    # 前向传播
    bottleneck, skip_features = encoder(x)
    
    # 打印各层输出形状
    print(f"输入形状: {x.shape}")
    print(f"瓶颈层形状: {bottleneck.shape}")
    for i, feature in enumerate(skip_features):
        print(f"跳跃连接 {i+1} 形状: {feature.shape}")