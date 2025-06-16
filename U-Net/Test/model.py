import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """双卷积模块：Conv → BN → ReLU → Conv → BN → ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """UNet模型（适配CIFAR-10分类任务）"""
    def __init__(self, in_channels=3, num_classes=10, features=[64, 128, 256, 512]):
        super().__init__()
        
        self.ups = nn.ModuleList()   # 上采样模块（转置卷积 + 双卷积）
        self.downs = nn.ModuleList() # 下采样模块（双卷积 + 池化）
        self.pool = nn.MaxPool2d(2, 2)  # 下采样池化层
        
        # 构建下采样路径（通道数逐步翻倍）
        for feat in features:
            self.downs.append(DoubleConv(in_channels, feat))
            in_channels = feat
        
        # 构建上采样路径（转置卷积恢复尺寸，结合跳跃连接）
        for feat in reversed(features):
            # 转置卷积：通道数从 feat×2 → feat，尺寸翻倍
            self.ups.append(
                nn.ConvTranspose2d(feat*2, feat, kernel_size=2, stride=2)
            )
            # 双卷积处理拼接后的特征（通道数：feat + feat → feat）
            self.ups.append(DoubleConv(feat*2, feat))
        
        # 瓶颈层（通道数：features[-1] → features[-1]×2）
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # 最终分类层（1×1卷积 + 全局池化）
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []  # 保存下采样的跳跃连接特征
        
        # 下采样阶段：双卷积 → 保存特征 → 池化
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # 瓶颈层处理
        x = self.bottleneck(x)
        
        # 反转跳跃连接，按上采样顺序取用
        skip_connections = skip_connections[::-1]
        
        # 上采样阶段：转置卷积 → 拼接跳跃连接 → 双卷积
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # 转置卷积上采样
            skip_conn = skip_connections[idx // 2]  # 对应跳跃连接
            
            # 处理尺寸不匹配（如输入为奇数尺寸）
            if x.shape[2:] != skip_conn.shape[2:]:
                x = F.resize(x, size=skip_conn.shape[2:], mode='bilinear', align_corners=True)
            
            # 拼接跳跃连接（通道维度）
            concat_skip = torch.cat([skip_conn, x], dim=1)
            x = self.ups[idx+1](concat_skip)  # 双卷积处理
        
        # 分类头：1×1卷积 → 全局池化 → 展平
        x = self.final_conv(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # 压缩为 [batch, num_classes, 1, 1]
        x = x.view(x.size(0), -1)  # 展平为 [batch, num_classes]
        
        return x