
#导入torch
import torch
#创建一个维度为4的tensor
c = torch.rand(2,3,25,25)
"""
上边创建的四维tensor ，各参数的物理意义可理解为;2表示进行处理的图片batch，这里为两张；
3表示的是该图片为彩色图片，channel数量为RGB3通道，25*25表示图片的长和宽。
"""
# print(c)
# print(c.shape)
# print(c.numel())#numel函数，及number element。即逐元素相乘。
# print(c.dim())#求tensor的维度为多少

features=[64, 128, 256, 512]

for feat in reversed(features):
    print(feat)