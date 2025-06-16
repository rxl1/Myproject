'''
Author       :error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date         :2025-05-06 10:07:25
LastEditors  :error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime :2025-05-07 10:05:23
FilePath     :\DL-Demos-master\dldemos\ddpm\test_01.py
Description  :这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch

'''
# 检验torch库是否正常
print(torch.cuda.is_available())
print(torch.__version__)

x = torch.tensor([0.1,0.2,0.3,0.4,0.5])
y = 1-x
print(y)

x_bar = torch.empty_like(x)
print(x_bar)

z = torch.randn_like(x)
print(z)

x_reshape = x.reshape(-1,1,1,1)
print(x_reshape)
print(x_reshape.shape)
'''
from dldemos.ddpm.dataset import get_dataloader

dataloader = get_dataloader(64)


for batch in dataloader:
    data, labels = batch
    print(data.shape, labels.shape)
    print(data)
    print(labels)
    break

for i, (inputs, targets) in enumerate(dataloader):
    print(f"Batch {i}:")
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Targets shape: {targets.shape}")
    # 处理你的数据...

'''
from tqdm import tqdm
for batch in tqdm(dataloader, desc="Processing batches"):
    data, labels = batch
    # 处理数据
'''
    
