import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

def download_dataset():
    dataset  = torchvision.datasets.CIFAR10(root='./U-Net/data/CIFAR10', download=True)
    # dataset  = torchvision.datasets.CIFAR10(root='./xinglin-data/CIFAR10', download=True) # 星鸾云data存放目录

    train_dataset = torchvision.datasets.CIFAR10(root='./U-Net/data/CIFAR10', train=True, download=False)
    test_dataset = torchvision.datasets.CIFAR10(root='./U-Net/data/CIFAR10', train=False, download=False)
    print('length of CIFAR10: ', len(dataset))
    print('train_dataset of CIFAR10 : ', len(train_dataset))
    print('test_dataset of CIFAR10 : ', len(test_dataset))
    print("size of picture: ", dataset[0][0].size)
    print("type of CIFAR10:", type(dataset))

    id = 0
    img, label = dataset[id]
    print(img, label)

    img_tensor = ToTensor()(img)
    print(f"img_tensor shape : {img_tensor.shape}")
    print(f"img_tensor dtype : {img_tensor.dtype}")
    print(img_tensor.min(), img_tensor.max())

def get_CIFAR10_dataloader(batch_size: int, train = True):
    # 预处理：转张量 + 归一化
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = torchvision.datasets.CIFAR10(root='./U-Net/data/CIFAR10', 
                                           train=train,
                                           transform=transform)
    # 构造数据加载器
    dataloader = DataLoader(dataset,
                            batch_size=batch_size, # 每批次多少个样本
                            shuffle=True,          # 训练时打乱数据,测试集不打乱
                            num_workers=4          # 使用4个进程并行加载数据
                            )

    return dataloader

if __name__ == '__main__':
    # import os
    # os.makedirs('work_dirs', exist_ok=True)
    download_dataset()
