
import torch
import torch.nn as nn
from model import UNet
from data import get_CIFAR10_dataloader
import time
from datetime import timedelta
from tqdm import tqdm

def train():
    # 设备设置：自动判断GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    
    # 模型初始化
    model = UNet(num_classes=10).to(device)
    
    # 损失函数与优化器
    loss_fn = nn.CrossEntropyLoss()  # 分类任务用交叉熵
    base_lr = 1e-3  # 初始学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10) # 学习率调度器,定期调整优化器的学习率
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                        mode = 'min',
    #                                                        patience = 5,
    #                                                        factor = 0.5) # 学习率调度器,余弦退火学习率

    # 数据加载器
    batch_size = 32
    train_loader = get_CIFAR10_dataloader(batch_size, train=True)
    test_loader = get_CIFAR10_dataloader(batch_size, train=False)
    
    # 训练参数
    total_epochs = 20  # 训练总轮数, 可根据需求调整
    save_interval = 20  # 每20轮保存一次模型
    warmup_epochs = 3  # 预热轮数, 前5轮学习率较小, CIFAR10一般设置为3
    best_test_acc = 0.0  # 记录最优测试准确率
    total_start_time = time.time()  # 记录总训练开始时间

    # 计算预热的总步数
    total_warmup_steps = warmup_epochs * len(train_loader)
    current_step = 0

    """
    # 要是需要恢复训练，可以使用下面的代码：    
    # 加载检查点
    checkpoint = torch.load('checkpoint_epoch_40.pth')  # 加载第40个epoch的检查点

    # 恢复模型和优化器状态
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 从保存的位置继续训练
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, total_epochs):
    # 继续训练...
    """ 

    # 正式开始训练
    for epoch in range(total_epochs):
        epoch_start_time = time.time()  # 记录当前轮次开始时间

#-------------------------训练-------------------------------------------------------
        # ─── 训练阶段 ───
        model.train()
        train_loss = 0.0
        train_start_time = time.time()  # 训练阶段开始时间

        # 使用 tqdm 包装 train_loader，显示训练进度条
        train_pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'Epoch {epoch+1}/{total_epochs} [训练]',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        for batch_idx, (images, labels) in train_pbar:
            batch_start_time = time.time()  # 批次开始时间

            # 学习率预热
            current_step += 1
            if current_step <= total_warmup_steps:
                # 线性预热: lr = initial_lr * (current_step / total_warmup_steps)
                warmup_lr = base_lr * (current_step / total_warmup_steps)
                # 更新优化器的学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 前向传播
            loss = loss_fn(outputs, labels)  # 计算损失
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            
            train_loss += loss.item() * images.size(0)  # 累计损失

            # 调用学习率调度器, 调整学习率
            # scheduler.step()

            # 更新进度条信息（显示当前批次的损失）
            batch_time = time.time() - batch_start_time
            train_pbar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'batch_time': f'{batch_time:.2f}s',
                'learning_rate': f'{warmup_lr:.6f}'
            })
        
        train_time = time.time() - train_start_time  # 训练阶段总耗时

        # 计算训练集平均损失
        train_loss /= len(train_loader.dataset)

#-------------------------测试-------------------------------------------------------        
        # ─── 测试阶段 ─── 
        model.eval()
        test_correct = 0
        test_total = 0
        test_start_time = time.time()  # 测试阶段开始时间
        
        # 使用 tqdm 包装 test_loader，显示测试进度条
        test_pbar = tqdm(
            test_loader,
            total=len(test_loader),
            desc=f'Epoch {epoch+1}/{total_epochs} [测试]',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        with torch.no_grad():  # 关闭梯度计算，加速推理
            for images, labels in test_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)  # 取预测类别
                test_total += labels.size(0)
                test_correct += (preds == labels).sum().item()
        
        test_time = time.time() - test_start_time  # 测试阶段总耗时
        # 计算测试集准确率
        test_acc = test_correct / test_total

#-------------------------保存检查点-------------------------------------------------
        # [可以设置为间隔n轮保存checkpoint, 也可根据测试结果决定是否保存checkpoint]
        # 每save_interval个epoch保存一次检查点
        if(epoch + 1) % save_interval == 0 or (epoch + 1) == total_epochs:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),           # 保存模型参数
                'optimizer_state_dict': optimizer.state_dict(),   # 保存优化器参数
                # 'scheduler_state_dict': scheduler.state_dict(), # 保存学习率调度器参数
                'loss': loss.item(),                              # 保存当前checkpoint损失
            }
        # 保存路径包含epoch信息
        checkpoint_path = f'./U-Net/Test/checkpoint/checkpoint_epoch_{epoch+1}.pth'
        # checkpoint_path = f'./Myproject/U-Net/Test/checkpoint/checkpoint_epoch_{epoch+1}.pth' # 星鸾云配置
        print(f"The checkpoint is being saved. Please wait for a moment...")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch : {checkpoint_path}")  # 打印保存信息
        
        # ─── 保存最优模型 ───
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), './U-Net/Test/checkpoint/Best_UNet_CIFAR10.pth')
            # torch.save(model.state_dict(), './Myproject/U-Net/Test/checkpoint/Best_UNet_CIFAR10.pth') # 星鸾云配置
        
        epoch_time = time.time() - epoch_start_time  # 当前轮次总耗时
        total_elapsed_time = time.time() - total_start_time  # 总耗时
        
         # ─── 打印本轮训练日志 ───
        print(f'\nEpoch {epoch+1:02d}/{total_epochs:02d}')
        print(f'  轮次耗时: {epoch_time:.2f}秒 (训练: {train_time:.2f}秒, 测试: {test_time:.2f}秒)')
        print(f'  总耗时: {str(timedelta(seconds=total_elapsed_time))}')
        print(f'  剩余预估: {str(timedelta(seconds=(total_epochs-epoch-1)*epoch_time))}')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  当前学习率: {warmup_lr:.6f}')
        print(f'  Test Acc:   {test_acc:.4f} (Best: {best_test_acc:.4f})\n')


if __name__ == '__main__':
    train()

    