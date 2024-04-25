# MNIST手写体识别

## 概述

MNIST手写体识别是深度学习最经典的入门任务之一，由 LeCun 等人提出。  
该任务基于[MNIST数据集](https://paperswithcode.com/dataset/mnist)，研究者通过构建机器学习模型，来识别10个手写数字（0～9）。

![mnist](/assets/mnist.jpg)

本案例主要：
- 使用`pytorch`进行CNN（卷积神经网络）的构建、模型训练与评估
- 使用`swanlab`跟踪超参数、记录指标和可视化监控整个训练周期

## 环境安装

本案例基于`Python>=3.8`，请在您的计算机上安装好Python。  
环境依赖：
```
torch
torchvision
swanlab
```
快速安装命令：
```bash
pip install torch torchvision swanlab
```


## 完整代码

```python
import os
import torch
from torch import nn, optim, utils
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import swanlab

# CNN网络构建
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1 = nn.Conv2d(1, 10, 5)  # 10, 24x24
        self.conv2 = nn.Conv2d(10, 20, 3)  # 128, 10x10
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)  # 24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  # 12
        out = self.conv2(out)  # 10
        out = F.relu(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


# 捕获并可视化前20张图像
def log_images(loader, num_images=16):
    images_logged = 0
    logged_images = []
    for images, labels in loader:
        # images: batch of images, labels: batch of labels
        for i in range(images.shape[0]):
            if images_logged < num_images:
                # 使用swanlab.Image将图像转换为wandb可视化格式
                logged_images.append(swanlab.Image(images[i], caption=f"Label: {labels[i]}"))
                images_logged += 1
            else:
                break
        if images_logged >= num_images:
            break
    swanlab.log({"MNIST-Preview": logged_images})


if __name__ == "__main__":

    # 初始化swanlab
    run = swanlab.init(
        project="MNIST-example",
        experiment_name="ConvNet",
        description="Train ConvNet on MNIST dataset.",
        config={
            "model": "CNN",
            "optim": "Adam",
            "lr": 0.001,
            "batch_size": 512,
            "num_epochs": 10,
            "train_dataset_num": 55000,
            "val_dataset_num": 5000,
        },
    )

    # 设置训练机、验证集和测试集
    dataset = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
    train_dataset, val_dataset = utils.data.random_split(
        dataset, [run.config.train_dataset_num, run.config.val_dataset_num]
    )

    train_loader = utils.data.DataLoader(train_dataset, batch_size=run.config.batch_size, shuffle=True)
    val_loader = utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=run.config.lr)

    # （可选）看一下数据集的前16张图像
    log_images(train_loader, 16)

    # 开始训练
    for epoch in range(1, run.config.num_epochs):
        swanlab.log({"train/epoch": epoch})
        # 训练循环
        for iter, batch in enumerate(train_loader):
            x, y = batch
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            print(
                f"Epoch [{epoch}/{run.config.num_epochs}], Iteration [{iter + 1}/{len(train_loader)}], Loss: {loss.item()}"
            )

            if iter % 20 == 0:
                swanlab.log({"train/loss": loss.item()}, step=(epoch - 1) * len(train_loader) + iter)

        # 每4个epoch验证一次
        if epoch % 2 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    output = model(x)
                    _, predicted = torch.max(output, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

            accuracy = correct / total
            swanlab.log({"val/accuracy": accuracy})

```