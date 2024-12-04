# PyTorch Lightning

[![](/assets/colab.svg)](https://colab.research.google.com/drive/1g1s86qobSvIuaFVxzDgzyZ-B4VzdTCym?usp=sharing)

[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning)是一个开源的机器学习库，它建立在 PyTorch 之上，旨在帮助研究人员和开发者更加方便地进行深度学习模型的研发。Lightning 的设计理念是将模型训练中的繁琐代码（如设备管理、分布式训练等）与研究代码（模型架构、数据处理等）分离，从而使研究人员可以专注于研究本身，而不是底层的工程细节。

![pytorch-lightning-image](/assets/ig-pytorch-lightning.png)

你可以使用PyTorch Lightning快速进行模型训练，同时使用SwanLab进行实验跟踪与可视化。

## 1. 引入SwanLabLogger

```python
from swanlab.integration.pytorch_lightning import SwanLabLogger
```

**SwanLabLogger**是适配于PyTorch Lightning的日志记录类。

**SwanLabLogger**可以定义的参数有：

- project、experiment_name、description 等与 swanlab.init 效果一致的参数, 用于SwanLab项目的初始化。
- 你也可以在外部通过`swanlab.init`创建项目，集成会将实验记录到你在外部创建的项目中。

## 2. 传入Trainer

```python (6,11)
import pytorch_lightning as pl

...

# 实例化SwanLabLogger
swanlab_logger = SwanLabLogger(project="lightning-visualization")

trainer = pl.Trainer(
    ...
    # 传入callbacks参数
    logger=swanlab_logger,
)

trainer.fit(...)
```

## 3. 完整案例代码

```python (1,65,70)
from swanlab.integration.pytorch_lightning import SwanLabLogger

import importlib.util
import os

import pytorch_lightning as pl
from torch import nn, optim, utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # test_step defines the test loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)


# setup data
dataset = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
train_dataset, val_dataset = utils.data.random_split(dataset, [55000, 5000])
test_dataset = MNIST(os.getcwd(), train=False, download=True, transform=ToTensor())

train_loader = utils.data.DataLoader(train_dataset)
val_loader = utils.data.DataLoader(val_dataset)
test_loader = utils.data.DataLoader(test_dataset)

swanlab_logger = SwanLabLogger(
    project="swanlab_example",
    experiment_name="example_experiment",
)

trainer = pl.Trainer(limit_train_batches=100, max_epochs=5, logger=swanlab_logger)


trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(dataloaders=test_loader)

```