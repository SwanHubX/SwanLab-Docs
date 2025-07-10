# PyTorch Lightning

[![](/assets/colab.svg)](https://colab.research.google.com/drive/1g1s86qobSvIuaFVxzDgzyZ-B4VzdTCym?usp=sharing)

[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) is an open-source machine learning library built on top of PyTorch, designed to help researchers and developers more conveniently develop deep learning models. The design philosophy of Lightning is to separate the tedious code in model training (such as device management, distributed training, etc.) from the research code (model architecture, data processing, etc.), so that researchers can focus on the research itself rather than the underlying engineering details.

![pytorch-lightning-image](/assets/ig-pytorch-lightning.png)

You can use PyTorch Lightning to quickly train models while using SwanLab for experiment tracking and visualization.

## 1. Import SwanLabLogger

```python
from swanlab.integration.pytorch_lightning import SwanLabLogger
```

**SwanLabLogger** is a logging class adapted for PyTorch Lightning.

**SwanLabLogger** can define parameters such as:

- `project`, `experiment_name`, `description`, and other parameters consistent with `swanlab.init`, used for initializing the SwanLab project.
- You can also create the project externally via `swanlab.init`, and the integration will log the experiment to the project you created externally.

## 2. Pass to Trainer

```python (6,11)
import pytorch_lightning as pl

...

# Instantiate SwanLabLogger
swanlab_logger = SwanLabLogger(project="lightning-visualization")

trainer = pl.Trainer(
    ...
    # Pass the logger parameter
    logger=swanlab_logger,
)

trainer.fit(...)
```

## 3. Complete Example Code

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

## 4. Note: If you call `trainer.fit` multiple times

If you call `trainer.fit` multiple times in a single process (e.g., N-fold cross-validation), you need to add the following line after `trainer.fit`:

```python
swanlab_logger.experiment.finish()
# or swanlab.finish()
```

Example code:

```python
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from swanlab.integration.pytorch_lightning import SwanLabLogger
import datetime
import argparse

class RandomDataset(Dataset):
    def __init__(self, size=100):
        self.x = torch.randn(size, 10)
        self.y = (self.x.sum(dim=1) > 0).long()  # Simple classification task

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SimpleClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(10, 2)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def main(args):
    dataset = RandomDataset(size=100)
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold + 1}/3")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=16)

        # Log name
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f"{args.save_name}_fold{fold + 1}_{current_time}"

        swanlab_logger = SwanLabLogger(
            project="swanlab_example",
            experiment_name=run_name,
        )

        model = SimpleClassifier()

        trainer = pl.Trainer(
            max_epochs=5,
            logger=swanlab_logger,
            log_every_n_steps=1
        )

        trainer.fit(model, train_loader, val_loader)
        swanlab_logger.experiment.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, default="test_swan")
    args = parser.parse_args()
    main(args)
```