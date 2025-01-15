# FashionMNIST

:::info
Image Classification, Machine Learning Introduction, Grayscale Images
:::

## Overview

FashionMNIST is a widely used image dataset for testing machine learning algorithms, particularly in the field of image recognition. It was released by Zalando and is designed to replace the traditional MNIST dataset, which primarily contains images of handwritten digits. The purpose of FashionMNIST is to provide a slightly more challenging problem while maintaining the same image size (28x28 pixels) and structure (60,000 images for the training set and 10,000 images for the test set) as the original MNIST dataset.

![fashion-mnist](/assets/example-fashionmnist.png)

FashionMNIST contains grayscale images of clothing and footwear items from 10 categories. These categories include:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

Each category has an equal number of images, making this dataset a balanced dataset. The simplicity and standardized size of these images make FashionMNIST an ideal choice for entry-level tasks in computer vision and machine learning. The dataset is widely used for education and research to test the effectiveness of various image recognition methods.

This case study primarily focuses on:

- Using `pytorch` to build, train, and evaluate a [ResNet34](https://arxiv.org/abs/1512.03385) (Residual Neural Network) model.
- Using `swanlab` to track hyperparameters, record metrics, and visualize monitoring throughout the training cycle.

## Environment Setup

This case study is based on `Python>=3.8`. Please ensure Python is installed on your computer.
Environment dependencies:
```
torch
torchvision
swanlab
```
Quick installation command:
```bash
pip install torch torchvision swanlab
```

## Complete Code

```python
import os
import torch
from torch import nn, optim, utils
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

import swanlab

# ResNet network construction
class Basicblock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.block1 = self._make_layer(block, 16, num_block[0], stride=1)
        self.block2 = self._make_layer(block, 32, num_block[1], stride=2)
        self.block3 = self._make_layer(block, 64, num_block[2], stride=2)
        # self.block4 = self._make_layer(block, 512, num_block[3], stride=2)

        self.outlayer = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_block, stride):
        layers = []
        for i in range(num_block):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.block1(x)  # [200, 64, 28, 28]
        x = self.block2(x)  # [200, 128, 14, 14]
        x = self.block3(x)  # [200, 256, 7, 7]
        # out = self.block4(out)
        x = F.avg_pool2d(x, 7)  # [200, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [200,256]
        out = self.outlayer(x)
        return out


# Capture and visualize the first 20 images
def log_images(loader, num_images=16):
    images_logged = 0
    logged_images = []
    for images, labels in loader:
        # images: batch of images, labels: batch of labels
        for i in range(images.shape[0]):
            if images_logged < num_images:
                # Use swanlab.Image to convert images to wandb visualization format
                logged_images.append(swanlab.Image(images[i], caption=f"Label: {labels[i]}", size=(128, 128)))
                images_logged += 1
            else:
                break
        if images_logged >= num_images:
            break
    swanlab.log({"Preview/MNIST": logged_images})


if __name__ == "__main__":
    # Set device
    try:
        use_mps = torch.backends.mps.is_available()
    except AttributeError:
        use_mps = False

    if torch.cuda.is_available():
        device = "cuda"
    elif use_mps:
        device = "mps"
    else:
        device = "cpu"

    # Initialize swanlab
    run = swanlab.init(
        project="FashionMNIST",
        experiment_name="Resnet34-Adam",
        config={
            "model": "Resnet34",
            "optim": "Adam",
            "lr": 0.001,
            "batch_size": 32,
            "num_epochs": 10,
            "train_dataset_num": 55000,
            "val_dataset_num": 5000,
            "device": device,
            "num_classes": 10,
        },
    )

    # Set up training, validation, and test sets
    dataset = FashionMNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
    train_dataset, val_dataset = utils.data.random_split(
        dataset, [run.config.train_dataset_num, run.config.val_dataset_num]
    )

    train_loader = utils.data.DataLoader(train_dataset, batch_size=run.config.batch_size, shuffle=True)
    val_loader = utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize model, loss function, and optimizer
    if run.config.model == "Resnet18":
        model = ResNet(Basicblock, [1, 1, 1, 1], 10)
    elif run.config.model == "Resnet34":
        model = ResNet(Basicblock, [2, 3, 5, 2], 10)
    elif run.config.model == "Resnet50":
        model = ResNet(Basicblock, [3, 4, 6, 3], 10)
    elif run.config.model == "Resnet101":
        model = ResNet(Basicblock, [3, 4, 23, 3], 10)
    elif run.config.model == "Resnet152":
        model = ResNet(Basicblock, [3, 8, 36, 3], 10)

    model.to(torch.device(device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=run.config.lr)

    # (Optional) Take a look at the first 16 images in the dataset
    log_images(train_loader, 16)

    # Start training
    for epoch in range(1, run.config.num_epochs+1):
        swanlab.log({"train/epoch": epoch}, step=epoch)
        # Training loop
        for iter, batch in enumerate(train_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if iter % 40 == 0:
                print(
                    f"Epoch [{epoch}/{run.config.num_epochs}], Iteration [{iter + 1}/{len(train_loader)}], Loss: {loss.item()}"
                )
                swanlab.log({"train/loss": loss.item()}, step=(epoch - 1) * len(train_loader) + iter)

        # Validate every 4 epochs
        if epoch % 2 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    _, predicted = torch.max(output, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

            accuracy = correct / total
            swanlab.log({"val/accuracy": accuracy}, step=epoch)
```

## Switching to Other ResNet Models

The above code supports switching to the following ResNet models:
- ResNet18
- ResNet34
- ResNet50
- ResNet101
- ResNet152

Switching is very simple; just modify the `model` parameter in the `config` to the corresponding model name. For example, to switch to ResNet50:

```python (5)
    # Initialize swanlab
    run = swanlab.init(
        ...
        config={
            "model": "Resnet50",
        ...
        },
    )
```

- How does `config` work? 👉 [Set Experiment Configuration](/guide_cloud/experiment_track/set-experiment-config)

## Demonstration of Results

![](/assets/example-fashionmnist-show.jpg)