# fastai

[fastai](https://github.com/fastai/fastai) 是一个基于 PyTorch 的高层次深度学习库，旨在使现代深度学习的应用更加容易和高效。它提供了一个简单的 API，使用户能够快速构建、训练和评估复杂的模型，而无需深入了解底层细节。

你可以使用fastai快速进行模型训练，同时使用SwanLab进行实验跟踪与可视化。

## 1.引入SwanLabCallback

```python
from swanlab.integration.fastai import SwanLabCallback
```
SwanLabCallback是适配于fastai的日志记录类。  
SwanLabCallback可以定义的参数有：
- project、experiment_name、description等与`swanlab.init`效果一致的参数

## 2.传入训练器

```python
from fastai.vision.all import *
from swanlab.integration.fastai import SwanLabCallback

...

# 定义模型
learn = vision_learner(...)

# 添加SwanLabCallback
learn.fit_one_cycle(5, cbs=SwanLabCallback)
```

## 3.案例-宠物分类

```python (2,16)
from fastai.vision.all import *
from swanlab.integration.fastai import SwanLabCallback

# 加载数据
path = untar_data(URLs.PETS)
dls = ImageDataLoaders.from_name_re(
    path, get_image_files(path / "images"), pat=r"([^/]+)_\d+.jpg$", item_tfms=Resize(224)
)

# 定义模型
learn = vision_learner(dls, resnet34, metrics=error_rate)

# 添加SwanLabCallback
learn.fit_one_cycle(
    5,
    cbs=SwanLabCallback(
        project="fastai-swanlab-integration-test",
        experiment_name="super-test",
        description="Test fastai integration with swanlab",
        logdir="./logs",
    ),
)
```