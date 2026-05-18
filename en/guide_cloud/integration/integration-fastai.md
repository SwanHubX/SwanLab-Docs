# fastai

[fastai](https://github.com/fastai/fastai) is a high-level deep learning library built on top of PyTorch, designed to make modern deep learning applications easier and more efficient. It provides a simple API that allows users to quickly build, train, and evaluate complex models without needing to delve into the underlying details.

You can use fastai to quickly train models while using SwanLab for experiment tracking and visualization.

## 1. Import SwanLabCallback

```python
from swanlab.integration.fastai import SwanLabCallback
```

**SwanLabCallback** is a logging class adapted for fastai.

**SwanLabCallback** can define parameters such as:
- `project`, `experiment_name`, `description`, and other parameters consistent with `swanlab.init`, used for initializing the SwanLab project.
- You can also create the project externally via `swanlab.init`, and the integration will log the experiment to the project you created externally.

## 2. Pass to Trainer

```python
from fastai.vision.all import *
from swanlab.integration.fastai import SwanLabCallback

...

# Define the model
learn = vision_learner(...)

# Add SwanLabCallback
learn.fit_one_cycle(5, cbs=SwanLabCallback)
```

## 3. Example - Pet Classification

```python (2,16)
from fastai.vision.all import *
from swanlab.integration.fastai import SwanLabCallback

# Load data
path = untar_data(URLs.PETS)
dls = ImageDataLoaders.from_name_re(
    path, get_image_files(path / "images"), pat=r"([^/]+)_\d+.jpg$", item_tfms=Resize(224)
)

# Define the model
learn = vision_learner(dls, resnet34, metrics=error_rate)

# Add SwanLabCallback
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