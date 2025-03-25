# Weights & Biases

[Weights & Biases](https://github.com/wandb/wandb) (Wandb) is a platform for experiment tracking, model optimization, and collaboration in machine learning and deep learning projects. W&B provides powerful tools for recording and visualizing experimental results, helping data scientists and researchers better manage and share their work.

![wandb](/assets/ig-wandb.png)

**You can synchronize projects from Wandb to SwanLab in two ways:**

1. **Synchronized Tracking**: If your current project uses wandb for experiment tracking, you can use the `swanlab.sync_wandb()` command to synchronize metrics to SwanLab while running the training script.
2. **Convert Existing Projects**: If you want to copy a project from wandb to SwanLab, you can use `swanlab convert` to convert an existing project on Wandb to a SwanLab project.

::: info
The current version only supports converting scalar charts.
:::

[[toc]]

## 1. Synchronized Tracking

### 1.1 Add the `sync_wandb` Command

Add the `swanlab.sync_wandb()` command anywhere before `wandb.init()` in your code to synchronize wandb metrics to SwanLab during training.

```python
import swanlab

swanlab.sync_wandb()

...

wandb.init()
```

In the above code, `wandb.init()` will simultaneously initialize swanlab, with the project name, experiment name, and configuration matching the `project`, `name`, and `config` in `wandb.init()`. Therefore, you do not need to manually initialize swanlab.

:::info

**`sync_wandb` supports two parameters:**

- `mode`: The recording mode of swanlab, supports `cloud`, `local`, and `disabled`.
- `wandb_run`: If this parameter is set to **False**, the data will not be uploaded to wandb, equivalent to setting `wandb.init(mode="offline")`.

:::

### 1.2 Alternative Approach

Another approach is to manually initialize swanlab first, then run the wandb code.

```python
import swanlab

swanlab.init(...)
swanlab.sync_wandb()

...

wandb.init()
```

In this approach, the project name, experiment name, and configuration will match the `project`, `experiment_name`, and `config` in `swanlab.init()`. The `project` and `name` in the subsequent `wandb.init()` will be ignored, and the `config` will be updated in `swanlab.config`.

### 1.3 Test Code

```python
import wandb
import random
import swanlab

swanlab.sync_wandb()
# swanlab.init(project="sync_wandb")

wandb.init(
  project="test",
  config={"a": 1, "b": 2},
  name="test",
  )

epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
  acc = 1 - 2 ** -epoch - random.random() / epoch - offset
  loss = 2 ** -epoch + random.random() / epoch + offset

  wandb.log({"acc": acc, "loss": loss})
```

![alt text](/assets/ig-wandb-4.png)

## 2. Convert Existing Projects

### 2.1 Locate Your Project, Entity, and Run ID on wandb.ai

The project, entity, and run ID are required for conversion (run ID is optional).  
The location of the project and entity:
![alt text](/assets/ig-wandb-2.png)

The location of the run ID:

![alt text](/assets/ig-wandb-3.png)

### 2.2 Method 1: Command Line Conversion

First, ensure that you are logged into wandb in the current environment and have access to the target project.

Conversion command:

```bash
swanlab convert -t wandb --wb-project [WANDB_PROJECT_NAME] --wb-entity [WANDB_ENTITY]
```

Supported parameters:

- `-t`: Conversion type, options are wandb and tensorboard.
- `--wb-project`: The name of the wandb project to be converted.
- `--wb-entity`: The space name where the wandb project is located.
- `--wb-runid`: The ID of the wandb Run (a specific experiment under the project).

If `--wb-runid` is not provided, all Runs under the specified project will be converted; if provided, only the specified Run will be converted.

### 2.3 Method 2: Conversion Within Code

```python
from swanlab.converter import WandbConverter

wb_converter = WandbConverter()
# wb_runid is optional
wb_converter.run(wb_project="WANDB_PROJECT_NAME", wb_entity="WANDB_USERNAME")
```
The effect is the same as command line conversion.