# Tensorboard

[TensorBoard](https://github.com/tensorflow/tensorboard) is a visualization tool provided by Google TensorFlow, designed to help understand, debug, and optimize machine learning models. It displays various metrics and data during the training process through a graphical interface, allowing developers to intuitively understand the performance and behavior of their models.

![TensorBoard](/assets/ig-tensorboard.png)

**You can synchronize projects tracked with Tensorboard to SwanLab in two ways:**

- **Synchronous Tracking**: If your current project uses Tensorboard for experiment tracking, you can use the `swanlab.sync_tensorboardX()` or `swanlab.sync_tensorboard_torch()` commands to synchronize metrics to SwanLab while running your training script.
- **Convert Existing Projects**: If you want to copy a project from Tensorboard to SwanLab, you can use `swanlab convert` to convert a directory containing TFevent files into a SwanLab project.

::: info
The current version only supports converting scalar and image charts.
:::

[[toc]]

## 1. Synchronous Tracking

### 1.1 TensorboardX: Add the `sync_tensorboardX` Command

If you are using TensorboardX, you can add the `swanlab.sync_tensorboardX()` command anywhere before executing `tensorboardX.SummaryWriter()` to synchronize metrics to SwanLab during training.

```python
import swanlab
from tensorboardX import SummaryWriter

swanlab.sync_tensorboardX()

writer = SummaryWriter(log_dir='./runs')
```

### 1.2 PyTorch: Add the `sync_tensorboard_torch` Command

If you are using PyTorch's built-in Tensorboard, you can add the `swanlab.sync_tensorboard_torch()` command anywhere before executing `torch.utils.tensorboard.SummaryWriter()` to synchronize metrics to SwanLab during training.

```python
import swanlab
import torch

swanlab.sync_tensorboard_torch()

writer = torch.utils.tensorboard.SummaryWriter(log_dir='./runs')
```

### 1.3 Alternative Approach

You can also manually initialize SwanLab first and then run the Tensorboard code.

::: code-group

```python [TensorboardX]
import swanlab
from tensorboardX import SummaryWriter

swanlab.init(...)
swanlab.sync_tensorboardX()

...

writer = SummaryWriter(log_dir='./runs')
```

```python [PyTorch]
import swanlab
from torch.utils.tensorboard import SummaryWriter

swanlab.init(...)
swanlab.sync_tensorboard_torch()

...

writer = SummaryWriter(log_dir='./runs')
```
:::

### 1.4 Test Code

::: code-group

```python [TensorboardX]
import swanlab
from tensorboardX import SummaryWriter

swanlab.sync_tensorboardX()

writer = SummaryWriter(log_dir='./runs')

epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
  acc = 1 - 2 ** -epoch - random.random() / epoch - offset
  loss = 2 ** -epoch + random.random() / epoch + offset

  writer.add_scalar("acc", acc, epoch)
  writer.add_scalar("loss", loss, epoch)
```

```python [PyTorch]
import swanlab
from torch.utils.tensorboard import SummaryWriter

swanlab.sync_tensorboard_torch()

writer = SummaryWriter(log_dir='./runs')

epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
  acc = 1 - 2 ** -epoch - random.random() / epoch - offset
  loss = 2 ** -epoch + random.random() / epoch + offset

  writer.add_scalar("acc", acc, epoch)
  writer.add_scalar("loss", loss, epoch)
```

:::

## 2. Convert Existing Projects

### 2.1 Method 1: Command Line Conversion

```bash
swanlab convert -t tensorboard --tb_logdir [TFEVENT_LOGDIR]
```

Here, `[TFEVENT_LOGDIR]` refers to the path of the log files generated when you previously recorded experiments with Tensorboard.

The SwanLab Converter will automatically detect `tfevent` files in the specified directory and its subdirectories (default depth is 3) and create a SwanLab experiment for each `tfevent` file.

### 2.2 Method 2: In-Code Conversion

```python
from swanlab.converter import TFBConverter

tfb_converter = TFBConverter(convert_dir="[TFEVENT_LOGDIR]")
tfb_converter.run()
```

This has the same effect as the command line conversion.

### 2.3 Parameter List

| Parameter | Corresponding CLI Parameter | Description | 
| ---- | ---------- | --------------------- | 
| convert_dir    | -      | Path to Tfevent files       | 
| project    | -p, --project      | SwanLab project name       |
| workspace  | -w, --workspace      | SwanLab workspace name |
| cloud    | --cloud      | Whether to use the cloud version, default is True       | 
| logdir    | -l, --logdir      | Path to save SwanLab log files       | 

Example:

```python
from swanlab.converter import TFBConverter

tfb_converter = TFBConverter(
    convert_dir="./runs",
    project="Tensorboard-Converter",
    workspace="SwanLab",
    logdir="./logs",
    )
tfb_converter.run()
```

The equivalent CLI command:
```bash
swanlab convert -t tensorboard --tb_logdir ./runs -p Tensorboard-Converter -w SwanLab -l ./logs
```

Executing the above script will create a project named `Tensorboard-Converter` in the `SwanLab` workspace, convert the `tfevent` files in the `./runs` directory into SwanLab experiments, and save the logs generated by SwanLab in the `./logs` directory.