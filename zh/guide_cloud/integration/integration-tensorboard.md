# Tensorboard

[TensorBoard](https://github.com/tensorflow/tensorboard) 是 Google TensorFlow 提供的一个可视化工具，用于帮助理解、调试和优化机器学习模型。它通过图形界面展示训练过程中的各种指标和数据，让开发者更直观地了解模型的性能和行为。

![TensorBoard](/assets/ig-tensorboard.png)

:::warning 其他工具的同步教程

- [Wandb](/guide_cloud/integration/integration-wandb.md)
- [MLFlow](/guide_cloud/integration/integration-mlflow.md)
:::

**你可以用两种方式将使用Tensorboard跟踪的项目同步到SwanLab：**

- **同步跟踪**：如果你现在的项目使用了Tensorboard进行实验跟踪，你可以使用`swanlab.sync_tensorboardX()`或`swanlab.sync_tensorboard_torch()`命令，在运行训练脚本时同步记录指标到SwanLab。
- **转换已存在的项目**：如果你想要将Tensorboard上的项目复制到SwanLab，你可以使用`swanlab convert`，将存放TFevent文件的目录转换成SwanLab项目。

::: info
在当前版本暂仅支持转换标量和图像图表。
:::

[[toc]]

## 1. 同步跟踪

### 1.1 TensorboardX: 添加sync_tensorboardX命令

如果你使用的是TensorboardX，可以在代码执行`tensorboardX.SummaryWriter()`之前的任何位置，添加一行`swanlab.sync_tensorboardX()`命令，即可在训练时同步记录指标到SwanLab。

```python
import swanlab
from tensorboardX import SummaryWriter

swanlab.sync_tensorboardX()

writer = SummaryWriter(log_dir='./runs')
```

### 1.2 PyTorch: 添加sync_tensorboard_torch命令

如果你使用的是PyTorch自带的tensorboard，那么可以在代码执行`torch.utils.tensorboard.SummaryWriter()`之前的任何位置，添加一行`swanlab.sync_tensorboard_torch()`命令，即可在训练时同步记录指标到SwanLab。

```python
import swanlab
import torch

swanlab.sync_tensorboard_torch()

writer = torch.utils.tensorboard.SummaryWriter(log_dir='./runs')
```

### 1.3 另一种写法

你也可以先手动初始化swanlab，再运行tensorboard的代码。

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

### 1.4 测试代码

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

## 2. 转换已存在的项目

### 2.1 方式一：命令行转换

```bash
swanlab convert -t tensorboard --tb_logdir [TFEVENT_LOGDIR]
```

支持的参数如下：

- `-t`: 转换类型，可选wandb、tensorboard和mlflow。
- `-p`: SwanLab项目名。
- `-w`: SwanLab工作空间名。
- `--cloud`: (bool) 是否上传模式为"cloud"，默认为True
- `-l`: logdir路径。
- `--tb_logdir`: Tensorboard日志文件路径。

这里的`[TFEVENT_LOGDIR]`是指你先前用Tensorboard记录实验时，生成的日志文件路径。

SwanLab Converter将会自动检测文件路径及其子目录下的`tfevent`文件（默认子目录深度为3），并为每个`tfevent`文件创建一个SwanLab实验。

### 2.2 方式二：代码内转换

```python
from swanlab.converter import TFBConverter

tfb_converter = TFBConverter(convert_dir="[TFEVENT_LOGDIR]")
tfb_converter.run()
```

效果与命令行转换一致。

### 2.3 参数列表

| 参数 | 对应CLI参数       | 描述                  | 
| ---- | ---------- | --------------------- | 
| convert_dir    | -      | Tfevent文件路径       | 
| project    | -p, --project      | SwanLab项目名       |
| workspace  | -w, --workspace      | SwanLab工作空间名 |
| cloud    | --cloud      | 是否使用云端版，默认为True       | 
| logdir    | -l, --logdir      | SwanLab日志文件保存路径       | 

例子：

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

与之作用相同的CLI：
```bash
swanlab convert -t tensorboard --tb_logdir ./runs -p Tensorboard-Converter -w SwanLab -l ./logs
```

执行上面的脚本，将会在`SwanLab`空间下，创建一个名为`Tensorboard-Converter`的项目，将`./runs`目录下tfevent文件创建为一个个swanlab实验，并将swanlab运行时产生的日志保存在`./logs`目录下。


## 3. API映射表

| 功能 | Tensorboard | SwanLab | 
| ---- | ---------- | --------------------- | 
| 创建实验    |  writer = SummaryWriter(logdir="./runs")   | swanlab.init(logdir="./runs")    | 
| 记录标量指标 | writer.add_scalar(key, value, step) | swanlab.log({key, value}, step=step) |
| 记录多个标量指标 | writer.add_scalar(key1, value1, step)<br> writer.add_scalar(key2, value2, step) | swanlab.log({key1: value1, key2: value2}, step=step) |
| 记录图像指标 | writer.add_image(key, data, step) | swanlab.log({key: swanlab.Image(data), step=step}) |
| 记录文本指标 | writer.add_text(key, data, step) | swanlab.log({key: swanlab.Text(data)}, step=step) |
| 记录音频指标 | writer.add_audio(key, data, step) | swanlab.log({key: swanlab.Audio(data), step=step}) |
| 记录视频指标 | writer.add_video(key, data, step) | swanlab.log({key: swanlab.Video(data), step=step}) |
| 记录PR曲线 | writer.add_pr_curve(key, labels, predictions, step) | swanlab.log({key: swanlab.PRCurve(labels, predictions), step=step}) |
| 关闭实验 | writer.close() | swanlab.finish() |