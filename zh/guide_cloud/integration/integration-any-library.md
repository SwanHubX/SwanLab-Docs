# 将SwanLab集成到你的库

本指南提供了如何将SwanLab集成到您的Python库中的最佳实践，以获得强大的实验跟踪、GPU和系统监控、超参数记录等功能。

下面我们将介绍，如果您正在处理的代码库比单个 Python 训练脚本或 Jupyter Notebook 更复杂时，我们整理的最佳实践。

**🪵目录：**

[[toc]]

## 1. 补充Requirements

在开始之前，请决定是否在您的库的依赖项中要求 SwanLab：

### 1.1 将swanlab作为依赖项

```plaintext
torch==2.5.0
...
swanlab==0.4.*
```

### 1.2 将swanlab作为可选安装

有两种设置swanlab成为可选安装的方法。

1. 在代码中使用try-except语句，当用户没有安装swanlab时，抛出错误。

```python
try:
    import swanlab
except ImportError:
    raise ImportError(
        "You are trying to use swanlab which is not currently installed."
        "Please install it using pip install swanlab"
    )
```

2. 如果你要构建Python包，请将`swanlab`作为可选依赖项添加到`pyproject.toml`文件中：

```toml
[project]
name = "my_awesome_lib"
version = "0.1.0"
dependencies = [
    "torch",
    "transformers"
]

[project.optional-dependencies]
dev = [
    "swanlab"
]
```

## 2. 用户登录

您的用户有几种方法可以登录SwanLab：

::: code-group

```bash [命令行]
swanlab login
```

```python [Python]
import swanlab
swanlab.login()
```

```bash [环境变量(Bash)]
export SWANLAB_API_KEY=$YOUR_API_KEY
```

```python [环境变量(Python)]
import os
os.environ["SWANLAB_API_KEY"] = "zxcv1234..."
```

:::

如果用户是第一次使用`swanlab`而没有遵循上述任何步骤，则当您的脚本调用`swanlab.init`时，系统会自动提示他们登录。

## 3. 启动SwanLab实验

实验是SwanLab的计算单元。通常，你可以为每个实验创建一个`Experiment`对象，并使用`swanlab.init`方法启动实验。

### 3.1 初始化实验

初始化SwanLab，并在您的代码种启动实验：

```python
swanlab.init()
```

你可以为这个实验提供项目名、实验名、工作空间等参数：

```python
swanlab.init(
    project="my_project",
    experiment_name="my_experiment",
    workspace="my_workspace",
    )
```

::: warning 最好把 swanlab.init 放在哪里？

您的库应该尽早创建SwanLab实验，因为SwanLab会自动收集控制台中的任何输出，这将使得调试更加容易。

:::

### 3.2 配置三种启动模式

你可以通过`mode`参数来配置SwanLab的启动模式：

::: code-group

```python [云端模式]
swanlab.init(
    mode="cloud",  # 默认模式
    )
```

```python [本地模式]
swanlab.init(
    mode="local",
    )
```

```python [禁用模式]
swanlab.init(
    mode="disabled",
    )
```

:::

- **云端模式**：默认模式。SwanLab会将实验数据上传到一个web服务器（SwanLab官方云或您自行部署的私有云）。
- **本地模式**：SwanLab不会将实验数据上传到云端，但会记录一个特殊的`swanlog`目录，可以被`dashboard`插件打开进行可视化。
- **禁用模式**：SwanLab不会收集任何数据，代码执行到`swanlab`相关代码时将不做任何处理。

### 3.3 定义实验超参数/配置

使用swanlab实验配置(config)，您可以在创建SwanLab实验时提供有关您的模型、数据集等的元数据。您可以使用这些信息来比较不同的实验并快速了解主要差异。

您可以记录的典型配置参数包括：

- 模型名称、版本、架构参数等
- 数据集名称、版本、训练/测试数据数等。
- 训练参数，例如学习率、批量大小、优化器等。

以下代码片段显示了如何记录配置：

```python
config = {"learning_rate": 0.001, ...}
swanlab.init(..., config=config)
```

**更新配置**：

使用`swanlab.config.update`方法来更新配置。在定义config字典后获取参数时，用此方法更新config字典非常方便。

例如，你可能希望在实例化模型后，添加模型的参数：

```python
swanlab.config.update({"model_params": "1.5B"})
```

## 4. 记录数据到SwanLab

创建一个字典，其中key是指标的名称，value是指标的值。将此字典对象传递给`swanlab.log`：

::: code-group

```python [记录一组指标]
metrics = {"loss": 0.5, "accuracy": 0.8}
swanlab.log(metrics)
```

```python [循环记录指标]
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data:
        prediction = model(input)
        loss = loss_fn(prediction, ground_truth)
        metrics = { "loss": loss }
        swanlab.log(metrics)
```

:::

如果您有很多指标，则可以在指标名称中使用前缀（如 `train/...` 和 `val/...`）。在 UI 中，SwanLab将自动对它们进行分组，来隔离不同门类的图表数据：

```python
metrics = {
    "train/loss": 0.5,
    "train/accuracy": 0.8,
    "val/loss": 0.6,
    "val/accuracy": 0.7,
}
swanlab.log(metrics)
```

有关`swanlab.log`的更多信息，请参阅[记录指标](../experiment_track/log-experiment-metric)章节。

## 5. 高级集成

您还可以在以下集成中查看高级 SwanLab 集成的形态：

- [HuggingFace Transformers](../integration/integration-huggingface-transformers.md)
- [PyTorch Lightning](../integration/integration-pytorch-lightning.md)
