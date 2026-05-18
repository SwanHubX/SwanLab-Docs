# 设置实验配置

使用 `swanlab.config` 保存你的训练配置，例如：
- 超参数
- 输入设置，例如数据集名称或模型类型
- 实验的任何其他变量

`swanlab.config` 使你可以轻松分析你的实验并在将来复现你的工作。你还可以在SwanLab应用中比较不同实验的配置，并查看不同的训练配置如何影响模型输出。

## 设置实验配置

`config` 通常在训练脚本的开头定义。当然，不同的人工智能工作流可能会有所不同，因此 `config` 也支持在脚本的不同位置定义，以满足灵活的需求。

以下部分概述了定义实验配置的不同场景。

### 在init中设置

下面的代码片段演示了如何使用Python字典定义 `config`，以及如何在初始化SwanLab实验时将该字典作为参数传递：

```python
import swanlab

# 定义一个config字典
config = {
  "hidden_layer_sizes": [64, 128],
  "activation": "ELU",
  "dropout": 0.5,
  "num_classes": 10,
  "optimizer": "Adam",
  "batch_normalization": True,
  "seq_length": 100,
}

# 在你初始化SwanLab时传递config字典
run = swanlab.init(project="config_example", config=config)
```

访问 `config` 中的值与在Python中访问其他字典的方式类似：

- 用键名作为索引访问值
  ```python
  hidden_layer_sizes = swanlab.config["hidden_layer_sizes"]
  ```
- 用 `get()` 方法访问值
  ```python
  activation = swanlab.config.get["activation"]
  ```
- 用点号访问值
  ```python
  dropout = swanlab.config.dropout
  ```

### 用argparse设置

你可以用 `argparse` 对象设置 `config`。`argparse` 是Python标准库（Python >= 3.2）中的一个非常强大的模块，用于从命令行接口（CLI）解析程序参数。这个模块让开发者能够轻松地编写用户友好的命令行界面。

可以直接传递 `argparse` 对象设置 `config`：

```python
import argparse
import swanlab

# 初始化Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=20)
parser.add_argument('--learning-rate', default=0.001)
args = parser.parse_args()

swanlab.init(config=args)
```

等同于 `swanlab.init(config={"epochs": 20, "learning-rate": 0.001})`

### 在脚本的不同位置设置

你可以在整个脚本的不同位置向 `config` 对象添加更多参数。

下面的代码片段展示了如何向 `config` 对象添加新的键值对：

```python
import swanlab

# 定义一个config字典
config = {
  "hidden_layer_sizes": [64, 128],
  "activation": "ELU",
  "dropout": 0.5,
  # ... 其他配置项
}

# 在你初始化SwanLab时传递config字典
run = swanlab.init(project="config_example", config=config)

# 在你初始化SwanLab之后，更新config
swanlab.config["dropout"] = 0.8
swanlab.config.epochs = 20
swanlab.config.set["batch_size", 32]
```

### 用配置文件设置

可以用json和yaml配置文件初始化 `config`，详情请查看[用配置文件创建实验](/guide_cloud/experiment_track/create-experiment-by-configfile)。