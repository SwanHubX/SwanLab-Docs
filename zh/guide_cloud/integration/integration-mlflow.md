# MLFlow

[MLFlow](https://github.com/mlflow/mlflow) 是一个开源的机器学习生命周期管理平台，由 Databricks 创建并维护。它旨在帮助数据科学家和机器学习工程师更高效地管理机器学习项目的整个生命周期，包括实验跟踪、模型管理、模型部署和协作。MLflow 的设计是模块化的，可以与任何机器学习库、框架或工具集成。

![mlflow](./mlflow/logo.png)

:::warning 其他工具的同步教程

- [TensorBoard](/zh/guide_cloud/integration/integration-tensorboard.md)
- [Weights & Biases](/zh/guide_cloud/integration/integration-wandb.md)
:::

**你可以用将MLflow上的项目转换到SwanLab：**

::: info
在当前版本暂仅支持转换标量图表。
:::

[[toc]]


## 1. 准备工作

**（必须）mlflow服务的url链接**

首先，需要记下mlflow服务的**url链接**，如`http://127.0.0.1:5000`。

> 如果还没有启动mlflow服务，那么需要使用`mlflow ui`命令启动服务，并记下url链接。

**（可选）实验ID**

如果你只想转换其中的一组实验，那么在下图所示的地方，记下该实验ID。

![](./mlflow/ui-1.png)

## 2. 方式一：命令行转换

转换命令行：

```bash
swanlab convert -t mlflow --mlflow-url <MLFLOW_URL> --mlflow-exp <MLFLOW_EXPERIMENT_ID>
```

支持的参数如下：

- `-t`: 转换类型，可选wandb、tensorboard和mlflow。
- `-p`: SwanLab项目名。
- `-w`: SwanLab工作空间名。
- `--cloud`: (bool) 是否上传模式为"cloud"，默认为True
- `-l`: logdir路径。
- `--mlflow-url`: mlflow服务的url链接。
- `--mlflow-exp`: mlflow实验ID。

如果不填写`--mlflow-exp`，则会将指定项目下的全部实验进行转换；如果填写，则只转换指定的实验组。

## 3. 方式二：代码内转换

```python
from swanlab.converter import MLFLowConverter

mlflow_converter = MLFLowConverter(project="mlflow_converter")
# mlflow_exp可选
mlflow_converter.run(tracking_uri="http://127.0.0.1:5000", experiment="1")
```

效果与命令行转换一致。

`MLFLowConverter`支持的参数：

- `project`: SwanLab项目名。
- `workspace`: SwanLab工作空间名。
- `cloud`: (bool) 是否上传模式为"cloud"，默认为True。
- `logdir`: wandb Run（项目下的某一个实验）的id。