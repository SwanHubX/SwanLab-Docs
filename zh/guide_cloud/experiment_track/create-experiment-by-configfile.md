# 用配置文件创建实验

本节将介绍如何使用json、yaml格式的配置文件来创建SwanLab实验。

## swanlab.config载入配置文件

`swanlab.init`的`config`参数支持传入json或yaml格式的配置文件路径，并将配置文件解析为字典以进行实验创建。

### 使用json文件

下面是一个json格式的配置文件示例：

```json
{
    "epochs": 20,
    "learning-rate": 0.001,
}
```

将配置文件的路径传入config参数，它会把配置文件解析为字典：

```python
swanlab.init(config="swanlab-init-config.json")
# 等价于swanlab.init(config={"epochs": 20, "learning-rate": 0.001})
```

### 使用yaml文件

下面是一个yaml格式的配置文件示例：

```yaml
epochs: 20
learning-rate: 0.001
```

将配置文件的路径传入`config`参数，它会把配置文件解析为字典：
```python
swanlab.init(config="swanlab-init-config.yaml")
# 等价于swanlab.init(config={"epochs": 20, "learning-rate": 0.001})
```

## swanlab.init载入配置文件

`swanlab.init`的`load`参数支持传入json或yaml格式的配置文件路径，并解析配置文件以进行实验创建。

### 使用json文件

下面是一个json格式的配置文件示例：

```json
{
    "project": "cat-dog-classification",
    "experiment_name": "Resnet50",
    "description": "我的第一个人工智能实验",
    "config":{
        "epochs": 20,
        "learning-rate": 0.001}
}
```

将配置文件的路径传入`load`参数，它会解析配置文件以初始化实验：

```python
swanlab.init(load="swanlab-config.json")
# 等价于
# swanlab.init(
#     project="cat-dog-classification",
#     experiment_name="Resnet50",
#     description="我的第一个人工智能实验",
#     config={
#         "epochs": 20,
#         "learning-rate": 0.001}
# )
```

### 使用yaml文件

下面是一个json格式的配置文件示例：

```yaml
project: cat-dog-classification
experiment_name: Resnet50
description: 我的第一个人工智能实验
config:
  epochs: 20
  learning-rate: 0.001
```

将配置文件的路径传入`load`参数，它会解析配置文件以初始化实验：

```python
swanlab.init(load="swanlab-config.yaml")
# 等价于
# swanlab.init(
#     project="cat-dog-classification",
#     experiment_name="Resnet50",
#     description="我的第一个人工智能实验",
#     config={
#         "epochs": 20,
#         "learning-rate": 0.001}
# )
```

## 常见问题

### 1. 配置文件命名是固定的吗？

配置文件的命名是自由的，但推荐使用`swanlab-init`和`swanlab-init-config`这两个配置名。

### 2. 配置文件和脚本内的参数之间是什么关系？

脚本内参数的优先级大于配置文件，即脚本内参数会覆盖配置文件参数。

比如，下面有一段yaml配置文件和示例代码片段：

```yaml
project: cat-dog-classification
experiment_name: Resnet50
description: 我的第一个人工智能实验
config:
  epochs: 20
  learning-rate: 0.001
```

```python
swanlab.init(
    experiment_name="resnet101"，
    config={"epochs": 30},
    load="swanlab-init.yaml"
)
```

最终`experiment_name`为resnet101，`config`为{"epochs":30}。