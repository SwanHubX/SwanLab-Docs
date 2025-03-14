# CSV表格记录器

如果你希望在训练过程中，将一些配置信息、指标信息记录在本地的CSV文件中（格式和SwanLab网页中的“表格视图“一致），那么非常推荐你使用`CSV记录器`插件。

:::warning 改进插件
SwanLab插件均为开源代码，你可以在[Github源代码](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/plugin/writer.py)中查看，欢迎提交你的建议和PR！
:::

## 插件用法

**1. 初始化CSV记录器：**

```python
from swanlab.plugin.writer import CSVWriter

csv_writer = CSVWriter(dir="logs")
```

`dir`参数指定了CSV文件的保存路径，默认保存到当前工作目录。

**2. 传入插件：**

```python
swanlab.init(
    ...
    callbacks=[csv_writer]
)
```

执行代码后，就会在`logs`目录下生成一个`swanlab_run.csv`文件，并开始记录数据。后续的每一次训练，都会在该csv文件中添加新的行。

如果想要指定其他文件名，可以传入`filename`参数：

```python
csv_writer = CSVWriter(dir="logs", filename="my_csv_file.csv")
```


## 示例代码


```python
import swanlab
from swanlab.plugin.writer import CSVWriter
import random

csv_writer = CSVWriter(dir="logs")

# 创建一个SwanLab项目
swanlab.init(
    # 设置项目名
    project="my-awesome-project",
    
    # 设置超参数
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
        "batch_size": 128
    },
    callbacks=[csv_writer]
)

# 模拟一次训练
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
  acc = 1 - 2 ** -epoch - random.random() / epoch - offset
  loss = 2 ** -epoch + random.random() / epoch + offset

  # 记录训练指标
  swanlab.log({"acc": acc, "loss2": loss})

# [可选] 完成训练，这在notebook环境中是必要的
swanlab.finish()
```