# 记录实验指标

使用SwanLab Python库记录训练每一步（step）的指标与媒体数据。

SwanLab用 `swanlab.log()` 在训练循环中收集指标名和数据（key-value），然后同步到云端服务器。

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/zh/guide_cloud/experiment_track/log-experiment-metric/line.png)

## 记录标量指标

在训练循环中，将指标名和数据组成一个键值对字典，传递给 `swanlab.log()` 完成1次指标的记录：

```python
for epoch in range(num_epochs):
    for data, ground_truth in dataloader:
        predict = model(data)
        loss = loss_fn(predict, ground_truth)
        # 记录指标，指标名为loss
        swanlab.log({"loss": loss})
```

在 `swanlab.log` 记录时，会根据指标名，将`{指标名: 指标}`字典汇总到一个统一位置存储。

⚠️需要注意的是，`swanlab.log({key: value})`中的value必须是`int` / `float` / `BaseType`这三种类型（如果传入的是`str`类型，会先尝试转为`float`，如果转换失败就会报错），其中`BaseType`类型主要是多媒体数据，详情请看[记录多媒体数据](./log-media.md)。

在每次记录时，会为该次记录赋予一个 `step`。在默认情况下，`step` 为0开始，并在你每一次在同一个指标名下记录时，`step` 等于该指标名历史记录的最大 `step` + 1，例如：

```python
import swanlab
swanlab.init()

...

swanlab.log({"loss": loss, "acc": acc})
# 此次记录中，loss的step为0, acc的step为0

swanlab.log({"loss": loss, "iter": iter})
# 此次记录中，loss的step为1, iter的step为0, acc的step为0

swanlab.log({"loss": loss, "iter": iter})
# 此次记录中，loss的step为2, iter的step为1, acc的step为0
```

## 指标分组

在脚本中可以通过指标名的前缀（以“/”为分隔）进行图表分组，例如 `train/loss` 会被分到名为“train”的分组、`val/loss` 会被分到名为 `val` 的分组：

```python
# 分到train组
swanlab.log({"train/loss": loss})
swanlab.log({"train/batch_cost": batch_cost})

# 分到val组
swanlab.log({"val/acc": acc})
```

:::tip
对于存在多个 `/` 分割线的指标名，目前的策略以最后一个分割线为准。
例如指标名称为 `a/b/c`，默认以 `a/b` 作为分组名称。
如果需要自定义分组名称，可以通过 [swanlab.define_metric()](../../api/py-define_metric.md) 在指标被上报之前进行定义。
:::

## 指定记录的step

在一些指标的记录频率不一致，但希望它们的step可以对齐时，可以通过设置 `swanlab.log` 的 `step` 参数实现对齐：

```python
for iter, (data, ground_truth) in enumerate(train_dataloader):
    predict = model(data)
    train_loss = loss_fn(predict, ground_truth)
    swanlab.log({"train/loss": loss}, step=iter)

    # 测试部分
    if iter % 1000 == 0:
        acc = val_trainer(model)
        swanlab.log({"val/acc": acc}, step=iter)
```

需要注意的是，同一个指标名不允许出现2个相同的step的数据，一旦出现，SwanLab将保留先记录的数据，抛弃后记录的数据。

## 打印指标

也许你希望在训练循环中打印指标，可以通过 `print_to_console` 参数控制是否将指标打印到控制台（以`dict`的形式）：

```python
swanlab.log({"acc": acc}, print_to_console=True)
```

或者：

```python
print(swanlab.log({"acc": acc}))
```

## 自动记录环境信息

SwanLab在实验期间自动记录以下信息：

- **命令行输出**：标准输出流和标准错误流被自动记录，并显示在实验页面的”日志”选项卡中。
- **实验环境**：记录包括操作系统、硬件配置、Python解释器路径、运行目录、Python库依赖等在内的数十项的环境信息。
- **训练时间**：记录训练开始时间和总时长。

## 异步记录指标

:::info
`swanlab.async_log()` 需要 SwanLab SDK **v0.8.0 或更高版本**。
:::

如果你需要记录需要耗时计算或 I/O 的指标，可以使用 `swanlab.async_log()` 在后台执行计算，不会阻塞训练循环。它接受与 `swanlab.log()` 相同的数据格式，并立即返回一个 `Future`。

```python
import swanlab
import time

swanlab.init(project=”my-project”)

def compute_metric():
    time.sleep(2)  # 模拟耗时计算
    return {“score”: 0.95}

# 在后台执行，训练继续不会被阻塞
future = swanlab.async_log(compute_metric, step=1)

swanlab.finish()
```

`swanlab.async_log()` 支持多种执行模式（`threading`、`asyncio`、`spawn`）。详细用法和所有模式选项请参阅 [async_log API 文档](../../api/py-async-log.md)。

## 自定义 X 轴

:::info
`swanlab.define_metric()` 自定义 X 轴需要 SwanLab SDK **v0.10.0 或更高版本**。
:::

默认情况下，指标图表以 step 作为 X 轴。在部分训练场景下（如希望按 epoch、学习率等查看指标变化），可以通过 `swanlab.define_metric()` 将图表的 X 轴关联为另一个指标：

```python
import swanlab

swanlab.init(project="my-project")

# train/loss 以 train/epoch 作为 X 轴
swanlab.define_metric("train/loss", x_axis="train/epoch")

for epoch in range(num_epochs):
    # 先记录 X 轴指标
    swanlab.log({"train/epoch": epoch})
    # ... 训练 ...
    swanlab.log({"train/loss": loss})
```

X 轴指标与 Y 轴指标可以分开记录，SDK 会自动为 Y 值补上最近一次的 X 值；`key` 也支持 glob 批量匹配（如 `train/*`），方便对一组指标统一定义 X 轴。

自定义 X 轴的详细参数说明与注意事项，请参阅 [define_metric API 文档](../../api/py-define_metric.md)。
