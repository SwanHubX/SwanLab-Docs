# swanlab.Histogram

[GitHub 源代码](https://github.com/SwanHubX/SwanLab/blob/feat/histogram/swanlab/sdk/internal/run/transforms/histogram/__init__.py)

```python
Histogram(
    sequence=None,
    bins=None,
    *,
    np_histogram=None,
    **kwargs,
)
```

将一组样本转换为分桶计数，通过 [`swanlab.log()`](./py-log.md) 记录每个 step 的分布快照，可用于跟踪模型参数、梯度等数据的分布变化。

| 参数           | 描述                                                                                                                                                              |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sequence`     | 原始样本，支持一维实数序列，如 `list`、`tuple`、`range` 和 NumPy 数组，也支持 PyTorch Tensor。还可传入已有的 `Histogram` 实例。未提供 `np_histogram` 时必须提供。 |
| `bins`         | 桶数或桶边界。桶数支持 Python / NumPy 整数，范围为 1–512；边界支持一维实数序列。默认 `None`，使用 64 个等宽桶。                                                   |
| `np_histogram` | 预分桶结果 `(counts, bin_edges)`，支持 `numpy.histogram()` 的返回值。默认 `None`。提供后忽略 `sequence`、`bins` 和 `num_bins`。                                   |
| `num_bins`     | 兼容 W&B 的桶数参数，通过关键字传入，仅在 `bins` 未指定时生效，取值要求与 `bins` 的桶数形式相同。与 `bins` 同时给出时以 `bins` 为准，并产生警告。                 |

`np_histogram` 和 `num_bins` 必须通过关键字传入。构造函数会忽略其余未识别的关键字参数。

## 从原始样本创建

传入桶数时，SDK 使用 `numpy.histogram()`，根据样本范围生成等宽桶：

```python
import numpy as np
import swanlab

swanlab.init(project="histogram-demo")

rng = np.random.default_rng(42)
for step in range(10):
    samples = rng.normal(loc=step * 0.1, size=1000)
    swanlab.log({"weights": swanlab.Histogram(samples, bins=64)}, step=step)

swanlab.finish()
```

每次构造 `Histogram` 都会独立计算分桶，不同 step 的桶边界可以不同。需要固定边界时，使用下面的显式边界方式。

### 指定桶边界

`bins` 可以传入非等宽边界。N 个桶需要 N+1 个边界：

```python
import swanlab

hist = swanlab.Histogram(
    [0.0, 0.5, 1.0, 2.0, 3.0],
    bins=[0.0, 1.0, 3.0],
)

print(hist.values)  # [2, 3]
print(hist.bins)  # [0.0, 1.0, 3.0]
```

除最后一桶外，区间为 `[bins[i], bins[i+1])`；最后一桶包含右端点。显式边界范围之外的样本会被丢弃，并产生警告。

## 从预分桶结果创建

已有 `numpy.histogram()` 计算结果时，通过 `np_histogram` 传入，SDK 校验后直接使用：

```python
import numpy as np
import swanlab

swanlab.init(project="histogram-demo")

samples = [0.0, 0.5, 1.0, 2.0, 3.0]
counts, bin_edges = np.histogram(samples, bins=[0.0, 1.0, 3.0])
hist = swanlab.Histogram(np_histogram=(counts, bin_edges))
swanlab.log({"weights": hist}, step=0)

swanlab.finish()
```

也可以直接传入计数与边界序列，例如 `swanlab.Histogram(np_histogram=([2, 3], [0.0, 1.0, 3.0]))`。

## 记录 PyTorch Tensor

PyTorch Tensor 可以直接传入，SDK 内部自动执行 `detach().cpu()` 转换，GPU 上的 Tensor、带梯度的 Tensor 以及 `bfloat16` 精度均受支持。多维参数需要先展平为一维：

```python
import torch
import swanlab

swanlab.init(project="histogram-demo")

layer = torch.nn.Linear(8, 4)
swanlab.log({"weights": swanlab.Histogram(layer.weight.flatten())}, step=0)

swanlab.finish()
```

## 数据规则与异常

- 桶数必须在 1–512 之间。显式边界和预分桶输入同样受此限制。
- 原始样本中的 `NaN`、`+Inf` 和 `-Inf` 会被丢弃，并产生警告。使用桶数分桶时，空序列或过滤后没有有限值会抛出 `ValueError`；提供合法显式边界时，允许生成计数全为 0 的直方图。
- 桶边界必须是一维有限实数序列，按非递减顺序排列。重复边界允许使用，但会警告产生零宽桶。
- 预分桶计数必须是一维、有限、非负的整数值，且不超过 `2**64 - 1`。`1.0` 等整数值浮点数会转换为整数；不支持概率密度或含小数的加权计数。
- 预分桶结果必须满足 `len(bin_edges) == len(counts) + 1`。
- `swanlab.Histogram(hist)` 可以复用已有实例的分桶结果，此时不能额外指定 `bins`、`num_bins` 或 `np_histogram`。

类型或形状不符合要求的输入不会抛出异常：实例被标记为无效，错误详情写入实验的调试日志，`swanlab.log()` 记录时跳过该条直方图，终端仅提示该指标没有产生数据。传入不支持的对象、非数值数据、多维数组，或把布尔值作为桶数时都会走这条路径。

上文列出的数值与长度规则不满足时抛出 `ValueError`。`sequence` 与 `np_histogram` 同时缺省、嵌套 `Histogram` 时额外指定分桶参数，同样抛出 `ValueError`。

## 读取分桶结果

构造后的 `Histogram` 提供以下属性：

| 属性       | 类型          | 说明                                                 |
| ---------- | ------------- | ---------------------------------------------------- |
| `values`   | `list[int]`   | 每个桶的样本计数，长度为 N。                         |
| `bins`     | `list[float]` | 桶边界，长度为 N+1。                                 |
| `is_valid` | `bool`        | 输入是否通过类型与形状校验，无效实例在记录时被跳过。 |

记录的数据保留每个 step 的计数和边界，无法恢复原始样本。使用独立的指标名记录直方图，同一个指标名不要混用标量、直方图或其他数据类型。
