# log

[Github源代码](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/sdk.py)

```python
log(
    data: Dict[str, DataType],
    step: int = None,
)
```

| 参数   | 描述                                       |
|--------|------------------------------------------|
| data   | (Dict[str, DataType]) 必须。传入一个键值对字典，key为指标名，value为指标值。value支持int、float、可被float()转换的类型、或任何`BaseType`类型。 |
| step   | (int) 可选，该参数设置了data的步数。如不设置step，则将以0开始，后续每1次step累加1。 |

## 介绍

`swanlab.log`是指标记录的核心API，使用它记录实验中的数据，例如标量、图像、音频和文本。  

最基本的用法是如下面代码所示，这将会将准确率与损失值记录到实验中，生成可视化图表并更新这些指标的汇总值（summary）。：

```python
swanlab.log({"acc": 0.9, "loss":0.1462})
```

除了标量以外，`swanlab.log`支持记录多媒体数据，包括图像、音频、文本等，并在UI上有很好的显示效果。

## 更多用法

- 记录[图像](/zh/api/py-Image.md)
- 记录[音频](/zh/api/py-Audio.md)
- 记录[文本](/zh/api/py-Text.md)