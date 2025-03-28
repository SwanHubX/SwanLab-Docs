# log

[Github源代码](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/sdk.py)

```python
log(
    data: Dict[str, DataType],
    step: int = None,
    print_to_console: bool = False,
)
```

| 参数   | 描述                                       |
|--------|------------------------------------------|
| data   | (Dict[str, DataType]) 必须。传入一个键值对字典，key为指标名，value为指标值。value支持int、float、可被float()转换的类型、或任何`BaseType`类型。 |
| step   | (int) 可选，该参数设置了data的步数。如不设置step，则将以0开始，后续每1次step累加1。 |
| print_to_console | (bool) 可选，默认值为False。当设置为True时，会将data的key和value以字典的形式打印到终端。 |

## 介绍

`swanlab.log`是指标记录的核心API，使用它记录实验中的数据，例如标量、图像、音频和文本。  

最基本的用法是如下面代码所示，这将会将准确率与损失值记录到实验中，生成可视化图表并更新这些指标的汇总值（summary）。：

```python
swanlab.log({"acc": 0.9, "loss":0.1462})
```

除了标量以外，`swanlab.log`支持记录多媒体数据，包括图像、音频、文本等，并在UI上有很好的显示效果。

## 打印传入的字典

`swanlab.log`支持打印传入的`data`的`key`和`value`到终端，默认情况下不打印。要开启打印的话，需要设置`print_to_console=True`。

```python
swanlab.log({"acc": 0.9, "loss":0.1462}, print_to_console=True)
```

当然，你也可以用这种方式打印：

```python
print(swanlab.log({"acc": 0.9, "loss":0.1462}))
```

## 更多用法

- 记录[图像](/api/py-Image.md)
- 记录[音频](/api/py-Audio.md)
- 记录[文本](/api/py-Text.md)