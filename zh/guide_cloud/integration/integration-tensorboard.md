# Tensorboard

[TensorBoard](https://github.com/tensorflow/tensorboard) 是 Google TensorFlow 提供的一个可视化工具，用于帮助理解、调试和优化机器学习模型。它通过图形界面展示训练过程中的各种指标和数据，让开发者更直观地了解模型的性能和行为。

![TensorBoard](/assets/ig-tensorboard.png)

你可以使用`swanlab convert`将Tensorboard生成的Tfevent文件转换成SwanLab实验。

## 方式一：命令行转换

```
swanlab convert [TFEVENT_LOGDIR]
```

## 方式二：代码内

```python
from swanlab.convert import TFBConverter
```