# Yolo目标检测

:::info
目标检测、计算机视觉
:::

[在线Demo](https://swanlab.cn/@ZeyiLin/ultratest/runs/yux7vclmsmmsar9ear7u5/chart) | [YOLO猫狗检测教程](https://zhuanlan.zhihu.com/p/702525559)

## 概述

YOLO（You Only Look Once）是一种由Joseph Redmon等人提出的目标检测模型，广泛应用于各种计算机视觉任务。YOLO通过将图像分成网格，并在每个网格内预测边界框和类别概率，能够实现实时的目标检测，在许多任务上表现出色。

在这个任务中，我们将使用YOLO模型在COCO128数据集上进行目标检测任务，同时用SwanLab进行监控和可视化。

![yolo](/assets/example-yolo-1.png)

COCO128 数据集是一个小型的目标检测数据集，来源于广泛使用的 COCO（Common Objects in Context）数据集。COCO128 数据集包含 128 张图像，是 COCO 数据集的一个子集，主要用于快速测试和调试目标检测模型。

## 环境安装

本案例基于`Python>=3.8`，请在您的计算机上安装好Python。 环境依赖：

```txt
ultralytics
swanlab
```

快速安装命令：

```bash
pip install ultralytics swanlab
```

> 本文的代码测试于ultralytics==8.2.18、swanlab==0.3.6

## 完整代码

```python
from ultralytics import YOLO
from swanlab.integration.ultralytics import add_swanlab_callback

def main():
    model = YOLO("yolov8n.pt")
    add_swanlab_callback(model)
    model.train(data="coco128.yaml", epochs=5, imgsz=640, batch=64)

if __name__ == "__main__":
    main()
```


## 演示效果

![yolo-2](/assets/example-yolo-2.png)

![yolo-3](/assets/example-yolo-3.png)