# Ultralytics

[Ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8 是一款尖端、最先进的 （SOTA） 模型，它建立在以前 YOLO 版本的成功基础上，并引入了新功能和改进，以进一步提高性能和灵活性。YOLOv8 设计为快速、准确且易于使用，使其成为各种对象检测和跟踪、实例分割、图像分类和姿态估计任务的绝佳选择。

![ultralytics](/assets/ig-ultralytics.png)

你可以使用Ultralytics快速进行计算机视觉模型训练，同时使用SwanLab进行实验跟踪与可视化。

## 1.引入add_swanlab_callback

```python
from swanlab.integration.ultralytics import add_swanlab_callback
```

`add_swanlab_callback`的作用是为Ultralytics模型添加回调函数，以在模型训练的各个生命周期执行SwanLab记录。

## 2.代码案例

下面是使用yolov8n模型在coco数据集上的训练，只需将model传入`add_swanlab_callback`函数，即可完成与SwanLab的集成。

```python {9}
from ultralytics import YOLO
from swanlab.integration.ultralytics import add_swanlab_callback


if __name__ == "__main__":
    model = YOLO("yolov8n.yaml")
    model.load()
    # 添加swanlab回调
    add_swanlab_callback(model)

    model.train(
        data="./coco.yaml",
        epochs=50, 
        imgsz=320,
    )
```