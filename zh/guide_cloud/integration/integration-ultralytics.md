# Ultralytics

[![](/assets/colab.svg)](https://colab.research.google.com/drive/1RAT2vSrvET4wEDd9syeDrgz0KBUDQAR1?usp=sharing)

[Ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8 是一款尖端、最先进的 （SOTA） 模型，它建立在以前 YOLO 版本的成功基础上，并引入了新功能和改进，以进一步提高性能和灵活性。YOLOv8 设计为快速、准确且易于使用，使其成为各种对象检测和跟踪、实例分割、图像分类和姿态估计任务的绝佳选择。

![ultralytics](./ultralytics/logo.png)

你可以使用Ultralytics快速进行计算机视觉模型训练，同时使用SwanLab进行实验跟踪与可视化。

下面介绍两种引入SwanLab的方式：  
1. `add_swanlab_callback`：无需修改源码，适用于单卡训练场景
2. `return_swanlab_callback`：需要修改源码，适用于单卡以及多卡DDP训练场景

## 1.1 引入add_swanlab_callback

```python
from swanlab.integration.ultralytics import add_swanlab_callback
```

`add_swanlab_callback`的作用是为Ultralytics模型添加回调函数，以在模型训练的各个生命周期执行SwanLab记录。

## 1.2 代码案例

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
        data="./coco128.yaml",
        epochs=3, 
        imgsz=320,
    )
```

如果需要自定义SwanLab的项目、实验名等参数，则可以在`add_swanlab_callback`中添加：

```python
add_swanlab_callback(
    model,
    project="ultralytics",
    experiment_name="yolov8n",
    description="yolov8n在coco128数据集上的训练。",
    mode="local",
    )
```

## 2.1 多卡训练/DDP训练

> swanlab>=0.3.7

在Ultralytics多卡训练的场景下，由于启动训练的方式与单卡完全不同，所以需要用一种不同的方式接入SwanLab回调。

这是一个ultralytics开启DDP训练的样例代码：

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    model.train(
        data="./coco128.yaml",
        epochs=3, 
        imgsz=320,
        # 开启DDP
        device=[0,1],
    )
```

我们需要修改ultralytics的源码，去到`ultralytics/utils/callbacks/base.py`，找到`add_integration_callbacks`函数，添加下面的三行代码：

```python (15,16,18)
def add_integration_callbacks(instance):
    ...
    
    # Load training callbacks
    if "Trainer" in instance.__class__.__name__:
        from .clearml import callbacks as clear_cb
        from .comet import callbacks as comet_cb
        from .dvc import callbacks as dvc_cb
        from .mlflow import callbacks as mlflow_cb
        from .neptune import callbacks as neptune_cb
        from .raytune import callbacks as tune_cb
        from .tensorboard import callbacks as tb_cb
        from .wb import callbacks as wb_cb

        from swanlab.integration.ultralytics import return_swanlab_callback
        sw_cb = return_swanlab_callback()

        callbacks_list.extend([..., sw_cb])
```

然后运行，就可以在ddp下正常跟踪实验了。

如果需要自定义SwanLab的项目、实验名等参数，则可以在`return_swanlab_callback`中添加：

```python
return_swanlab_callback(
    model,
    project="ultralytics",
    experiment_name="yolov8n",
    description="yolov8n在coco128数据集上的训练。",
    mode="local",
    )
```

:::warning ps
1. 写入源码之后，之后运行就不需要在训练脚本中增加`add_swanlab_callback`了。
2. 项目名由model.train()的project参数定义，实验名由name参数定义。
:::

## 2.2 代码案例

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    model.train(
        data="./coco128.yaml",
        epochs=3, 
        imgsz=320,
        # 开启DDP
        device=[0,1,2,3],
        # 可以通过project参数设置SwanLab的project，name参数设置SwanLab的experiment_name
        project="Ultralytics",
        name="yolov8n"
    )
```