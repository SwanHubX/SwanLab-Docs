# Ultralytics

[Ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8 is a cutting-edge, state-of-the-art (SOTA) model that builds on the success of previous YOLO versions and introduces new features and improvements to further enhance performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a variety of object detection and tracking, instance segmentation, image classification, and pose estimation tasks.

![ultralytics](/assets/ig-ultralytics.png)

You can use Ultralytics to quickly train computer vision models while using SwanLab for experiment tracking and visualization.

Below are two methods to integrate SwanLab:
1. `add_swanlab_callback`: No need to modify the source code, suitable for single-card training scenarios.
2. `return_swanlab_callback`: Requires modifying the source code, suitable for single-card and multi-card DDP training scenarios.

## 1.1 Introducing add_swanlab_callback

```python
from swanlab.integration.ultralytics import add_swanlab_callback
```

The `add_swanlab_callback` function is used to add callback functions to the Ultralytics model, enabling SwanLab logging at various lifecycle stages of the model training.

## 1.2 Code Example

Below is an example of training using the yolov8n model on the COCO dataset. Simply pass the model to the `add_swanlab_callback` function to complete the integration with SwanLab.

```python {9}
from ultralytics import YOLO
from swanlab.integration.ultralytics import add_swanlab_callback


if __name__ == "__main__":
    model = YOLO("yolov8n.yaml")
    model.load()
    # Add swanlab callback
    add_swanlab_callback(model)

    model.train(
        data="./coco128.yaml",
        epochs=3, 
        imgsz=320,
    )
```

If you need to customize the SwanLab project, experiment name, and other parameters, you can add them in the `add_swanlab_callback` function:

```python
add_swanlab_callback(
    model,
    project="ultralytics",
    experiment_name="yolov8n",
    description="Training of yolov8n on the coco128 dataset.",
    mode="local",
    )
```

## 2.1 Multi-Card Training/DDP Training

> swanlab>=0.3.7

In the case of Ultralytics multi-card training, since the method of starting training is completely different from single-card training, a different approach is needed to integrate SwanLab callbacks.

Here is an example code for starting DDP training with Ultralytics:

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    model.train(
        data="./coco128.yaml",
        epochs=3, 
        imgsz=320,
        # Enable DDP
        device=[0,1],
    )
```

We need to modify the Ultralytics source code. Go to `ultralytics/utils/callbacks/base.py`, find the `add_integration_callbacks` function, and add the following three lines of code:

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

Then run, and you can track experiments normally under DDP.

If you need to customize the SwanLab project, experiment name, and other parameters, you can add them in the `return_swanlab_callback` function:

```python
return_swanlab_callback(
    model,
    project="ultralytics",
    experiment_name="yolov8n",
    description="Training of yolov8n on the coco128 dataset.",
    mode="local",
    )
```

:::warning ps
1. After writing to the source code, you no longer need to add `add_swanlab_callback` in the training script for subsequent runs.
2. The project name is defined by the `project` parameter in `model.train()`, and the experiment name is defined by the `name` parameter.
:::

## 2.2 Code Example

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    model.train(
        data="./coco128.yaml",
        epochs=3, 
        imgsz=320,
        # Enable DDP
        device=[0,1,2,3],
        # Set SwanLab project with the project parameter and experiment name with the name parameter
        project="Ultralytics",
        name="yolov8n"
    )
```