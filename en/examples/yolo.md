# YOLO Object Detection

:::info
Object Detection, Computer Vision
:::

[Online Demo](https://swanlab.cn/@ZeyiLin/ultratest/runs/yux7vclmsmmsar9ear7u5/chart) | [YOLO Cat and Dog Detection Tutorial](https://zhuanlan.zhihu.com/p/702525559)

## Overview

YOLO (You Only Look Once) is an object detection model proposed by Joseph Redmon et al., widely used in various computer vision tasks. YOLO divides the image into a grid and predicts bounding boxes and class probabilities within each grid cell, enabling real-time object detection and performing well in many tasks.

In this task, we will use the YOLO model for object detection on the COCO128 dataset while using SwanLab for monitoring and visualization.

![yolo](/assets/example-yolo-1.png)

The COCO128 dataset is a small object detection dataset derived from the widely used COCO (Common Objects in Context) dataset. The COCO128 dataset contains 128 images, a subset of the COCO dataset, primarily used for quick testing and debugging of object detection models.

## Environment Setup

This case study is based on `Python>=3.8`. Please ensure Python is installed on your computer. Environment dependencies:

```txt
ultralytics
swanlab
```

Quick installation command:

```bash
pip install ultralytics swanlab
```

> The code in this article is tested with ultralytics==8.2.18, swanlab==0.3.6

## Complete Code

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

## Demonstration of Results

![yolo-2](/assets/example-yolo-2.png)

![yolo-3](/assets/example-yolo-3.png)