# MMDetection

:::info Tutorial
[How to Use SwanLab to View Training Logs Remotely with MMDetection](https://zhuanlan.zhihu.com/p/699058426)
:::

<div align="center">
<img src="/assets/integration-mmdetection.png" width=600>
</div>

[MMDetection](https://github.com/open-mmlab/mmdetection) is a deep learning training framework developed by the [OpenMMLab](https://openmmlab.com/) community, built on top of the PyTorch deep learning framework. It aims to provide researchers and engineers with an efficient, flexible, and easily extensible platform for object detection. MMDetection supports various mainstream object detection methods and offers a large number of pre-trained models and rich configuration options, making applications and development in object detection tasks more convenient.

<div align="center">
<img src="/assets/integration-mmdetection-intro.png">
</div>

You can modify the MMDetection configuration file to use SwanLab as an experiment logging tool.

## Specify SwanLab as VisBackend in the Configuration File

Ensure you have installed SwanLab, or use `pip install -U swanlab` to install the latest version.

Add the following content to the MMDetection config file, where the dictionary of parameters in `init_kwargs` follows the rules of `swanlab.init`:

```python
# swanlab visualizer
custom_imports = dict(  # Import SwanLab as a logger, for projects that do not support custom_imports, directly initialize SwanlabVisBackend and add it to vis_backends
    imports=["swanlab.integration.mmengine"], allow_failed_imports=False
)

vis_backends = [
    dict(
        type="SwanlabVisBackend",
        init_kwargs={ # swanlab.init parameters
            "project": "swanlab-mmengine",
            "experiment_name": "Your exp",  # Experiment name
            "description": "Note whatever you want",  # Description of the experiment
        },
    ),
]

visualizer = dict(
    type="Visualizer",
    vis_backends=vis_backends,
    name="visualizer",
)
```

> For other import methods and more flexible configurations, refer to [MMEngine Integration with SwanLab](https://docs.swanlab.cn/zh/guide_cloud/integration/integration-mmengine.html)

## Example: Training Faster-RCNN with MMDetection

First, clone the [MMDetection](https://github.com/open-mmlab/mmdetection) project to your local machine.

Then, add the following code at the end of the corresponding config file for faster-rcnn (`configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py`):

```python
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# swanlab
custom_imports = dict(  # Import SwanLab as a logger
    imports=["swanlab.integration.mmengine"], allow_failed_imports=False
)
vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="SwanlabVisBackend",
        init_kwargs={  # swanlab.init parameters
            "project": "MMDetection",  # Project name
            "experiment_name": "faster-rcnn",  # Experiment name
            "description": "faster-rcnn r50 fpn 1x coco",  # Description of the experiment
        },
    ),
]
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
```

**Then start the training:**

```bash
python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py
```

![ig-mmengine-1](/assets/ig-mmengine-1.png)

**View training logs remotely in SwanLab:**

![ig-mmengine-2](/assets/ig-mmengine-2.png)