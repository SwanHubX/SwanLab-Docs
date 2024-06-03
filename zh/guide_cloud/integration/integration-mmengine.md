# MMEngine

:::info 教程
[mmdetection如何使用swanlab远程查看训练日志](https://zhuanlan.zhihu.com/p/699058426)
:::

[MMEngine](https://github.com/open-mmlab/mmengine) 是一个由 [OpenMMLab](https://openmmlab.com/) 社区开发的深度学习训练框架，专为深度学习研究和开发而设计。MMEngine 提供了一种高效、灵活且用户友好的方式来构建、训练和测试深度学习模型，尤其是在计算机视觉领域。它的目标是简化研究人员和开发者在深度学习项目中的工作流程，并提高其开发效率。

<div align="center">
<img src="/assets/integration-mmengine.jpeg" width=440>
</div>

MMEngine 为 OpenMMLab 算法库实现了下一代训练架构，为 OpenMMLab 中的 30 多个算法库提供了统一的执行基础。其核心组件包括训练引擎、评估引擎和模块管理。

SwanLab将专为MMEngine设计的`SwanlabVisBackend`集成到MMEngine中，可用于记录训练、评估指标、记录实验配置、记录图像等。

## 1. 引入SwanlabVisBackend

将如下内容添加到mm系列框架的任意config文件中, 其中`init_kwargs`中填入的参数字典与`swanlab.init`的规则一致:

```python
# swanlab visualizer
custom_imports = dict(  # 引入SwanLab作为日志记录器
    imports=["swanlab.integration.mmengine"], allow_failed_imports=False
)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="SwanlabVisBackend",
        init_kwargs={ # swanlab.init 参数
            "project": "swanlab-mmengine",
            "project": "MMDetection",  # 项目名称
            "experiment_name": "faster-rcnn",  # 实验名称
            "description": "faster-rcnn r50 fpn 1x coco",  # 实验的描述信息
        },
    ),
]

visualizer = dict(
    type="Visualizer",
    vis_backends=vis_backends,
    name="visualizer",
)
```

## 2.传入visualizer，开始训练

:::info
如果用官方自带的训练脚本，那么这一步已经默认做了，无需做改动。
:::

接下来，只需在训练脚本中，初始化一个`runner`，传入`visualizer`即可：

```python (12)
from mmengine.runner import Runner

...

# 构建mmengine的Runner
runner = Runner(
    model,
    work_dir='runs/gan/',
    train_dataloader=train_dataloader,
    train_cfg=train_cfg,
    optim_wrapper=opt_wrapper_dict,
    visualizer=visualizer,
)

# 开始训练
runner.train()
```

## 3.案例：MMDetection训练faster-rcnn

首先克隆[MMDetction](https://github.com/open-mmlab/mmdetection)项目到本地。

然后在faster-rnn对应的config文件（`configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py`）的最后增加下面的代码：

```python
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# swanlab
custom_imports = dict(  # 引入SwanLab作为日志记录器
    imports=["swanlab.integration.mmengine"], allow_failed_imports=False
)
vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="SwanlabVisBackend",
        init_kwargs={  # swanlab.init 参数
            "project": "MMDetection",  # 项目名称
            "experiment_name": "faster-rcnn",  # 实验名称
            "description": "faster-rcnn r50 fpn 1x coco",  # 实验的描述信息
        },
    ),
]
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
```

**然后开启训练即可**：

```bash
python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py
```

![ig-mmengine-1](/assets/ig-mmengine-1.png)

**在swanlab中远程查看训练日志**：

![ig-mmengine-2](/assets/ig-mmengine-2.png)