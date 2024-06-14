# MMDetection

:::info 教程
[mmdetection如何使用swanlab远程查看训练日志](https://zhuanlan.zhihu.com/p/699058426)
:::

<div align="center">
<img src="/assets/integration-mmdetection.png" width=600>
</div>

[MMdetection](https://github.com/open-mmlab/mmdetection) 是一个由 [OpenMMLab](https://openmmlab.com/) 社区开发的深度学习训练框架，建立在 PyTorch 深度学习框架之上，旨在为研究人员和工程师提供一个高效、灵活、易于扩展的目标检测平台。MMDetection 支持多种主流的目标检测方法，并提供了大量预训练模型和丰富的配置选项，使得在目标检测任务中的应用和开发变得更加便捷。

<div align="center">
<img src="/assets/integration-mmdetection-intro.png">
</div>

可以通过修改MMDetection的配置文件来使用SwanLab作为实验记录工具。

## 在配置文件中指定SwanLab作为VisBackend

确保你安装了SwanLab，或者使用`pip install -U swanlab`安装最新版。

将如下内容添加到mmdetection的config文件中, 其中`init_kwargs`中填入的参数字典与`swanlab.init`的规则一致:

```python
# swanlab visualizer
custom_imports = dict(  # 引入SwanLab作为日志记录器，对于部分不支持custom_imports的项目可以直接初始化SwanlabVisBackend并加入vis_backends
    imports=["swanlab.integration.mmengine"], allow_failed_imports=False
)

vis_backends = [
    dict(
        type="SwanlabVisBackend",
        init_kwargs={ # swanlab.init 参数
            "project": "swanlab-mmengine",
            "experiment_name": "Your exp",  # 实验名称
            "description": "Note whatever you want",  # 实验的描述信息
        },
    ),
]

visualizer = dict(
    type="Visualizer",
    vis_backends=vis_backends,
    name="visualizer",
)
```

参考[快速开始](https://docs.swanlab.cn/zh/guide_cloud/general/quick-start.html)注册并[获得SwanLab的在线跟踪key](https://swanlab.cn/settings/overview)，并使用`swanlab login`完成跟踪配置。当然你也可以使用[离线看板](https://docs.swanlab.cn/zh/guide_cloud/self_host/offline-board.html)来离线查看训练结果。wanLab作为VisBackend

有关其他引入方法和更灵活的配置，可以参考[MMEngine接入SwanLab](https://docs.swanlab.cn/zh/guide_cloud/integration/integration-mmengine.html)

## 使用案例：MMDetection训练faster-rcnn

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
