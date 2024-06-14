# MMEngine

[MMEngine](https://github.com/open-mmlab/mmengine) 是一个由 [OpenMMLab](https://openmmlab.com/) 社区开发的深度学习训练框架，专为深度学习研究和开发而设计。MMEngine 提供了一种高效、灵活且用户友好的方式来构建、训练和测试深度学习模型，尤其是在计算机视觉领域。它的目标是简化研究人员和开发者在深度学习项目中的工作流程，并提高其开发效率。

<div align="center">
<img src="/assets/integration-mmengine.jpeg" width=440>
</div>

MMEngine 为 OpenMMLab 算法库实现了下一代训练架构，为 OpenMMLab 中的 30 多个算法库提供了统一的执行基础。其核心组件包括训练引擎、评估引擎和模块管理。

SwanLab将专为MMEngine设计的`SwanlabVisBackend`集成到MMEngine中，可用于记录训练、评估指标、记录实验配置、记录图像等。

## MMEngine系列框架兼容性说明

理论上使用mmengine的框架都可以使用如下方法引入SwanLab，包括[mmdetection](https://docs.swanlab.cn/zh/guide_cloud/integration/integration-mmdetection.html)，[mmsegmentation](https://docs.swanlab.cn/zh/guide_cloud/integration/integration-mmsegmentation.html)等，以及[自己基于mmengine实现的训练框架](https://mmengine.readthedocs.io/zh-cn/latest/get_started/15_minutes.html)，可以在[OpenMMLab官方GitHub账号](https://github.com/open-mmlab)下查看有哪些优秀框架，不过[Xtuner](https://github.com/InternLM/xtuner)项目由于其没有完全兼容mmengine需要做一点点改动，可以前往[SwanLab的Xtuner集成](https://docs.swanlab.cn/zh/guide_cloud/integration/integration-xtuner.html)查看如何在Xtuner中使用SwanLab。mmengine有两种引入SwanLab进行实验可视化跟踪的方法。

## 使用方法一：训练脚本传入visualizer，开始训练

:::info
可以参考[mmengine15分钟教程](https://mmengine.readthedocs.io/zh-cn/latest/get_started/15_minutes.html)将自己的训练代码适配mmengine
:::

确保你安装了SwanLab，或者使用`pip install -U swanlab`安装最新版。

如果你按照官方案例使用了mmengine作为你的训练框架。只需在训练脚本中进行如下改动：
1.
2. 在初始化`visualizer`时加入SwanLabVis
3. 初始化`runner`传入`visualizer`即可：

```python
from mmengine.visualization import Visualizer
from mmengine.runner import Runner

from swanlab.integration.mmengine import SwanlabVisBackend
...
# 初始化SwanLab
swanlab_vis_backend = SwanlabVisBackend(init_kwargs={})# init args can be found in https://docs.swanlab.cn/zh/guide_cloud/integration/integration-mmengine.html
# 初始化mmegine的Visulizer，并引入SwanLab作为Visual Backend
visualizer = Visualizer(
    vis_backends=swanlab_vis_backend
)  

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

如果希望像平常使用swanlab那样指定实验名等信息，可以在实例化SwanlabVisBackend时在init_kwargs中指定参数，具体参考[init api](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/sdk.py#L71)，不过不像使用`swanlab.init`那样直接作为参数传入，而是需要构建字典，下面列举了两者的不同：

直接使用`swanlab.init`

```pyhton
run = swanlab.init(
    project="cat-dog-classification",
    experiment_name="Resnet50",
    description="我的第一个人工智能实验",
)
```

使用`SwanlabVisBackend`，需要以字典的形式传入`init`的参数

```python
swanlab_vis_backend = SwanlabVisBackend(
    init_kwargs={
        "project": "cat-dog-classification",
        "experiment_name": "Resnet50",
        "description": "我的第一个人工智能实验",
    }
)
```

参考[快速开始](https://docs.swanlab.cn/zh/guide_cloud/general/quick-start.html)注册并[获得SwanLab的在线跟踪key](https://swanlab.cn/settings/overview)，并使用`swanlab login`完成跟踪配置。当然你也可以使用[离线看板](https://docs.swanlab.cn/zh/guide_cloud/self_host/offline-board.html)来离线查看训练结果。wanLab作为VisBackend

## 使用方法二：config文件引入SwanlabVisBackend

:::info
此方法对于大多数基于mmengine的训练框架都是适用的
:::

将如下内容添加到mm系列框架的任意config文件中, 其中`init_kwargs`中填入的参数字典与`swanlab.init`的规则一致:

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

可以使用如下代码测试config文件是否能够成功引入SwanLab，将上面的config文件保存为`my_swanlab_config.py`，创建一个`test_config.py`写入如下代码并运行：

```python
from mmengine.config import Config
import mmengine

print(mmengine.__version__)
cfg = Config.fromfile(
    "my_swanlab_config.py"
)

from mmengine.registry import VISUALIZERS

custom_vis = VISUALIZERS.build(cfg.visualizer)
print(custom_vis)

```

如果看到终端打印蕾丝如下信息，则表示成功引入了swanlab作为Visual Backend：

```txt
MMEngine Version: 0.10.4
SwanLab Version: 0.3.1
<mmengine.visualization.visualizer.Visualizer object at 0x7f7cf15b1e20>
```

## 3.案例：MMEngine训练ResNet-50

:::info 参考MMEngine官方15分钟上手教程
[15 分钟上手 MMENGINE](https://mmengine.readthedocs.io/zh-cn/latest/get_started/15_minutes.html)
:::

按照[MMEngine官方教程](https://mmengine.readthedocs.io/zh-cn/latest/get_started/installation.html)安装MMEngine。

这里将安装环境的命令抄录下来，强烈建议按照官方文档安装，以环境为python3.11，CUDA12.1为例。

```sh
# with cuda12.1 or you can find torch version you want at pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim
mim install mmengine
pip install swanlab
```

使用如下代码构建ResNet-50网络并引入Cifar10数据集开始训练

```python
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.utils.data import DataLoader

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from swanlab.integration.mmengine import SwanlabVisBackend


class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == "loss":
            return {"loss": F.cross_entropy(x, labels)}
        elif mode == "predict":
            return x, labels


class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append(
            {
                "batch_size": len(gt),
                "correct": (score.argmax(dim=1) == gt).sum().cpu(),
            }
        )

    def compute_metrics(self, results):
        total_correct = sum(item["correct"] for item in results)
        total_size = sum(item["batch_size"] for item in results)
        return dict(accuracy=100 * total_correct / total_size)


norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(
    batch_size=32,
    shuffle=True,
    dataset=torchvision.datasets.CIFAR10(
        "data/cifar10",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**norm_cfg),
            ]
        ),
    ),
)

val_dataloader = DataLoader(
    batch_size=32,
    shuffle=False,
    dataset=torchvision.datasets.CIFAR10(
        "data/cifar10",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(**norm_cfg)]
        ),
    ),
)

visualizer = Visualizer(
    vis_backends=SwanlabVisBackend(init_kwargs={})
)  # init args can be found in https://docs.swanlab.cn/zh/guide_cloud/integration/integration-mmengine.html

runner = Runner(
    model=MMResNet50(),
    work_dir="./work_dir",
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
    visualizer=visualizer,
)
runner.train()

```

可以在[公开训练图表](https://swanlab.cn/@ShaohonChen/cifar10_with_resnet50/runs/f8znz8vj06huv6rm7j5a8/chart)查看到上脚本的训练结果。

<div align="center">
<img src="/assets/integration-mmegine-train.png" width=600>
</div>
