# MMEngine

[MMEngine](https://github.com/open-mmlab/mmengine) 是一个由 [OpenMMLab](https://openmmlab.com/) 社区开发的深度学习训练框架，专为深度学习研究和开发而设计。MMEngine 提供了一种高效、灵活且用户友好的方式来构建、训练和测试深度学习模型，尤其是在计算机视觉领域。它的目标是简化研究人员和开发者在深度学习项目中的工作流程，并提高其开发效率。

<div align="center">
<img src="/assets/integration-mmengine.jpeg" width=440>
</div>

MMEngine 为 OpenMMLab 算法库实现了下一代训练架构，为 OpenMMLab 中的 30 多个算法库提供了统一的执行基础。其核心组件包括训练引擎、评估引擎和模块管理。

SwanLab将专为MMEngine设计的`SwanlabVisBackend`集成到MMEngine中，可用于记录训练、评估指标、记录实验配置、记录图像等。

## 1. 引入SwanlabVisBackend

将如下内容添加到mm系列框架的任意config文件中, 其中`init_kwargs`中填入的参数字典与`swanlab.init`的规则一致:

```python
custom_imports = dict(imports=["swanlab.integration.mmengine"], allow_failed_imports=False)

vis_backends = [
    dict(
        type="SwanlabVisBackend",
        save_dir="runs/swanlab",
        init_kwargs={
            "project": "swanlab-mmengine",
        },
    ),
]

visualizer = dict(
    type="Visualizer",
    vis_backends=vis_backends,
)
```

## 2.开始训练

接下来，只需初始化一个`runner`，传入`visualizer`即可：

```python (10)
from mmengine.runner import Runner

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