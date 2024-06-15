# MMSegmentation

[MMSegmentation](https://github.com/open-mmlab/mmengine) 是一个由 [OpenMMLab](https://openmmlab.com/) 社区开发的深度学习训练框架，基于 PyTorch 构建，旨在为研究人员和开发人员提供便捷高效的图像分割解决方案。

<div align="center">
<img src="/assets/integration-mmsegmentation.png" width=440>
</div>

该工具箱采用模块化设计，提供多种预训练模型如 U-Net、DeepLabV3 和 PSPNet 等，支持语义分割、实例分割和全景分割任务。MMSegmentation 内置强大的数据处理功能和多种分割性能评价指标，如 mIoU 和 Dice 系数，能够全面评估模型性能。其灵活的配置系统允许用户快速进行实验配置和参数调整。

<div align="center">
<img src="/assets/integration-mmsegmentation-demo.gif">
</div>

MMSegmentation 提供详细的文档和示例，帮助用户快速上手，并支持分布式训练和模型加速推理。该工具箱广泛应用于医学图像分割、遥感图像分割和自动驾驶等领域。

可以通过修改MMSegmentation的配置文件来使用SwanLab作为实验记录工具。

## 在配置文件中指定

确保你安装了SwanLab，或者使用`pip install -U swanlab`安装最新版。

将如下内容添加到所使用的mmsegmentation的config文件中, 其中`init_kwargs`中填入的参数字典与`swanlab.init`的规则一致:

```python
...
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
...
```

> 有关其他引入方法和更灵活的配置，可以参考[MMEngine接入SwanLab](https://docs.swanlab.cn/zh/guide_cloud/integration/integration-mmengine.html)
