# MMPretrain

[MMPretrain](https://github.com/open-mmlab/mmpretrain) 是 [OpenMMLab](https://openmmlab.com/) 旗下的一个开源预训练模型库，专注于为计算机视觉任务提供高效、易用的预训练模型。

<div align="center">
<img src="/assets/integration-mmpretrain.jpg" width=440>
</div>

基于 PyTorch 构建，MMPretrain 旨在帮助研究人员和开发人员快速应用和评估预训练模型，从而提升下游任务的性能和效率。该库包含了多种预训练模型，如 ResNet、Vision Transformer（ViT）和 Swin Transformer 等，这些模型经过大规模数据集的训练，能够直接用于图像分类、目标检测和分割等任务。此外，MMPretrain 提供了灵活的配置系统和丰富的接口，用户可以方便地进行模型的加载、微调和评估。详细的文档和教程使得用户能够快速上手和应用，适用于学术研究和工业实践中的各种场景。通过使用 MMPretrain，用户可以显著减少模型训练时间，专注于模型优化和应用创新。

可以通过修改MMPretrain的配置文件来使用SwanLab作为实验记录工具。

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

参考[快速开始](https://docs.swanlab.cn/zh/guide_cloud/general/quick-start.html)注册并[获得SwanLab的在线跟踪key](https://swanlab.cn/settings/overview)，并使用`swanlab login`完成跟踪配置。当然你也可以使用[离线看板](https://docs.swanlab.cn/zh/guide_cloud/self_host/offline-board.html)来离线查看训练结果。wanLab作为VisBackend

有关其他引入方法和更灵活的配置，可以参考[MMEngine接入SwanLab](https://docs.swanlab.cn/zh/guide_cloud/integration/integration-mmengine.html)
