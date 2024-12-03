# MMSegmentation

[MMSegmentation](https://github.com/open-mmlab/mmengine) is a deep learning training framework developed by the [OpenMMLab](https://openmmlab.com/) community, built on PyTorch, aiming to provide researchers and developers with convenient and efficient image segmentation solutions.

<div align="center">
<img src="/assets/integration-mmsegmentation.png" width=440>
</div>

The toolbox adopts a modular design, providing various pre-trained models such as U-Net, DeepLabV3, and PSPNet, supporting semantic segmentation, instance segmentation, and panoptic segmentation tasks. MMSegmentation features powerful data processing capabilities and multiple segmentation performance evaluation metrics, such as mIoU and Dice coefficient, to comprehensively evaluate model performance. Its flexible configuration system allows users to quickly configure experiments and adjust parameters.

<div align="center">
<img src="/assets/integration-mmsegmentation-demo.gif">
</div>

MMSegmentation offers detailed documentation and examples to help users get started quickly and supports distributed training and model accelerated inference. The toolbox is widely used in medical image segmentation, remote sensing image segmentation, and autonomous driving, among other fields.

You can modify the MMSegmentation configuration file to use SwanLab as an experiment logging tool.

## Specify in the Configuration File

Ensure you have installed SwanLab, or use `pip install -U swanlab` to install the latest version.

Add the following content to the MMSegmentation config file you are using, where the dictionary of parameters in `init_kwargs` follows the rules of `swanlab.init`:

```python
...
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
...
```

> For other import methods and more flexible configurations, refer to [MMEngine Integration with SwanLab](https://docs.swanlab.cn/zh/guide_cloud/integration/integration-mmengine.html)