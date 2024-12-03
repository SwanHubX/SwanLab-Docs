# MMPretrain

[MMPretrain](https://github.com/open-mmlab/mmpretrain) is an open-source pre-training model library under [OpenMMLab](https://openmmlab.com/), focusing on providing efficient and easy-to-use pre-trained models for computer vision tasks.

<div align="center">
<img src="/assets/integration-mmpretrain.jpg" width=440>
</div>

Built on PyTorch, MMPretrain aims to help researchers and developers quickly apply and evaluate pre-trained models, thereby improving the performance and efficiency of downstream tasks. The library includes various pre-trained models such as ResNet, Vision Transformer (ViT), and Swin Transformer, which have been trained on large datasets and can be directly used for tasks such as image classification, object detection, and segmentation. Additionally, MMPretrain provides a flexible configuration system and rich interfaces, allowing users to easily load, fine-tune, and evaluate models. Detailed documentation and tutorials enable users to quickly get started and apply, suitable for various scenarios in academic research and industrial practice. By using MMPretrain, users can significantly reduce model training time and focus on model optimization and application innovation.

You can modify the MMPretrain configuration file to use SwanLab as an experiment logging tool.

## Specify in the Configuration File

Ensure you have installed SwanLab, or use `pip install -U swanlab` to install the latest version.

Add the following content to the MMPretrain config file you are using, where the dictionary of parameters in `init_kwargs` follows the rules of `swanlab.init`:

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