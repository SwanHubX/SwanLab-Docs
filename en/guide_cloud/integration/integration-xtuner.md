# Xtuner

[XTuner](https://github.com/InternLM/xtuner) is a highly efficient, flexible, and versatile tool library for fine-tuning large models.

<div align="center">
<img src="/assets/integration-xtuner.png" width=440>
</div>

Xtuner supports adaptation with multiple open-source large models such as InternLM and Llama, and can perform tasks such as incremental pre-training, instruction fine-tuning, and tool-based instruction fine-tuning. In terms of hardware requirements, developers can train with the lowest consumer-grade graphics cards, such as Tesla T4 and A100, to achieve specific demand capabilities of large models.

<div align="center">
<img src="/assets/integration-xtuner-intro.png">
</div>

Xtuner supports online tracking using SwanLab through MMEngine. By adding a few lines of code to the configuration file, you can track and visualize metrics such as loss and memory usage.

## Visualizing and Tracking Xtuner Fine-Tuning Progress with SwanLab

Open the configuration file you want to train (for example, [qwen1_5_7b_chat_full_alpaca_e3.py](https://github.com/InternLM/xtuner/blob/main/xtuner/configs/qwen/qwen1_5/qwen1_5_7b_chat/qwen1_5_7b_chat_full_alpaca_e3.py)), find the `visualizer` parameter, and replace it with:

```python
# set visualizer
from mmengine.visualization import Visualizer
from swanlab.integration.mmengine import SwanlabVisBackend

visualizer = dict(type=Visualizer, vis_backends=[dict(type=SwanlabVisBackend)])
```

Then, run the fine-tuning command as usual to achieve SwanLab experiment tracking:

```bash
xtuner train qwen1_5_7b_chat_full_alpaca_e3.py
```

---

If you want to specify project name, experiment name, and other information as you normally would with SwanLab, you can specify them in the `init_kwargs` parameter when instantiating `SwanlabVisBackend`. You can refer to [swanlab init](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/sdk.py#L71) to see the configurable parameters.

By passing `init_kwargs` in the form of a dictionary, this parameter will eventually be passed to the `swanlab.init` method. Below is an example of specifying a project name.

```python (5)
visualizer = dict(
  type=Visualizer,
  vis_backends=[dict(
        type=SwanlabVisBackend,
        init_kwargs=dict(project='toy-example', experiment_name='Qwen'),
    )])
```

For other integration methods with the MM series and more flexible configurations, please refer to [MMEngine Integration with SwanLab](https://docs.swanlab.cn/en/guide_cloud/integration/integration-mmengine.html).