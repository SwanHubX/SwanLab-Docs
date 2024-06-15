# Xtuner

[XTuner](https://github.com/InternLM/xtuner) 是一个高效、灵活、全能的轻量化大模型微调工具库。

<div align="center">
<img src="/assets/integration-xtuner.png" width=440>
</div>

Xtuner支持与书生·浦语（InternLM）、Llama等多款开源大模型的适配，可执行增量预训练、指令微调、工具类指令微调等任务类型。硬件要求上，在Tesla T4、A100等传统数据中心之外，开发者最低使用消费级显卡便可进行训练，实现大模型特定需求能力。

<div align="center">
<img src="/assets/integration-xtuner-intro.png">
</div>

Xtuner 支持通过 MMEngine 使用 SwanLab 进行在线跟踪，只需在配置文件中添加几行代码，就可以跟踪和可视化损失、显存占用等指标。

## 使用SwanLab可视化跟踪Xtuner微调进展

打开要训练的配置文件（比如[qwen1_5_7b_chat_full_alpaca_e3.py](https://github.com/InternLM/xtuner/blob/main/xtuner/configs/qwen/qwen1_5/qwen1_5_7b_chat/qwen1_5_7b_chat_full_alpaca_e3.py)）），找到`visualizer`参数的位置，将它替换成：

```python
# set visualizer
from mmengine.visualization import Visualizer
from swanlab.integration.mmengine import SwanlabVisBackend

visualizer = dict(type=Visualizer, vis_backends=[dict(type=SwanlabVisBackend)])
```

然后照样运行微调命令，即可实现SwanLab实验跟踪：

```bash
xtuner train qwen1_5_7b_chat_full_alpaca_e3.py
```

---

如果希望像平常使用SwanLab那样指定项目名、实验名等信息，可以在实例化`SwanlabVisBackend`时在`init_kwargs`参数中指定，可以参考 [swanlab init](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/sdk.py#L71) 查看可配置的参数。

通过以字典的形式传入`init_kwargs`，该参数最终会传给 `swanlab.init` 方法，下面举了个指定项目名称的案例。

```python (5)
visualizer = dict(
  type=Visualizer,
  vis_backends=[dict(
        type=SwanlabVisBackend,
        init_kwargs=dict(project='toy-example', experiment_name='Qwen'),
    )])
```

有关MM系列的其他引入方法和更灵活的配置，可以参考[MMEngine接入SwanLab](https://docs.swanlab.cn/zh/guide_cloud/integration/integration-mmengine.html)。