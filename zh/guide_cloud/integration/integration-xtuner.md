# Xtuner

[XTuner](https://github.com/InternLM/xtuner) 是一个高效、灵活、全能的轻量化大模型微调工具库。

<div align="center">
<img src="/assets/integration-xtuner.png" width=440>
</div>

Xtuner支持与书生·浦语（InternLM）、Llama等多款开源大模型的适配，可执行增量预训练、指令微调、工具类指令微调等任务类型。硬件要求上，在Tesla T4、A100等传统数据中心之外，开发者最低使用消费级显卡便可进行训练，实现大模型特定需求能力。

<div align="center">
<img src="/assets/integration-xtuner-intro.png">
</div>

## 使用SwanLab可视化跟踪Xtuner微调进展

Xtuner 支持通过 MMEngine 使用 SwanLab 进行在线跟踪，只需在 config 中添加一行代码，就可以跟踪和可视化损失、显存占用等指标。

在`swanlab.integration.mmengine`中引入`SwanlabVisBackend`，接下来设置 config 中的 visualizer 字段，并将 vis_backends 设置为 SwanLab：

```python
# set visualizer
- visualizer = None
+ from mmengine.visualization import TensorboardVisBackend
+ from swanlab.integration.mmengine import SwanlabVisBackend
+ visualizer = dict(type=Visualizer, vis_backends=[dict(type=SwanlabVisBackend)])
```

如果希望像平常使用swanlab那样指定实验名等信息，可以在实例化SwanlabVisBackend时在init_kwargs中指定参数，可以参考 [SwanLab init API](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/sdk.py#L71) 查看可配置的参数。通过以字典的形式传入init_kwargs，该参数最终会传给 swanlab.init 方法，下面举了个指定项目名称的案例。

```python
# set visualizer
- visualizer = None
+ from mmengine.visualization import Visualizer
+ from swanlab.integration.mmengine import SwanlabVisBackend
+ visualizer = dict(
+   type=Visualizer,
+   vis_backends=[
+       dict(type=SwanlabVisBackend, init_kwargs=dict(project='toy-example'))])
```

参考[快速开始](https://docs.swanlab.cn/zh/guide_cloud/general/quick-start.html)注册并[获得SwanLab的在线跟踪key](https://swanlab.cn/settings/overview)，并使用`swanlab login`完成跟踪配置。当然你也可以使用[离线看板](https://docs.swanlab.cn/zh/guide_cloud/self_host/offline-board.html)来离线查看训练结果。wanLab作为VisBackend

启动实验后，既可在[swanlab.cn](https://swanlab.cn/)中查看训练的可视化结果
