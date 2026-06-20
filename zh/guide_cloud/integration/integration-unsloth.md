# Unsloth

[微信公众号文章](https://mp.weixin.qq.com/s/re7R7WhTYNuiDj0fSwAnWQ)

[Unsloth](https://github.com/unslothai/unsloth) 是一个用于加速 LLM（大型语言模型）微调的轻量级库 。它与 Hugging Face 生态系统完全兼容，包括 Hub、transformers 和 PEFT 。

![logo](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/zh/guide_cloud/integration/unsloth/logo.png)

你可以使用Unsloth与Tranformers或TRL结合加速LLM模型训练，同时使用SwanLab进行实验跟踪与可视化。

## 1. 引入SwanLabCallback

```python
from swanlab.integration.transformers import SwanLabCallback
```

SwanLabCallback是适配于Transformers的日志记录类。

SwanLabCallback可以定义的参数有：

- project、experiment_name、description 等与 swanlab.init 效果一致的参数, 用于SwanLab项目的初始化。
- 你也可以在外部通过swanlab.init创建项目，集成会将实验记录到你在外部创建的项目中。

## 2. 传入Trainer

```python {1,7,12}
from swanlab.integration.transformers import SwanLabCallback
from trl import GRPOTrainer

...

# 实例化SwanLabCallback
swanlab_callback = SwanLabCallback(project="unsloth-example")

trainer = GRPOTrainer(
    ...
    # 传入callbacks参数
    callbacks=[swanlab_callback],
)

trainer.train()
```

## 3. 与Unsloth结合的案例模板

```python
from swanlab.integration.transformers import SwanLabCallback
from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)  # 对 TRL 进行补丁处理
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

...

model, tokenizer = FastLanguageModel.from_pretrained(
...
)

# PEFT 模型
model = FastLanguageModel.get_peft_model(
...
)

# 实例化SwanLabCallback
swanlab_callback = SwanLabCallback(
  project="trl_integration",
  experiment_name="qwen2.5-sft",
  description="测试swanlab和trl的集成",
  config={"framework": "🤗TRL"},
)

# 定义GRPOTrainer
trainer = GRPOTrainer(
    ...
    # 传入callbacks参数
    callbacks=[swanlab_callback],
)

#开启训练！
trainer.train()
```
