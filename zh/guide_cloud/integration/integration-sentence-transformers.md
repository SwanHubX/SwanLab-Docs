# Sentence Transformers

[Sentence Transformers](https://github.com/UKPLab/sentence-transformers)(又名SBERT)是访问、使用和训练文本和图像嵌入（Embedding）模型的Python库。

![](/assets/ig-sentence-transformers.png)

你可以使用Sentence Transformers快速进行模型训练，同时使用SwanLab进行实验跟踪与可视化。

## 1. 引入SwanLabCallback

```python
from swanlab.integration.transformers import SwanLabCallback
```

**SwanLabCallback**是适配于HuggingFace系列工具（Transformers等）的日志记录类。

**SwanLabCallback**可以定义的参数有：

- project、experiment_name、description 等与 swanlab.init 效果一致的参数, 用于SwanLab项目的初始化。
- 你也可以在外部通过`swanlab.init`创建项目，集成会将实验记录到你在外部创建的项目中。

## 2. 传入Trainer

```python (1,7,12)
from swanlab.integration.transformers import SwanLabCallback
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer

...

# 实例化SwanLabCallback
swanlab_callback = SwanLabCallback(project="hf-visualization")

trainer = SentenceTransformerTrainer(
    ...
    # 传入callbacks参数
    callbacks=[swanlab_callback],
)

trainer.train()
```

## 3.完整案例代码

```python (4,12,19)
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from swanlab.integration.transformers import SwanLabCallback

model = SentenceTransformer("bert-base-uncased")

train_dataset = load_dataset("sentence-transformers/all-nli", "pair", split="train[:10000]")
eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
mnrl_loss = MultipleNegativesRankingLoss(model)

swanlab_callback = SwanLabCallback(project="sentence-transformers", experiment_name="bert-all-nli")

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=mnrl_loss,
    callbacks=[swanlab_callback],
)

trainer.train()
```