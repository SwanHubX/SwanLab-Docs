# Sentence Transformers

[Sentence Transformers](https://github.com/UKPLab/sentence-transformers) (also known as SBERT) is a Python library for accessing, using, and training text and image embedding models.

![](/assets/ig-sentence-transformers.png)

You can use Sentence Transformers to quickly train models while using SwanLab for experiment tracking and visualization.

## 1. Import SwanLabCallback

```python
from swanlab.integration.transformers import SwanLabCallback
```

**SwanLabCallback** is a logging class adapted for HuggingFace series tools (such as Transformers).

**SwanLabCallback** can define parameters such as:

- `project`, `experiment_name`, `description`, and other parameters consistent with `swanlab.init`, used for initializing the SwanLab project.
- You can also create the project externally via `swanlab.init`, and the integration will log the experiment to the project you created externally.

## 2. Pass to Trainer

```python (1,7,12)
from swanlab.integration.transformers import SwanLabCallback
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer

...

# Instantiate SwanLabCallback
swanlab_callback = SwanLabCallback(project="hf-visualization")

trainer = SentenceTransformerTrainer(
    ...
    # Pass the callbacks parameter
    callbacks=[swanlab_callback],
)

trainer.train()
```

## 3. Complete Example Code

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