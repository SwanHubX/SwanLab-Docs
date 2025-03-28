# Unsloth

[WeChat Official Account Article](https://mp.weixin.qq.com/s/re7R7WhTYNuiDj0fSwAnWQ)

[Unsloth](https://github.com/unslothai/unsloth) is a lightweight library designed to accelerate the fine-tuning of LLMs (Large Language Models). It is fully compatible with the Hugging Face ecosystem, including the Hub, transformers, and PEFT.

![logo](./unsloth/logo.png)

You can use Unsloth in conjunction with Transformers or TRL to accelerate LLM model training, while utilizing SwanLab for experiment tracking and visualization.

## 1. Introducing SwanLabCallback

```python
from swanlab.integration.transformers import SwanLabCallback
```

SwanLabCallback is a logging class adapted for Transformers.

The parameters that can be defined for SwanLabCallback include:

- `project`, `experiment_name`, `description`, and other parameters consistent with `swanlab.init`, used for initializing the SwanLab project.
- You can also create a project externally via `swanlab.init`, and the integration will log the experiment to the project you created externally.

## 2. Passing to Trainer

```python {1,7,12}
from swanlab.integration.transformers import SwanLabCallback
from trl import GRPOTrainer

...

# Instantiate SwanLabCallback
swanlab_callback = SwanLabCallback(project="unsloth-example")

trainer = GRPOTrainer(
    ...
    # Pass the callbacks parameter
    callbacks=[swanlab_callback],
)

trainer.train()
```

## 3. Example Template Combining Unsloth

```python
from swanlab.integration.transformers import SwanLabCallback
from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)  # Patch TRL
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

...

model, tokenizer = FastLanguageModel.from_pretrained(
...
) 

# PEFT Model
model = FastLanguageModel.get_peft_model(
...
)

# Instantiate SwanLabCallback
swanlab_callback = SwanLabCallback(
  project="trl_integration",
  experiment_name="qwen2.5-sft",
  description="Testing the integration of swanlab and trl",
  config={"framework": "🤗TRL"},
)

# Define GRPOTrainer
trainer = GRPOTrainer(
    ...
    # Pass the callbacks parameter
    callbacks=[swanlab_callback],
)

# Start training!
trainer.train()
```