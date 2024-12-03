# Modelscope Swift

[Modelscope](https://modelscope.cn/) [Swift](https://github.com/modelscope/swift) is a framework that integrates model training, fine-tuning, inference, and deployment.

SWIFT supports the training, inference, evaluation, and deployment of over 250 LLMs and 35+ MLLMs (multimodal large models). Developers can directly apply Swift to their research and production environments, achieving a complete pipeline from model training and evaluation to application. In addition to supporting lightweight training solutions provided by PEFT, Swift also offers a comprehensive Adapters library to support the latest training techniques such as NEFTune, LoRA+, LLaMA-PRO, etc. This adapter library can be used directly in custom workflows without relying on training scripts.

![alt text](/assets/ig-swift.png)

You can use Swift to quickly conduct model training, while using SwanLab for experiment tracking and visualization.

## 1. Introducing SwanLabCallback

Since Swift's `trainer` is integrated from transformers, you can directly use the `SwanLabCallback` integrated with swanlab and huggingface:

```python
from swanlab.integration.huggingface import SwanLabCallback
```

The parameters that can be defined for SwanLabCallback include:

- Parameters such as `project`, `experiment_name`, `description`, which have the same effect as `swanlab.init`, are used for the initialization of the SwanLab project.
You can also create a project externally via `swanlab.init`, and the integration will record the experiment to the project you created externally.

## 2. Introducing Trainer

```python
from swanlab.integration.huggingface import SwanLabCallback
from swift import Seq2SeqTrainer, Seq2SeqTrainingArguments

···

# Instantiate SwanLabCallback
swanlab_callback = SwanLabCallback(project="swift-visualization")

trainer = Seq2SeqTrainer(
    ...
    callbacks=[swanlab_callback],
    )

trainer.train()
```

## 3. Complete Example Code

> Fine-tuning a Qwen2-0.5B model with Lora

```python
from swift import Seq2SeqTrainer, Seq2SeqTrainingArguments
from modelscope import MsDataset, AutoTokenizer
from modelscope import AutoModelForCausalLM
from swift import Swift, LoraConfig
from swift.llm import get_template, TemplateType
import torch
from swanlab.integration.huggingface import SwanLabCallback

# Load the model
model = AutoModelForCausalLM.from_pretrained('qwen/Qwen2-0.5B', torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)

lora_config = LoraConfig(
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=32,
                lora_dropout=0.05)

model = Swift.prepare_model(model, lora_config)
tokenizer = AutoTokenizer.from_pretrained('qwen/Qwen2-0.5B', trust_remote_code=True)

dataset = MsDataset.load('AI-ModelScope/alpaca-gpt4-data-en', split='train')
template = get_template(TemplateType.chatglm3, tokenizer, max_length=1024)

def encode(example):
    inst, inp, output = example['instruction'], example.get('input', None), example['output']
    if output is None:
        return {}
    if inp is None or len(inp) == 0:
        q = inst
    else:
        q = f'{inst}\n{inp}'
    example, kwargs = template.encode({'query': q, 'response': output})
    return example

dataset = dataset.map(encode).filter(lambda e: e.get('input_ids'))
dataset = dataset.train_test_split(test_size=0.001)

train_dataset, val_dataset = dataset['train'], dataset['test']

train_args = Seq2SeqTrainingArguments(
    output_dir='output',
    report_to="none",
    learning_rate=1e-4,
    num_train_epochs=2,
    eval_steps=500,
    save_steps=500,
    evaluation_strategy='steps',
    save_strategy='steps',
    dataloader_num_workers=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    logging_steps=10,
)

swanlab_callback = SwanLabCallback(
    project="swift-visualization",
    experiment_name="qwen2-0.5b",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=train_args,
    data_collator=template.data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    callbacks=[swanlab_callback],
    )

trainer.train()
```

## 4. GUI Demonstration

Viewing Hyperparameters:

![alt text](/assets/ig-swift-2.png)

Viewing Metrics:

![alt text](/assets/ig-swift-3.png)