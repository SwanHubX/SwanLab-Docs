# Modelscope Swift

[![](/assets/colab.svg)](https://colab.research.google.com/drive/1eAxKyTiLXLcQw7EOQngjV2csf5UdNQY4?usp=sharing)

[Modelscope魔搭社区](https://modelscope.cn/) 的 [Swift](https://github.com/modelscope/swift) 是一个集模型训练、微调、推理、部署于一体的框架。

SWIFT支持250+ LLM和35+ MLLM（多模态大模型）的训练、推理、评测和部署。开发者可以直接将Swift应用到自己的Research和生产环境中，实现模型训练评测到应用的完整链路。Swift除支持了PEFT提供的轻量训练方案外，也提供了一个完整的Adapters库以支持最新的训练技术，如NEFTune、LoRA+、LLaMA-PRO等，这个适配器库可以脱离训练脚本直接使用在自己的自定流程中。

![alt text](/assets/ig-swift.png)

你可以使用Swift快速进行模型训练，同时使用SwanLab进行实验跟踪与可视化。

## 1.引入SwanLabCallback

因为Swift的`trainer`集成自transformers，所以可以直接使用swanlab与huggingface集成的`SwanLabCallback`：

```python
from swanlab.integration.huggingface import SwanLabCallback
```

SwanLabCallback可以定义的参数有：

- project、experiment_name、description 等与 swanlab.init 效果一致的参数, 用于SwanLab项目的初始化。
你也可以在外部通过`swanlab.init`创建项目，集成会将实验记录到你在外部创建的项目中。

## 2.引入Trainer

```python
from swanlab.integration.huggingface import SwanLabCallback
from swift import Seq2SeqTrainer, Seq2SeqTrainingArguments

···

#实例化SwanLabCallback
swanlab_callback = SwanLabCallback(project="swift-visualization")

trainer = Seq2SeqTrainer(
    ...
    callbacks=[swanlab_callback],
    )

trainer.train()
```

## 3.完整案例代码

> Lora微调一个Qwen2-0.5B模型

```python
from swift import Seq2SeqTrainer, Seq2SeqTrainingArguments
from modelscope import MsDataset, AutoTokenizer
from modelscope import AutoModelForCausalLM
from swift import Swift, LoraConfig
from swift.llm import get_template, TemplateType
import torch
from swanlab.integration.huggingface import SwanLabCallback

# 拉起模型
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


## 4.GUI演示

超参数查看：

![alt text](/assets/ig-swift-2.png)

指标查看：

![alt text](/assets/ig-swift-3.png)

