# Qwen1.5指令微调

:::info
文本分类，大语言模型，大模型微调
:::

[知乎](https://zhuanlan.zhihu.com/p/701370317) ｜[竞赛地址](https://www.modelscope.cn/datasets/huangjintao/zh_cls_fudan-news/summary) | [实验过程](https://swanlab.cn/@KashiwaByte/hf-visualization/runs/7hy42f6o66d5ibtsnlcjw/overview)

## 概述
Qwen1.5是通义千问开源模型的1.5版本，研发团队是阿里云。

指令微调是一种通过在由（指令，输出）对组成的数据集上进一步训练LLMs的过程。 其中，指令代表模型的人类指令，输出代表遵循指令的期望输出。 这个过程有助于弥合LLMs的下一个词预测目标与用户让LLMs遵循人类指令的目标之间的差距。

以Qwen-1.5作为基座大模型，通过指令微调的方式实现高准确率的文本分类是学习大语言模型微调的入门级任务。

在这个任务中我们会使用Qwen-1.5-7b模型在zh_cls_fudan_news数据集上进行指令微调任务，同时使用SwanLab进行监控和可视化。

![Qwencompetition](/assets/Qwencompetition.png)

本案例主要：

- 使用`transformers`加载模型、训练以及推理。
- 使用`datasets`下载、加载数据集。
- 使用`peft`进行微调训练，与微调后模型推理。
- 使用`accelerate`提供混合精度训练的支持。
- 使用`modelscope`在国内环境中下载Qwen大模型
- 使用`swanlab`跟踪超参数、记录指标和可视化监控整个训练周期



  
  


## 环境安装

本案例基于`Python>=3.10`，请在您的计算机上安装好Python。  
环境依赖
本文的代码测试于如下版本的开源库
```
swanlab==0.3.8
modelscope==1.14.0
transformers==4.41.2
datasets==2.18.0
peft==0.11.1
accelerate==0.30.1
```


一键安装命令：

```bash
pip install swanlab modelscope transformers datasets peft
```




## 数据集下载

本案例首先需要下载名为`zh_cls_fudan-news`的数据集，该数据集包含几千条的新闻与书籍文本、文本类型选项、文本正确类型。主要用于文本分类任务。

数据集例子如下：
```
"""
[PROMPT]Text: 第四届全国大企业足球赛复赛结束新华社郑州５月３日电（实习生田兆运）上海大隆机器厂队昨天在洛阳进行的第四届牡丹杯全国大企业足球赛复赛中，以５：４力克成都冶金实验厂队，进入前四名。沪蓉之战，双方势均力敌，９０分钟不分胜负。最后，双方互射点球，沪队才以一球优势取胜。复赛的其它３场比赛，青海山川机床铸造厂队３：０击败东道主洛阳矿山机器厂队，青岛铸造机械厂队３：１战胜石家庄第一印染厂队，武汉肉联厂队１：０险胜天津市第二冶金机械厂队。在今天进行的决定九至十二名的两场比赛中，包钢无缝钢管厂队和河南平顶山矿务局一矿队分别击败河南平顶山锦纶帘子布厂队和江苏盐城无线电总厂队。４日将进行两场半决赛，由青海山川机床铸造厂队和青岛铸造机械厂队分别与武汉肉联厂队和上海大隆机器厂队交锋。本届比赛将于６日结束。（完）
Category: Sports, Politics
Output:[OUTPUT]Sports
"""

```

该数据集的下载地址为[魔搭社区](https://modelscope.cn/datasets/huangjintao/zh_cls_fudan-news/summary)
或者
[百度云](https://pan.baidu.com/s/1a6lDSiHST-cIP2-bQlwRJQ?pwd=90j1 )，大家可以下载后放置到代码同级目录下使用。


## 完整代码
以下是完整代码的文件结构
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240608121049.png)


```python
import os
import csv
import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoModel, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model


jsonl_file = 'news_train.jsonl'

# 生成JSONL文件
messages = []


# 读取jsonl文件
with open(dataset, 'r') as file:
    for line in file:
        # 解析每一行的json数据
        data = json.loads(line)
        context = data["text"]
        catagory = data["category"]
        label = data["output"]
        message={ "instruction":"你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型","input": f'文本:{context},类型选型:{catagory}',"output":label}
        messages.append(message)

# 保存为JSONL文件
with open(jsonl_file, 'w', encoding='utf-8') as file:
    for message in messages:
        file.write(json.dumps(message, ensure_ascii=False) + '\n')





# 将jsonl文件转换为CSV文件
df = pd.read_json('./news_train.jsonl',lines = True)
ds = Dataset.from_pandas(df)


def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|im_start|>system\n你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


tokenized_id = ds.map(process_func, remove_columns=ds.column_names)




model_dir = snapshot_download('qwen/Qwen1.5-7B-Chat', cache_dir='./', revision='master')

tokenizer = AutoTokenizer.from_pretrained('./qwen/Qwen1___5-7B-Chat/', use_fast=False, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained('./qwen/Qwen1___5-7B-Chat/', device_map="auto",torch_dtype=torch.bfloat16)

model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法



config = LoraConfig(

    task_type=TaskType.CAUSAL_LM,

    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    inference_mode=False, # 训练模式

    r=8, # Lora 秩

    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理

    lora_dropout=0.1# Dropout 比例

)


model = get_peft_model(model, config)


args = TrainingArguments(
    output_dir="./output/Qwen1.5",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)




swanlab_callback = SwanLabCallback(project="hf-visualization")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)


trainer.train()
```

## 效果演示

![Qwendisplay](/assets/Qwendisplay.png)