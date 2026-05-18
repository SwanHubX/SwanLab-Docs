# 从零预训练一个自己的大模型

[![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/assets/badge1.svg)](https://swanlab.cn/@ShaohonChen/WikiLLM/overview)

大语言模型（Large Language Model，简称LLM），指使用大量文本数据训练的深度学习模型，可以生成自然语言文本或理解语言文本的含义。

![llm](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/assets/examples/pretrain_llm/llm.png)

虽然网上有大量关于transformer理论、大语言模型微调的教程。但是少有关于预训练的解释。本文则从如何自己实战预训练一个大语言模型的角度，使用wiki数据集进行一个简单的从零预训练工作，并附上使用swanlab launch白嫖显卡的方法

* 本教程完整代码：[GitHub](https://github.com/ShaohonChen/transformers_from_scratch)

* 实验记录：[SwanLab](https://swanlab.cn/@ShaohonChen/WikiLLM/overview)

* 数据集下载：[百度网盘（j8ee）](https://pan.baidu.com/s/1p5F52bRlnpSY7F78q0hz7A?pwd=j8ee)，[huggingface](https://huggingface.co/datasets/fjcanyue/wikipedia-zh-cn)

## 安装环境

首先，项目推荐使用python3.10。需要安装的python包如下：

```txt
swanlab
transformers
datasets
accelerate
```

使用如下命令一键安装：

```bash
pip install swanlab transformers datasets accelerate modelscope
```

## 下载数据集

本教程使用的是中文wiki数据，理论上预训练数据集种类越丰富、数据量越大越好，后续会增加别的数据集。

![dataset](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/assets/examples/pretrain_llm/dataset.png)

huggingface链接：[wikipedia-zh-cn](https://huggingface.co/datasets/fjcanyue/wikipedia-zh-cn)

百度网盘下载地址：[百度网盘（j8ee）](https://pan.baidu.com/s/1p5F52bRlnpSY7F78q0hz7A?pwd=j8ee)

下载`wikipedia-zh-cn-20240820.json`文件后放到项目目录下`./WIKI_CN/`文件夹中

该数据集文件约1.99G大，共有1.44M条数据。虽然数据集中包含文章标题，但是实际上在预训练阶段用不上。正文片段参考：

```txt
数学是研究数量、结构以及空间等概念及其变化的一门学科，属于形式科学的一种。数学利用抽象化和逻辑推理，从计数、计算、量度、对物体形状及运动的观察发展而成。数学家们拓展这些概念...
```

使用[🤗Huggingface Datasets](https://huggingface.co/docs/datasets/index)加载数据集的代码如下：

```python
from datasets import load_dataset

ds = load_dataset("fjcanyue/wikipedia-zh-cn")
```

如果使用百度网盘下载的json文件，可以通过如下代码加载

```python
raw_datasets = datasets.load_dataset(
    "json", data_files="data/wikipedia-zh-cn-20240820.json"
)

raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1, seed=2333)
print("dataset info")
print(raw_datasets)
```

## 构建自己的大语言模型

本教程使用[🤗huggingface transformers](https://huggingface.co/docs/transformers/index)构建自己的大模型。

因为目标是训练一个中文大模型。因此我们参考[通义千问2](https://qwen.readthedocs.io/zh-cn/latest/run_locally/mlx-lm.html)的tokenize和模型架构，仅仅做一些简单的更改让模型更小更好训练。

因为国内无法直接访问到huggingface，推荐使用modelscope先把模型配置文件和checkpoint下载到本地，运行如下代码

```python
import modelscope

modelscope.AutoConfig.from_pretrained("Qwen/Qwen2-0.5B").save_pretrained(
    "Qwen2-0.5B"
)
modelscope.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B").save_pretrained(
    "Qwen2-0.5B"
)
```

配置参数，并修改模型注意力头数量、模型层数和中间层大小，把模型控制到大概120M参数左右（跟GPT2接近）。

```python
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("./Qwen2-0.5B")   # 这里使用qwen2的tokenzier
config = transformers.AutoConfig.from_pretrained(
        "./Qwen2-0.5B",
        vocab_size=len(tokenizer),
        hidden_size=512,
        intermediate_size=2048,
        num_attention_heads=8,
        num_hidden_layers=12,
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
print("Model Config:")
print(config)
```

使用transformers库初始化模型

```python
model = transformers.Qwen2ForCausalLM(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"Model Size: {model_size/1000**2:.1f}M parameters")
```

## 设置训练参数

设置预训练超参数：

```python
args = transformers.TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=24,  # 每个GPU的训练batch数
    per_device_eval_batch_size=24,  # 每个GPU的测试batch数
    eval_strategy="steps",
    eval_steps=5_000,
    logging_steps=500,
    gradient_accumulation_steps=12,  # 梯度累计总数
    num_train_epochs=2, # 训练epoch数
    weight_decay=0.1,
    warmup_steps=1_000,
    optim="adamw_torch",  # 优化器使用adamw
    lr_scheduler_type="cosine",  # 学习率衰减策略
    learning_rate=5e-4,  # 基础学习率，
    save_steps=5_000,
    save_total_limit=10,
    bf16=True,  # 开启bf16训练, 对于Amper架构以下的显卡建议替换为fp16=True
)
print("Train Args:")
print(args)
```

## 初始化训练+使用swanlab进行记录

使用transformers自带的train开始训练，并且引入swanlab作为可视化日志记录

```python
from swanlab.integration.transformers import SwanLabCallback
trainer = transformers.Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    callbacks=[SwanLabCallback()],
)
trainer.train()
```

如果是第一次使用SwanLab，需要登陆SwanLab官网[https://swanlab.cn/](https://swanlab.cn/)，注册，并且在如下位置找到和复制自己的key。

![findkey](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/assets/examples/pretrain_llm/findkey.png)

接下来在命令行中输入

```sh
swanlab login
```

会看到提示输入key

![login](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/assets/examples/pretrain_llm/login.png)

按照提示将key粘贴进去（注意key是不会显示到终端当中的）就可以完成配置，完成效果如下：

![login2](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/assets/examples/pretrain_llm/login2.png)

## 完整代码

项目目录结构：

```txt
|---data\
|------wikipedia-zh-cn-20240820.json    # 数据集放在data文件夹中
|--- pretrain.py
```

`pretrain.py`代码如下：

```python
import datasets
import transformers
import swanlab
from swanlab.integration.transformers import SwanLabCallback
import modelscope

def main():
    # using swanlab to save log
    swanlab.init("WikiLLM")

    # load dataset
    raw_datasets = datasets.load_dataset(
        "json", data_files="/data/WIKI_CN/wikipedia-zh-cn-20240820.json"
    )

    raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1, seed=2333)
    print("dataset info")
    print(raw_datasets)

    # load tokenizers
    # 因为国内无法直接访问HuggingFace，因此使用魔搭将模型的配置文件和Tokenizer下载下来
    modelscope.AutoConfig.from_pretrained("Qwen/Qwen2-0.5B").save_pretrained(
        "Qwen2-0.5B"
    )
    modelscope.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B").save_pretrained(
        "Qwen2-0.5B"
    )
    context_length = 512  # use a small context length
    # tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "./Qwen2-0.5B"
    )  # download from local

    # preprocess dataset
    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    print("tokenize dataset info")
    print(tokenized_datasets)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # prepare a model from scratch
    config = transformers.AutoConfig.from_pretrained(
        "./Qwen2-0.5B",
        vocab_size=len(tokenizer),
        hidden_size=512,
        intermediate_size=2048,
        num_attention_heads=8,
        num_hidden_layers=12,
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = transformers.Qwen2ForCausalLM(config)
    model_size = sum(t.numel() for t in model.parameters())
    print("Model Config:")
    print(config)
    print(f"Model Size: {model_size/1000**2:.1f}M parameters")

    # train
    args = transformers.TrainingArguments(
        output_dir="WikiLLM",
        per_device_train_batch_size=32,  # 每个GPU的训练batch数
        per_device_eval_batch_size=32,  # 每个GPU的测试batch数
        eval_strategy="steps",
        eval_steps=5_00,
        logging_steps=50,
        gradient_accumulation_steps=8,  # 梯度累计总数
        num_train_epochs=2,  # 训练epoch数
        weight_decay=0.1,
        warmup_steps=2_00,
        optim="adamw_torch",  # 优化器使用adamw
        lr_scheduler_type="cosine",  # 学习率衰减策略
        learning_rate=5e-4,  # 基础学习率，
        save_steps=5_00,
        save_total_limit=10,
        bf16=True,  # 开启bf16训练, 对于Amper架构以下的显卡建议替换为fp16=True
    )
    print("Train Args:")
    print(args)
    # enjoy training
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        callbacks=[SwanLabCallback()],
    )
    trainer.train()

    # save model
    model.save_pretrained("./WikiLLM/Weight")  # 保存模型的路径

    # generate
    pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("GENERATE:", pipe("人工智能", num_return_sequences=1)[0]["generated_text"])
    prompts = ["牛顿", "北京市", "亚洲历史"]
    examples = []
    for i in range(3):
        # 根据提示词生成数据
        text = pipe(prompts[i], num_return_sequences=1)[0]["generated_text"]
        text = swanlab.Text(text)
        examples.append(text)
    swanlab.log({"Generate": examples})


if __name__ == "__main__":
    main()

```

## 训练结果演示

运行如下命令

```
python pretrain.py
```

可以看到如下训练日志。由于训练时间较长，推荐使用tmux将训练任务hold住

![terminal](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/assets/examples/pretrain_llm/terminal.png)

可以在[SwanLab](https://swanlab.cn)中查看最终的训练结果：

![log](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/assets/examples/pretrain_llm/log.png)

<!-- 并且能够看到一些最终生成的案例：

![sample]() -->

## 使用训练好的模型进行推理

以“人工智能”为开头生成内容的代码如下：

```python
pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
print("GENERATE:", pipe("人工智能", num_return_sequences=1)[0]["generated_text"])
```

推理效果如下：

（模型训练ing，可以在[https://swanlab.cn/@ShaohonChen/WikiLLM/overview](https://swanlab.cn/@ShaohonChen/WikiLLM/overview)实时查看训练进展和推理效果）
<!-- ![result]() -->


## 参考链接

* 本教程完整代码:[GitHub](https://github.com/ShaohonChen/transformers_from_scratch)

* 实验记录：[SwanLab](https://swanlab.cn/@ShaohonChen/WikiLLM/overview)

* 数据集下载：[百度网盘（j8ee）](https://pan.baidu.com/s/1p5F52bRlnpSY7F78q0hz7A?pwd=j8ee)，[huggingface](https://huggingface.co/datasets/fjcanyue/wikipedia-zh-cn)
