# 📖大模型训练课程

## 📕简介

本课程以**大模型实践为主**，涵盖了LLM训练的整个流程，包括预训练、微调以及强化学习等内容，下面是我们整理出来的LLM的文章列表，方便读者查阅👇。

## 📝文章列表

### 第一章 传统模型

| 教程名称    | 描述 | 地址 | 代码 |
|:------------------- |:-----|:-----:|:-----:|
| 1.1 Bert文本分类 |微调BERT模型，实现IMDB电影评论进行情感分类任务|[教程](../01-traditionmodel/1.bert/README.md) | 🚧|
| 1.2 LSTM股票预测 |LSTM是一种特殊的RNN，教程使用Google的股票标注数据训练LSTM模型|[教程](../01-traditionmodel/2.lstm/README.md) | 🚧|
| 1.3 RNN教程|分为上下两部，分别对应RNN原理以及RNN模型构建实战|[原理](../01-traditionmodel/3.rnn/rnn_tutorial_1.md)、[实战](../01-traditionmodel/3.rnn/rnn_tutorial_2.md)| [代码](https://github.com/ShaohonChen/tutorial_with_rnn)|


### 第二章 预训练

| 教程名称    | 描述 | 地址 | 代码 |
|:------------------- |:-----|:-----:|:-----:|
| 2.1 LLM预训练 |使用wiki数据集进行一个简单的从零预训练工作，并附上使用swanlab launch白嫖显卡的方法|[教程](../02-pretrain/1.qwen-pretrain/README.md) | [代码](https://github.com/ShaohonChen/transformers_from_scratch)|


### 第三章 微调

| 教程名称    | 描述 | 地址 | 代码 |
|:------------------- |:-----|:--------:|:-----:|
| 3.1 Qwen文本分类 |在这个任务中我们会使用Qwen-1.5-7b模型在zh_cls_fudan_news数据集上进行指令微调任务| [教程](../03-sft/1.text_classification/README.md) | 🚧|
| 3.2 Qwen命名体识别 |使用 Qwen2-1.5b-Instruct 模型在中文NER数据集上做指令微调训练|[教程](../03-sft/2.ner/README.md) | [代码](https://github.com/Zeyi-Lin/LLM-Finetune)|
| 3.3 GLM4指令微调 |使用指令遵从微调GLM4模型，为了便于实现，减少代码量，本文使用了🤗HuggingFace的TRL框架实现|[教程](../03-sft/3.glm4-instruct/README.md) | [代码](https://github.com/SwanHubX/glm4-finetune)|
| 3.4 Qwen3医学模型微调 |以Qwen3作为基座大模型，通过全参数微调的方式，实现医学专业领域聊天，甚至支持DeepSeek R1 / QwQ式的带推理过程的对话|[教程](../03-sft/4.qwen3-medical-finetune/README.md) | [代码](https://github.com/Zeyi-Lin/Qwen3-Medical-SFT)|
| 3.5 Mac上微调Qwen3模型 |本篇教程基于MLX-LM(Mac)教程给大家介绍下如何使用Macbook微调Qwen3模型|[教程](../03-sft/5.mac-qwen3-finetune/README.md) | [代码](https://github.com/ShaohonChen/Finetune_Qwen3_on_MacBook)|
| 3.6 llamafactory框架QLoRA微调 |用llama-factory框架来实现大模型的lora和qlora的教程，并且对比分析运行的结果|[原理](../03-sft/6.llamafactory-finetune/lora1.md)、[实战](../03-sft/6.llamafactory-finetune/lora2.md) |  🚧|
| 3.7 deepseek模型lora微调 |对deepseek模型的全流程lora多轮对话微调实战|[教程](../03-sft/7.deepseek-lora/README.md) | [代码](https://github.com/828Tina/deepseek-llm-7B-chat-lora-ft)|
| 3.8 其他框架微调 |除了基础的Transformers框架，还有些国内其他框架可以实现模型微调|[paddle](../03-sft/8.other_frameworks/paddlenlp_finetune.md)、[ms-swift](../03-sft/8.other_frameworks/ms-swift.md) | [代码](https://github.com/Akiyamaice/PaddleNLP-FINETUNE)、[代码](https://github.com/828Tina/qwen3-ft-swift)|

### 第四章 强化学习

| 教程名称    | 描述 | 地址 | 代码 |
|:------------------- |:-----|:--------:|:-----:|
| 4.1 Qwen复现R1-Zero |对deepseek-r1-zero进行复现实验，简单介绍了从r1原理到代码实现，再到结果观测的整个过程|[教程](../04-reinforce/2.qwen_grpo/README.md) | 🚧|
| 4.2 数独游戏GRPO训练 |使用GRPO的方法，用lora来做微调，分别在GPU、NPU的AI训练卡上训练数独游戏任务|[教程](../04-reinforce/3.sudoku_grpo/README.md) | [代码](https://github.com/828Tina/sudoku_trl_grpo?tab=readme-ov-file)|

### 第五章 评估

| 教程名称    | 描述 | 地址 | 代码 |
|:------------------- |:-----|:--------:|:-----:|
| 5.1 EvalScope使用 |基于魔搭社区的官方模型评估和基准测试框架EvalScope做微调后模型的评估测试| [教程](../05-eval/1.evalscope/README.md) | [代码](https://github.com/828Tina/evalscope_qwen3_eval/tree/main)|

### 第六章 视觉大模型

| 教程名称    | 描述 | 地址 | 代码 |
|:------------------- |:-----|:--------:|:-----:|
| 6.1 Qwen2-VL微调 |Qwen2-VL-2B-Instruct模型在COCO2014图像描述上进行Lora微调训练| [教程](../06-multillm/1.qwen_vl_coco/README.md) | [代码](https://github.com/Zeyi-Lin/LLM-Finetune/tree/main/qwen2_vl)|
| 6.2 Qwen3-smVL模型拼接微调 |使用沐曦GPU芯片，把Qwen3与SmolVLM2直接拼接后微调|[教程](../06-multillm/2.qwen3_smolvlm_muxi/README.md) | [代码](https://github.com/ShaohonChen/Qwen3-SmVL)|
| 6.3 Qwen2.5-VL目标检测微调 |对Qwen2.5-VL模型进行单目标检测任务微调|[教程](../06-multillm/4.grounding/README.md) | [代码](https://github.com/828Tina/textvqa_grounding_task_qwen2.5-vl-ft)|

### 第七章 音频大模型

| 教程名称    | 描述 | 地址 | 代码 |
|:------------------- |:-----|:--------:|:-----:|
| 7.1 用CosyVoice2实现派蒙语音的微调 |CosyVoice2模型来SFT微调原神中的派蒙语音| [教程](../07-audio/1.cosyvoice-sft/README.md) |[代码](https://github.com/828Tina/cosyvoice-paimon-sft/tree/main)|


### 第八章 扩散模型

| 教程名称    | 描述 | 地址 | 代码 |
|:------------------- |:-----|:--------:|:-----:|
| 8.1 LLaDA预训练和微调 |LLaDA是文本类的扩散模型，本节实现其预训练和微调| [教程](../08-diffusion/1.llada/README.md) |[代码]()|

