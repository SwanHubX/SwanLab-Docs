# Specforge

[SpecForge](https://github.com/sgl-project/SpecForge) 是由 SGLang 团队开发的生态系统项目。它是一个用于训练 speculative decoding models 的框架，以便开发者可以平滑地将它们移植到 SGLang 服务框架，从而加快推理速度。

![specforge](./specforge/logo.png)

- SpecForge官方文档：https://docs.sglang.io/SpecForge/basic_usage/training.html

你可以使用SpecForge快速进行模型训练，同时使用SwanLab进行实验跟踪与可视化。


## 将Specforge集成SwanLab

> 参考文档：https://docs.sglang.io/SpecForge/basic_usage/training.html#experiment-tracking

只需要在SpecForge提供的 shell 脚本的命令行中添加 `--report-to` 参数，并传入 `swanlab` 即可。

```bash {13}
torchrun \
    --standalone \
    --nproc_per_node 8 \
    scripts/prepare_hidden_states.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-aux-hidden-states \
    --data-path ./cache/dataset/sharegpt_train.jsonl \
    --output-path ./cache/hidden_states/sharegpt_train_Llama-3.1-8B-Instruct \
    --chat-template llama3 \
    --max-length 4096 \
    --tp-size 1 \
    --batch-size 32 \
    --report-to swanlab
```

