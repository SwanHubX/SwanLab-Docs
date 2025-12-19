# Specforge  

[SpecForge](https://github.com/sgl-project/SpecForge) is an ecosystem project developed by the SGLang team. It is a framework for training speculative decoding models, enabling developers to seamlessly integrate them into the SGLang service framework to accelerate inference speed.  

![specforge](./specforge/logo.png)  

- SpecForge Official Documentation: https://docs.sglang.io/SpecForge/basic_usage/training.html  

You can use SpecForge for rapid model training while employing SwanLab for experiment tracking and visualization.  

## Integrating Specforge with SwanLab  

> Reference Documentation: https://docs.sglang.io/SpecForge/basic_usage/training.html#experiment-tracking  

Simply add the `--report-to` parameter to the command line of the shell script provided by SpecForge and pass `swanlab` as its value.  

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