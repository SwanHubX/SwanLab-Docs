# Torchtune

[Torchtune](https://github.com/pytorch/torchtune)是一个 PyTorch 库，用于轻松编写、微调和试验LLMs。

你可以使用`torchtune`快速进行LLM微调，同时使用SwanLab进行实验跟踪与可视化。

## 1. 修改配置文件，引入SwanLabLogger

我们以使用`torchtune`微调Google的`gemma-2b`模型为例。

torchtune在微调一个模型时，需要训练者先准备一个配置文件，如用QLoRA微调Gemma-2b模型：[2B_qlora_single_device.yaml](https://github.com/pytorch/torchtune/blob/main/recipes/configs/gemma/2B_qlora_single_device.yaml)。

下载后，编辑这个配置文件。我们在文件中找到下面的代码段：

```yaml
# Logging
metric_logger:
  _component_: torchtune.utils.metric_logging.DiskLogger
  log_dir: ${output_dir}
```

将该代码段替换为：

```yaml
# Logging
metric_logger:
  _component_: swanlab.integration.torchtune.SwanLabLogger
  project: "gemma-fintune"
  experiment_name: "gemma-2b"
  log_dir: ${output_dir}
```

其中，`_component_`对应的`swanlab.integration.torchtune.SwanLabLogger`是适配于PyTorch torchtune的日志记录类。而`project`、`experiment_name`等则是创建SwanLab项目传入的参数，支持传入的参数与[swanlab.init](http://localhost:5173/zh/api/py-init.html)规则一致。


## 2. 开始训练

```bash
tune run lora_finetune_single_device --config 2B_qlora_single_device.yaml
```