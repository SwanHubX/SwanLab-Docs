# Torchtune

[Torchtune](https://github.com/pytorch/torchtune) is a PyTorch library for easily writing, fine-tuning, and experimenting with LLMs.

You can use `torchtune` to quickly fine-tune LLMs while using SwanLab for experiment tracking and visualization.

## 1. Modify the Configuration File to Introduce SwanLabLogger

We will use `torchtune` to fine-tune Google's `gemma-2b` model as an example.

When fine-tuning a model with `torchtune`, the trainer needs to prepare a configuration file first. For example, fine-tuning the Gemma-2b model with QLoRA: [2B_qlora_single_device.yaml](https://github.com/pytorch/torchtune/blob/main/recipes/configs/gemma/2B_qlora_single_device.yaml).

After downloading, edit this configuration file. Find the following code snippet in the file:

```yaml
# Logging
metric_logger:
  _component_: torchtune.utils.metric_logging.DiskLogger
  log_dir: ${output_dir}
```

Replace this code snippet with:

```yaml
# Logging
metric_logger:
  _component_: swanlab.integration.torchtune.SwanLabLogger
  project: "gemma-fintune"
  experiment_name: "gemma-2b"
  log_dir: ${output_dir}
```

Here, `_component_` corresponds to `swanlab.integration.torchtune.SwanLabLogger`, which is a logging class adapted for PyTorch torchtune. The parameters like `project`, `experiment_name`, etc., are the parameters passed when creating the SwanLab project. The supported parameters are consistent with the rules of [swanlab.init](/zh/api/py-init.html).

## 2. Start Training

```bash
tune run lora_finetune_single_device --config 2B_qlora_single_device.yaml
```