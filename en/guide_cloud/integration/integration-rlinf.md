# RLinf

[RLinf](https://github.com/RLinf/RLinf) is an open-source framework for Reinforcement Learning training, focused on supporting the reinforcement learning training of large language models (LLMs) and vision-language models (VLAs). RLinf provides advanced features such as unified programming interfaces, flexible execution modes, automatic scheduling, and elastic communication, supporting various reinforcement learning algorithms like PPO and GRPO.

![rlinf-logo](./rlinf/logo.svg)

RLinf supports real-time experiment tracking, allowing you to stream loss curves, accuracy, GPU utilization, and any custom metrics to **SwanLab**.

You can use RLinf to quickly conduct reinforcement learning model training while using SwanLab for experiment tracking and visualization.

> RLinf Official Documentation: https://rlinf.readthedocs.io/zh-cn/latest/

## Quick Integration with SwanLab

RLinf supports enabling SwanLab as a logging backend in the YAML configuration file. Simply add `"swanlab"` to the `runner.logger.logger_backends` in the configuration file.

Add the following configuration to your YAML configuration file:

```yaml
runner:
  task_type: math  # or "embodied", etc.
  logger:
    log_path: ${runner.output_dir}/${runner.experiment_name}
    project_name: rlinf
    experiment_name: ${runner.experiment_name}
    logger_backends: ["swanlab"]  # [!code ++]
  experiment_name: grpo-1.5b
  output_dir: ./logs
```

After running the training, RLinf will create a subdirectory for the enabled SwanLab backend:

```
logs/grpo-1.5b/
├── checkpoints/
├── converted_ckpts/
├── log/
├── swanlab/            # SwanLab runtime directory
```

## Setting Project Name and Experiment Name

You can set the project name and experiment name by configuring `project_name` and `experiment_name` in the YAML configuration file:

```yaml
runner:
  project_name: rlinf
  experiment_name: grpo-1.5b
```

## Setting Log File Save Location

You can set the log file save location by configuring `log_path` in the YAML configuration file:

```yaml
runner:
  log_path: ./logs
```