# RLinf

[RLinf](https://github.com/RLinf-Team/RLinf) 是一个用于强化学习（Reinforcement Learning）训练的开源框架，专注于支持大规模语言模型（LLM）和视觉语言模型（VLA）的强化学习训练。RLinf 提供了统一的编程接口、灵活的执行模式、自动调度和弹性通信等高级特性，支持多种强化学习算法如 PPO、GRPO 等。

![rlinf-logo](./rlinf/logo.svg)

RLinf 支持实时实验追踪，可以将损失曲线、准确率、GPU 利用率以及任意自定义指标流式传输到 **SwanLab**。

你可以使用 RLinf 快速进行强化学习模型训练，同时使用 SwanLab 进行实验跟踪与可视化。

> RLinf 官方文档：https://rlinf.readthedocs.io/zh-cn/latest/

## 快速集成SwanLab

RLinf 支持在 YAML 配置文件中启用 SwanLab 作为日志记录后端。只需要在配置文件的 `runner.logger.logger_backends` 中添加 `"swanlab"` 即可。

在你的 YAML 配置文件中，添加以下配置：

```yaml
runner:
  task_type: math  # 或 "embodied" 等任务类型
  logger:
    log_path: ${runner.output_dir}/${runner.experiment_name}
    project_name: rlinf
    experiment_name: ${runner.experiment_name}
    logger_backends: ["swanlab"]  # [!code ++]
  experiment_name: grpo-1.5b
  output_dir: ./logs
```

运行训练后，RLinf 会为启用的 SwanLab 后端创建一个子目录：

```
logs/grpo-1.5b/
├── checkpoints/
├── converted_ckpts/
├── log/
├── swanlab/            # SwanLab 运行目录
```

## 设置项目名与实验名

你可以通过在 YAML 配置文件中设置 `project_name` 和 `experiment_name` 来设置项目名和实验名：

```yaml
runner:
  project_name: rlinf
  experiment_name: grpo-1.5b
```

## 设置日志文件保存位置

你可以通过在 YAML 配置文件中设置 `log_path` 来设置日志文件保存位置：

```yaml
runner:
  log_path: ./logs
```