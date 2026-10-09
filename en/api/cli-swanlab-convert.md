# swanlab convert

```bash
swanlab convert [OPTIONS]
```

| Option              | Description                                                                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-t`, `--type`      | Select the conversion type, options include `tensorboard`, `wandb`, `mlflow`, `wandb-local`, default is `tensorboard`.                                  |
| `-p`, `--project`   | Set the SwanLab project name for the conversion, default is None.                                                                                       |
| `-w`, `--workspace` | Set the workspace where the SwanLab project is located, default is None.                                                                                |
| `--mode`            | Set the SwanLab logging mode, options include `online`, `local`, `offline`, `disabled`, default is `online`.                                            |
| `-l`, `--logdir`    | Set the log file save path for the SwanLab project, default is None.                                                                                    |
| `--tb-log-dir`      | Path to the TensorBoard log files (tfevent) to be converted.                                                                                            |
| `--tb-types`        | The types of TensorBoard logs to convert, options include `scalar`, `image`, `audio`, `text`, default is `scalar`; separate multiple types with commas. |
| `--wb-project`      | Name of the W&B project to be converted.                                                                                                                |
| `--wb-entity`       | Entity where the W&B project to be converted is located.                                                                                                |
| `--wb-runid`        | ID of the W&B Run to be converted.                                                                                                                      |
| `--wb-dir`          | Directory where the W&B local log files are stored, default is `./wandb`.                                                                               |
| `--wb-run-dir`      | The specific W&B local run directory name; if omitted, all runs under `--wb-dir` will be converted.                                                     |
| `--mlflow-url`      | The tracking URL of the MLflow server, default is `http://127.0.0.1:5000`.                                                                              |
| `--mlflow-exp`      | The name or ID of the MLflow experiment to be converted (required).                                                                                     |
| `--mlflow-runid`    | ID of a specific MLflow Run to be converted.                                                                                                            |
| `--resume`          | Resume mode: uses the source Run ID as the SwanLab Run ID for resuming; must be used together with `--wb-runid`.                                        |

## Introduction

Convert content from other logging tools into SwanLab projects.  
Supported tools for conversion include: `TensorBoard`, `Weights & Biases`, `MLflow`.

## Usage Examples

### TensorBoard

[Integration - TensorBoard](../guide_cloud/integration/integration-tensorboard.md)

### Weights & Biases

[Integration - Weights & Biases](../guide_cloud/integration/integration-wandb.md)

### MLflow

[Integration - MLflow](../guide_cloud/integration/integration-mlflow.md)
