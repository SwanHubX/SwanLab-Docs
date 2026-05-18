# swanlab convert

```bash
swanlab convert [OPTIONS]
```

| Option | Description |
| --- | --- |
| `-t`, `--type` | Select the conversion type, options include `tensorboard`, `wandb`, default is `tensorboard`. |
| `-p`, `--project` | Set the SwanLab project name for the conversion, default is None. |
| `-w`, `--workspace` | Set the workspace where the SwanLab project is located, default is None. |
| `-l`, `--logdir` | Set the log file save path for the SwanLab project, default is None. |
| `--cloud` | Set whether the SwanLab project logs are uploaded to the cloud, default is True. |
| `--tb-logdir` | Path to the Tensorboard log files (tfevent) to be converted. |
| `--wb-project` | Name of the Wandb project to be converted. |
| `--wb-entity` | Entity where the Wandb project to be converted is located. |
| `--wb-runid` | ID of the Wandb Run to be converted. |

## Introduction

Convert content from other logging tools into SwanLab projects.  
Supported tools for conversion include: `Tensorboard`, `Weights & Biases`.

## Usage Examples

### Tensorboard

[Integration - Tensorboard](/en/guide_cloud/integration/integration-tensorboard.md)

### Weights & Biases

[Integration - Weights & Biases](/en/guide_cloud/integration/integration-wandb.md)