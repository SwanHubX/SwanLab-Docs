# swanlab convert

```bash
swanlab convert [OPTIONS]
```

| 选项 | 描述 |
| --- | --- |
| `-t`, `--type` | 选择转换类型，可选`tensorboard`、`wandb`、`mlflow`，默认为`tensorboard`。 |
| `-p`, `--project` | 设置转换创建的SwanLab项目名，默认为None。 |
| `-w`, `--workspace` | 设置SwanLab项目所在空间，默认为None。 |
| `-l`, `--logdir` | 设置SwanLab项目的日志文件保存路径，默认为None。 |
| `--cloud` | 设置SwanLab项目是否将日志上传到云端，默认为True。 |
| `--tb-logdir` | 需要转换的Tensorboard日志文件路径(tfevent) |
| `--wb-project` | 需要转换的Wandb项目名 |
| `--wb-entity` | 需要转换的Wandb项目所在实体 |
| `--wb-runid` | 需要转换的Wandb Run的id |
| `--mlflow-uri` | 需要转换的MLFlow项目URI |
| `--mlflow-exp` | 需要转换的MLFlow实验ID |

## 介绍

将其他日志工具的内容转换为SwanLab项目。  
支持转换的工具包括：`Tensorboard`、`Weights & Biases`、`MLFlow`。

## 使用案例

### Tensorboard

[集成-Tensorboard](/guide_cloud/integration/integration-tensorboard.md)

### Weights & Biases

[集成-Weights & Biases](/guide_cloud/integration/integration-wandb.md)

### MLFlow

[集成-MLFlow](/guide_cloud/integration/integration-mlflow.md)