# swanlab convert

```bash
swanlab convert [OPTIONS]
```

| 选项                | 描述                                                                                                            |
| ------------------- | --------------------------------------------------------------------------------------------------------------- |
| `-t`, `--type`      | 选择转换类型，可选`tensorboard`、`wandb`、`mlflow`、`wandb-local`，默认为`tensorboard`。                        |
| `-p`, `--project`   | 设置转换创建的SwanLab项目名，默认为None。                                                                       |
| `-w`, `--workspace` | 设置SwanLab项目所在空间，默认为None。                                                                           |
| `--mode`            | 设置SwanLab的记录模式，可选`online`、`local`、`offline`、`disabled`，默认为`online`。                           |
| `-l`, `--logdir`    | 设置SwanLab项目的日志文件保存路径，默认为None。                                                                 |
| `--tb-log-dir`      | 需要转换的TensorBoard日志文件路径(tfevent)。                                                                    |
| `--tb-types`        | 需要转换的TensorBoard日志类型，可选`scalar`、`image`、`audio`、`text`，默认为`scalar`，多个类型用英文逗号分隔。 |
| `--wb-project`      | 需要转换的W&B项目名。                                                                                           |
| `--wb-entity`       | 需要转换的W&B项目所在实体。                                                                                     |
| `--wb-runid`        | 需要转换的W&B Run的id。                                                                                         |
| `--wb-dir`          | 需要转换的W&B本地日志目录，默认为`./wandb`。                                                                    |
| `--wb-run-dir`      | 指定的W&B本地run目录名，不填写则转换`--wb-dir`下的全部run。                                                     |
| `--mlflow-url`      | MLflow服务的tracking url，默认为`http://127.0.0.1:5000`。                                                       |
| `--mlflow-exp`      | 需要转换的MLflow实验的name或id（必填）。                                                                        |
| `--mlflow-runid`    | 需要转换的MLflow Run的id。                                                                                      |
| `--resume`          | 续传模式，将源Run的id作为SwanLab Run的id进行续传，需要与`--wb-runid`一起使用。                                  |

## 介绍

将其他日志工具的内容转换为SwanLab项目。  
支持转换的工具包括：`TensorBoard`、`Weights & Biases`、`MLflow`。

## 使用案例

### TensorBoard

[集成-TensorBoard](../guide_cloud/integration/integration-tensorboard.md)

### Weights & Biases

[集成-Weights & Biases](../guide_cloud/integration/integration-wandb.md)

### MLflow

[集成-MLflow](../guide_cloud/integration/integration-mlflow.md)
