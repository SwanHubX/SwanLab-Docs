# MLFlow

[MLFlow](https://github.com/mlflow/mlflow) is an open-source machine learning lifecycle management platform created and maintained by Databricks. It aims to assist data scientists and machine learning engineers in managing the entire lifecycle of machine learning projects more efficiently, including experiment tracking, model management, model deployment, and collaboration. MLflow is designed to be modular and can integrate with any machine learning library, framework, or tool.

![mlflow](./mlflow/logo.png)

:::warning Synchronization Tutorials for Other Tools

• [TensorBoard](/zh/guide_cloud/integration/integration-tensorboard.md)
• [Weights & Biases](/zh/guide_cloud/integration/integration-wandb.md)
:::

**You can convert projects from MLflow to SwanLab:**

::: info
The current version only supports the conversion of scalar charts.
:::

[[toc]]

## 1. Preparation

**Required: MLflow Service URL**

First, note down the **URL** of the MLflow service, such as `http://127.0.0.1:5000`.

> If the MLflow service is not yet started, you need to start it using the `mlflow ui` command and note down the URL.

**Optional: Experiment ID**

If you only want to convert a specific experiment, note down the experiment ID as shown in the image below.

![](./mlflow/ui-1.png)

## 2. Method 1: Command Line Conversion

Conversion Command:

```bash
swanlab convert -t mlflow --mlflow-url <MLFLOW_URL> --mlflow-exp <MLFLOW_EXPERIMENT_ID>
```

Supported parameters:

• `-t`: Conversion type, options include wandb, tensorboard, and mlflow.
• `-p`: SwanLab project name.
• `-w`: SwanLab workspace name.
• `--cloud`: (bool) Whether the upload mode is "cloud", default is True.
• `-l`: Log directory path.
• `--mlflow-url`: URL of the MLflow service.
• `--mlflow-exp`: MLflow experiment ID.

If `--mlflow-exp` is not specified, all experiments under the specified project will be converted; if specified, only the designated experiment group will be converted.

## 3. Method 2: Conversion Within Code

```python
from swanlab.converter import MLFLowConverter

mlflow_converter = MLFLowConverter(project="mlflow_converter")
# mlflow_exp is optional
mlflow_converter.run(tracking_uri="http://127.0.0.1:5000", experiment="1")
```

The effect is consistent with command line conversion.

Parameters supported by `MLFLowConverter`:

• `project`: SwanLab project name.
• `workspace`: SwanLab workspace name.
• `cloud`: (bool) Whether the upload mode is "cloud", default is True.
• `logdir`: ID of the wandb Run (a specific experiment under the project).