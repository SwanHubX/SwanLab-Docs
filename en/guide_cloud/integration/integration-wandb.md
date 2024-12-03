# Weights & Biases

[Weights & Biases](https://github.com/wandb/wandb) (Wandb) is an experiment tracking, model optimization, and collaboration platform for machine learning and deep learning projects. W&B provides powerful tools to log and visualize experiment results, helping data scientists and researchers better manage and share their work.

![wandb](/assets/ig-wandb.png)

You can use `swanlab convert` to convert existing projects on Wandb into SwanLab projects.

::: info
In the current version, only scalar charts are supported for conversion.
:::

## Find Your Project, Entity, and Run ID

The `project`, `entity`, and `runid` are required for the conversion (runid is optional).  
Location of `project` and `entity`:
![alt text](/assets/ig-wandb-2.png)

Location of `runid`:

![alt text](/assets/ig-wandb-3.png)

## Method 1: Command Line Conversion

First, ensure that you are logged into Wandb in the current environment and have access to the target project.

Conversion command line:

```bash
swanlab convert -t wandb --wb-project [WANDB_PROJECT_NAME] --wb-entity [WANDB_ENTITY]
```

Supported parameters are as follows:

- `-t`: Conversion type, options include `wandb` and `tensorboard`.
- `--wb-project`: The name of the Wandb project to be converted.
- `--wb-entity`: The space name where the Wandb project is located.
- `--wb-runid`: The ID of the Wandb Run (an experiment under the project).

If `--wb-runid` is not filled in, all Runs under the specified project will be converted; if filled in, only the specified Run will be converted.

## Method 2: Conversion Within Code

```python
from swanlab.converter import WandbConverter

wb_converter = WandbConverter()
# wb_runid is optional
wb_converter.run(wb_project="WANDB_PROJECT_NAME", wb_entity="WANDB_USERNAME")
```

This method achieves the same effect as the command line conversion.