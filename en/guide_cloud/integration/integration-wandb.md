# Weights & Biases

Weights & Biases (Wandb) is a platform for experiment tracking, model optimization, and collaboration in machine learning and deep learning projects. W&B provides powerful tools to log and visualize experimental results, helping data scientists and researchers better manage and share their work.

![wandb](/assets/ig-wandb.png)  

:::warning Synchronization Tutorials for Other Tools  

- [TensorBoard](/guide_cloud/integration/integration-tensorboard.md)  
- [MLFlow](/guide_cloud/integration/integration-mlflow.md)  
:::  

**You can sync Wandb projects to SwanLab in two ways:**  

1. **Live Synchronization**: If your current project uses Wandb for experiment tracking, you can use the `swanlab.sync_wandb()` command to log metrics to SwanLab simultaneously while running the training script.  
2. **Convert Existing Projects**: If you want to replicate a Wandb project on SwanLab, you can use `swanlab convert` to migrate an existing Wandb project to SwanLab.  

::: info  
The current version only supports converting scalar charts.  
:::  

[[toc]]  

## 1. Live Synchronization  

### 1.1 Add the `sync_wandb` Command  

Add the `swanlab.sync_wandb()` command anywhere in your code before `wandb.init()` to synchronize Wandb metrics to SwanLab during training.  

```python  
import swanlab  

swanlab.sync_wandb()  

...  

wandb.init()  
```  

With this implementation, `wandb.init()` will simultaneously initialize SwanLab, using the same `project`, `name`, and `config` parameters from `wandb.init()`. Therefore, you don’t need to manually initialize SwanLab.  

:::info  

**`sync_wandb` supports two parameters:**  

- `mode`: SwanLab logging mode, supporting `cloud`, `local`, and `disabled`.  
- `wandb_run`: If set to **False**, data will not be uploaded to Wandb (equivalent to `wandb.init(mode="offline")`).  

:::  

### 1.2 Alternative Implementation  

Another approach is to manually initialize SwanLab first before running Wandb code.  

```python  
import swanlab  

swanlab.init(...)  
swanlab.sync_wandb()  

...  

wandb.init()  
```  

In this implementation, the project name, experiment name, and configuration will follow the `project`, `experiment_name`, and `config` parameters from `swanlab.init()`. Subsequent `wandb.init()` parameters for `project` and `name` will be ignored, while `config` will update `swanlab.config`.  

### 1.3 Test Code  

```python  
import wandb  
import random  
import swanlab  

swanlab.sync_wandb()  
# swanlab.init(project="sync_wandb")  

wandb.init(  
  project="test",  
  config={"a": 1, "b": 2},  
  name="test",  
)  

epochs = 10  
offset = random.random() / 5  
for epoch in range(2, epochs):  
  acc = 1 - 2 ** -epoch - random.random() / epoch - offset  
  loss = 2 ** -epoch + random.random() / epoch + offset  

  wandb.log({"acc": acc, "loss": loss})  
```  

![alt text](/assets/ig-wandb-4.png)  

## 2. Convert Existing Projects  

### 2.1 Locate Your `project`, `entity`, and `runid` on wandb.ai  

The conversion requires `project`, `entity`, and optionally `runid`.  
Locations of `project` and `entity`:  
![alt text](/assets/ig-wandb-2.png)  

Location of `runid`:  
![alt text](/assets/ig-wandb-3.png)  

### 2.2 Method 1: Command-Line Conversion  

First, ensure you are logged into Wandb and have access to the target project.  

Conversion command:  

```bash  
swanlab convert -t wandb --wb-project [WANDB_PROJECT_NAME] --wb-entity [WANDB_ENTITY]  
```  

Supported parameters:  

- `-t`: Conversion type (`wandb` or `tensorboard`).  
- `-p`: SwanLab project name.  
- `-w`: SwanLab workspace name.  
- `--mode`: (str) Logging mode (default: `"cloud"`), options: `["cloud", "local", "offline", "disabled"]`.  
- `-l`: Log directory path.  
- `--wb-project`: Wandb project name to convert.  
- `--wb-entity`: Wandb entity (username/team) where the project resides.  
- `--wb-runid`: Wandb Run ID (specific experiment under the project).  

If `--wb-runid` is omitted, all Runs under the project will be converted. If specified, only the selected Run will be converted.  

---  

**Asynchronous Conversion (Download Data Locally First, Then Upload to SwanLab)**  

1. Download data locally:  

```bash  
swanlab convert --mode 'offline' -t wandb --wb-project [WANDB_PROJECT_NAME] --wb-entity [WANDB_ENTITY]  
```  

2. Upload to SwanLab:  

```bash  
swanlab sync [LOG_DIRECTORY_PATH]  
```  

[SwanLab Sync Documentation](/en/api/cli-swanlab-sync.md)  

### 2.3 Method 2: In-Code Conversion  

```python  
from swanlab.converter import WandbConverter  

wb_converter = WandbConverter()  
# wb_runid is optional  
wb_converter.run(wb_project="WANDB_PROJECT_NAME", wb_entity="WANDB_USERNAME")  
```  

This achieves the same result as command-line conversion.  

`WandbConverter` parameters:  

- `project`: SwanLab project name.  
- `workspace`: SwanLab workspace name.  
- `mode`: (str) Logging mode (default: `"cloud"`), options: `["cloud", "local", "offline", "disabled"]`.  
- `logdir`: Log directory path.  

`WandbConverter.run` parameters:  

- `wb_project`: Wandb project name.  
- `wb_entity`: Wandb entity (username/team).  
- `wb_runid`: Wandb Run ID (specific experiment).  

**Asynchronous Conversion (Download Data Locally First, Then Upload to SwanLab)**  

1. Download data locally:  

```python  
from swanlab.converter import WandbConverter  

wb_converter = WandbConverter(mode="offline")  
# wb_runid is optional  
wb_converter.run(wb_project="WANDB_PROJECT_NAME", wb_entity="WANDB_USERNAME")  
```  

2. Upload to SwanLab:  

```bash  
swanlab sync [LOG_DIRECTORY_PATH]  
```  

[SwanLab Sync Documentation](/en/api/cli-swanlab-sync.md)