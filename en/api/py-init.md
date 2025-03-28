# swanlab.init

```python
init(
    project: str = None,
    workspace: str = None,
    experiment_name: str = None,
    description: str = None,
    config: Union[dict, str] = None,
    logdir: str = None,
    mode: str = "cloud",
    load: str = None,
    public: bool = None,
    callbacks: list = None,
    **kwargs,
)
```

| Parameter         | Description |
|-------------------|-------------|
| project           | (str) The name of the project. If not specified, the name of the current working directory will be used. |
| workspace         | (str) The workspace. By default, experiments are synchronized to your personal space. If you want to upload to an organization, specify the organization's username. |
| experiment_name   | (str) The name of the experiment. If not specified, it will default to a format like "swan-1" (animal name + sequence number). |
| description       | (str) A description of the experiment. If not specified, it defaults to None. |
| config            | (dict, str) Configuration for the experiment. You can record hyperparameters and other information here. Supports passing a configuration file path (yaml or json). |
| logdir            | (str) The path to store offline dashboard log files. Defaults to `swanlog`. |
| mode              | (str) Sets the mode for creating SwanLab experiments. Options are "cloud", "local", or "disabled". Default is "cloud".<br>`cloud`: Uploads the experiment to the cloud (public or private deployment).<br>`local`: Does not upload to the cloud but records experiment information locally.<br>`disabled`: Neither uploads nor records. |
| load              | (str) The path to a configuration file to load. Supports yaml and json files. |
| public            | (bool) Sets the visibility of the SwanLab project created directly via code. Default is False (private). |
| callbacks         | (list) Sets experiment callback functions. Supports subclasses of `swankit.callback.SwanKitCallback`. |
| name              | (str) Same effect as `experiment_name`. Lower priority than `experiment_name`. |
| notes             | (str) Same effect as `description`. Lower priority than `description`. |

## Introduction

• In machine learning workflows, you can add `swandb.init()` at the beginning of training and testing scripts. SwanLab will track every step of the machine learning process.

• `swanlab.init()` spawns a new background process to log data to the experiment. By default, it also synchronizes the data to swanlab.cn, allowing you to view real-time visualizations online.

• Before using `swanlab.log()` to record data, you must call `swanlab.init()`:

```python
import swanlab

swanlab.init()
swanlab.log({"loss": 0.1846})
```

• Calling `swanlab.init()` returns an object of type `SwanLabRun`, which can also perform `log` operations:

```python
import swanlab

run = swanlab.init()
run.log({"loss": 0.1846})
```

• At the end of the script, `swanlab.finish` will be automatically called to conclude the SwanLab experiment. However, if `swanlab.init()` is called from a subprocess (e.g., in a Jupyter notebook), you must explicitly call `swanlab.finish` at the end of the subprocess.

```python
import swanlab

swanlab.init()
swanlab.finish()
```

## Additional Usage

### Setting Project, Experiment Name, and Description

```python
swanlab.init(
    project="cats-detection",
    experiment_name="YoloX-baseline",
    description="Baseline experiment for the YoloX detection model, primarily for subsequent comparisons.",
)
```

### Setting the Log File Save Location

> Only valid when mode="local"

The following code demonstrates how to save log files to a custom directory:

```python
swanlab.init(
    logdir="path/to/my_custom_dir",
    mode="local",
)
```

### Adding Experiment Metadata to the Configuration

```python
swanlab.init(
    config={
        "learning-rate": 1e-4,
        "model": "CNN",
    }
)
```

### Uploading to an Organization

```python
swanlab.init(
    workspace="[organization's username]"
)
```

### Plugins

For more information about plugins, refer to the [Plugins](/zh/plugin/plugin-index.md) documentation.

```python
from swanlab.plugin.notification import EmailCallback

email_callback = EmailCallback(...)

swanlab.init(
    callbacks=[email_callback]
)
```

## Deprecated Parameters

• `cloud`: Replaced by the `mode` parameter in v0.3.4. The parameter is still available and will override the `mode` setting.