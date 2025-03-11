# swanlab.init

```python
init(
    project: str = None,
    workspace: str = None,
    experiment_name: str = None,
    description: str = None,
    config: Union[dict, str] = None,
    logdir: str = None,
    suffix: str = "default",
    mode: str = "cloud",
    load: str = None,
    public: bool = None,
    callbacks: list = None,
    **kwargs,
)
```

| Parameter     | Description |
|---------------|-------------|
| project       | (str) Project name. If not specified, the name of the running directory is used. |
| workspace     | (str) Workspace. By default, the experiment is synchronized to your personal space. If you want to upload to an organization, fill in the organization's username. |
| experiment_name | (str) Experiment name. If not specified, it defaults to "exp". The full experiment name is composed of `experiment_name + "_" + suffix`. |
| description   | (str) Experiment description. If not specified, it defaults to None. |
| config        | (dict, str) Experiment configuration. You can record some hyperparameters and other information here. Supports passing in configuration file paths, supports yaml and json files. |
| logdir        | (str) Log file storage path. Defaults to `swanlog`. |
| suffix        | (str, None, bool) Suffix for `experiment_name`. The full experiment name is composed of `experiment_name` and `suffix`. <br> The default value is "default", which means the default suffix rule is `'%b%d-%h-%m-%s'`, for example: `Feb03_14-45-37`. <br> Setting it to `None` or `False` will not add a suffix. |
| mode          | (str) Sets the mode for creating the SwanLab experiment. Options include "cloud", "local", "disabled". Defaults to "cloud". <br> `cloud`: Uploads the experiment to the cloud. <br> `local`: Does not upload to the cloud but records the experiment locally. <br> `disabled`: Does not upload or record. |
| load          | (str) Path to the configuration file to load. Supports yaml and json files. |
| public        | (bool) Sets the visibility of the SwanLab project created directly by code. Defaults to False, i.e., private. |
| callbacks     | (list) Sets the experiment callback functions. Supports `swankit.callback.SwanKitCallback` subclasses. |
| name       | (str) The same as `experiment_name`. |
| notes       | (str) The same as `description`. |

## Introduction

- In the machine learning training process, we can add `swandb.init()` at the beginning of the training and testing scripts. SwanLab will track each step of the machine learning process.

- `swanlab.init()` generates a new background process to record data into the experiment. By default, it also synchronizes the data to swanlab.pro so that you can see the visualization results online in real-time.

- Before using `swanlab.log()` to record data, you need to call `swanlab.init()`:

```python
import swanlab

swanlab.init()
swanlab.log({"loss": 0.1846})
```

- Calling `swanlab.init()` returns an object of type `SwanLabRun`, which can also perform `log` operations:

```python
import swanlab

run = swanlab.init()
run.log({"loss": 0.1846})
```

- At the end of the script, we will automatically call `swanlab.finish` to end the SwanLab experiment. However, if `swanlab.init()` is called from a subprocess, such as in a Jupyter notebook, you must explicitly call `swanlab.finish` at the end of the subprocess.

```python
import swanlab

swanlab.init()
swanlab.finish()
```

## More Usage

### Setting Project, Experiment Name, and Description

```python
swanlab.init(
    project="cats-detection",
    experiment_name="YoloX-baseline",
    description="Baseline experiment for the YoloX detection model, mainly used for subsequent comparisons.",
)
```

### Setting the Log File Save Location

The following code demonstrates how to save log files to a custom directory:

```python
swanlab.init(
    logdir="path/to/my_custom_dir"
)
```

### Adding Experiment-related Metadata to the Experiment Configuration

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

## Deprecated Parameters

- `cloud`: Replaced by the `mode` parameter in v0.3.4. The parameter is still available but will override the `mode` setting.