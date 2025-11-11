# swanlab.init

```python
init(
    project: str = None,
    workspace: str = None,
    experiment_name: str = None,
    description: str = None,
    job_type: str = None,
    group: str = None,
    tags: List[str] = None,
    config: Union[dict, str] = None,
    logdir: str = None,
    mode: MODES = None,    
    load: str = None,
    public: bool = None,
    callbacks: List[SwanKitCallback] = None,
    settings: Settings = None,
    id: str = None,
    resume: Union[Literal['must', 'allow', 'never'], bool] = None,
    reinit: bool = None,
    **kwargs,
)
```

| Parameter         | Description |
|-------------------|-------------|
| project           | (str) The name of the project. If not specified, the name of the current working directory will be used. |
| workspace         | (str) The workspace. By default, experiments are synchronized to your personal space. If you want to upload to an organization, specify the organization's username. |
| experiment_name   | (str) The name of the experiment. If not specified, it will default to a format like "swan-1" (animal name + sequence number). |
| job_type          | (str) The type of job. If not specified, it will default to empty string. |
| group             | (str) The group of the experiment. If not specified, it will default to None. |
| tags              | (list) Tags for the experiment. Can pass a list of strings, and the tags will be displayed in the tag bar at the top of the experiment. |
| description       | (str) A description of the experiment. If not specified, it defaults to None. |
| config            | (dict, str) Configuration for the experiment. You can record hyperparameters and other information here. Supports passing a configuration file path (yaml or json). |
| logdir            | (str) The path to store offline dashboard log files. Defaults to `swanlog`. |
| mode              | (str) Sets the mode for creating SwanLab experiments. Options are "cloud", "local", "offline", or "disabled". Default is "cloud".<br>`cloud`: Uploads the experiment to the cloud (public or private deployment).<br>`offline`: Only records experiment data locally.<br>`local`: Does not upload to the cloud but records experiment information locally.<br>`disabled`: Neither uploads nor records. |
| load              | (str) The path to a configuration file to load. Supports yaml and json files. |
| public            | (bool) Sets the visibility of the SwanLab project created directly via code. Default is False (private). |
| callbacks         | (list) Sets experiment callback functions. Supports subclasses of `swanlab.toolkit.callback.SwanKitCallback`. |
| name              | (str) Same effect as `experiment_name`. Lower priority than `experiment_name`. |
| notes             | (str) Same effect as `description`. Lower priority than `description`. |
| tags              | (list) Tags for the experiment. |
| settings          | (dict) Settings for the experiment. Supports passing a `swanlab.Settings` object. |
| id                | (str) The ID of the last experiment. Used to resume the last experiment. Must be a 21-character string. |
| resume            | (str) Resume mode. Can be "must", "allow", "never", or True/False. Default is None.<br>`must`: You must pass the `id` parameter, and the experiment must exist.<br>`allow`: If an experiment exists, it will be resumed. Otherwise, a new experiment will be created.<br>`never`: The `id` parameter cannot be passed, and a new experiment will be created. (This is equivalent to not enabling `resume`.) |
| reinit            | (bool) Whether to reinitialize the experiment. If True, the last experiment will be `finish`ed every time `swanlab.init()` is called; default is None. |


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

### Setting Tags

```python
swanlab.init(
    tags=["yolo", "detection", "baseline"]
)
```

### Setting Group

```python
swanlab.init(
    group="good_try",
)
```

### Setting the Log File Save Location

The following code demonstrates how to save log files to a custom directory:

```python
swanlab.init(
    logdir="path/to/my_custom_dir",
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

### Resume Training

"Resume Training" refers to the ability to resume training from a previous state. If you previously had an experiment with the status of `completed` or `interrupted`, and you need to add more experimental data, you can use the `resume` and `id` parameters to restore this experiment.

```python
swanlab.init(
    resume=True,
    id="14pk4qbyav4toobziszli",  # id must be a 21-character string
)
```

The experiment ID can be found in the experiment's "Environment" tab or in the URL. It must be a 21-character string.

:::tip Use Cases for `resume`

1. The previous training process was interrupted. You want to continue training based on a checkpoint and have the experiment chart continue from the previous swanlab experiment, rather than creating a new one.
2. Training and evaluation are split into two separate processes, but you want both the evaluation and training records to appear in the same swanlab experiment.
3. Some configuration parameters were incorrectly filled out, and you want to update them.

:::

:::warning ⚠️ Notes

1. Experiments generated by cloning a project cannot be resumed.

:::

Breakpoint continuation supports three modes:

1. `allow`: If an experiment corresponding to the provided `id` exists under the project, it will be resumed. Otherwise, a new experiment will be created.
2. `must`: If an experiment corresponding to the provided `id` exists under the project, it will be resumed. Otherwise, an error will be thrown.
3. `never`: The `id` parameter cannot be passed, and a new experiment will be created instead. (This is equivalent to not enabling `resume`.)

:::info
Setting `resume=True` has the same effect as `resume="allow"`.  
Setting `resume=False` has the same effect as `resume="never"`.
:::

Test code:

```python
import swanlab

run = swanlab.init()
swanlab.log({"loss": 2, "acc": 0.4})
run.finish()

run = swanlab.init(resume=True, id=run.id)
swanlab.log({"loss": 0.2, "acc": 0.9})
```

## Deprecated Parameters

• `cloud`: Replaced by the `mode` parameter in v0.3.4. The parameter is still available and will override the `mode` setting.