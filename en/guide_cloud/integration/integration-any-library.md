# Add SwanLab into Your Library

This guide provides best practices on how to integrate SwanLab into your Python library to leverage powerful experiment tracking, GPU and system monitoring, hyperparameter logging, and more.

Below, we outline the best practices we've compiled for when you're working with a codebase more complex than a single Python training script or Jupyter Notebook.

**🪵 Table of Contents:**

[[toc]]

## 1. Supplementing Requirements

Before you begin, decide whether to require SwanLab in your library's dependencies:

### 1.1 Adding SwanLab as a Dependency

```plaintext
torch==2.5.0
...
swanlab==0.4.*
```

### 1.2 Making SwanLab an Optional Installation

There are two ways to set up SwanLab as an optional installation.

1. Use a try-except block in your code to raise an error when SwanLab is not installed by the user.

```python
try:
    import swanlab
except ImportError:
    raise ImportError(
        "You are trying to use SwanLab, which is not currently installed."
        "Please install it using pip install swanlab"
    )
```

2. If you are building a Python package, add `swanlab` as an optional dependency in the `pyproject.toml` file:

```toml
[project]
name = "my_awesome_lib"
version = "0.1.0"
dependencies = [
    "torch",
    "transformers"
]

[project.optional-dependencies]
dev = [
    "swanlab"
]
```

## 2. User Login

Your users have several methods to log in to SwanLab:

::: code-group

```bash [Command Line]
swanlab login
```

```python [Python]
import swanlab
swanlab.login()
```

```bash [Environment Variable (Bash)]
export SWANLAB_API_KEY=$YOUR_API_KEY
```

```python [Environment Variable (Python)]
import os
os.environ["SWANLAB_API_KEY"] = "zxcv1234..."
```

:::

If the user is using `swanlab` for the first time without following any of the above steps, they will be automatically prompted to log in when your script calls `swanlab.init`.

## 3. Starting a SwanLab Experiment

An experiment is the computational unit of SwanLab. Typically, you can create an `Experiment` object for each experiment and start it using the `swanlab.init` method.

### 3.1 Initializing the Experiment

Initialize SwanLab and start an experiment in your code:

```python
swanlab.init()
```

You can provide parameters such as project name, experiment name, and workspace for this experiment:

```python
swanlab.init(
    project="my_project",
    experiment_name="my_experiment",
    workspace="my_workspace",
    )
```

::: warning Where to Best Place swanlab.init?

Your library should create the SwanLab experiment as early as possible, as SwanLab automatically collects any output from the console, making debugging easier.

:::

### 3.2 Configuring Three Startup Modes

You can configure SwanLab's startup mode using the `mode` parameter:

::: code-group

```python [Cloud Mode]
swanlab.init(
    mode="cloud",  # Default mode
    )
```

```python [Local Mode]
swanlab.init(
    mode="local",
    )
```

```python [Disabled Mode]
swanlab.init(
    mode="disabled",
    )
```

:::

- **Cloud Mode**: The default mode. SwanLab uploads experiment data to a web server (SwanLab's official cloud or your privately deployed cloud).
- **Local Mode**: SwanLab does not upload experiment data to the cloud but records a special `swanlog` directory that can be opened by the `dashboard` plugin for visualization.
- **Disabled Mode**: SwanLab does not collect any data, and code execution will bypass any `swanlab` related code.

### 3.3 Defining Experiment Hyperparameters/Configuration

Using SwanLab's experiment configuration (config), you can provide metadata about your model, dataset, etc., when creating a SwanLab experiment. You can use this information to compare different experiments and quickly understand the main differences.

Typical configuration parameters you might log include:

- Model name, version, architecture parameters, etc.
- Dataset name, version, number of training/test data points, etc.
- Training parameters such as learning rate, batch size, optimizer, etc.

The following code snippet shows how to log configuration:

```python
config = {"learning_rate": 0.001, ...}
swanlab.init(..., config=config)
```

**Updating Configuration**:

Use the `swanlab.config.update` method to update the configuration. This method is convenient for updating the config dictionary after defining it.

For example, you might want to add model parameters after instantiating the model:

```python
swanlab.config.update({"model_params": "1.5B"})
```

## 4. Logging Data to SwanLab

Create a dictionary where the key is the name of the metric and the value is the metric's value. Pass this dictionary object to `swanlab.log`:

::: code-group

```python [Logging a Set of Metrics]
metrics = {"loss": 0.5, "accuracy": 0.8}
swanlab.log(metrics)
```

```python [Logging Metrics in a Loop]
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data:
        prediction = model(input)
        loss = loss_fn(prediction, ground_truth)
        metrics = { "loss": loss }
        swanlab.log(metrics)
```

:::

If you have many metrics, you can use prefixes in the metric names (e.g., `train/...` and `val/...`). In the UI, SwanLab will automatically group them to isolate different categories of chart data:

```python
metrics = {
    "train/loss": 0.5,
    "train/accuracy": 0.8,
    "val/loss": 0.6,
    "val/accuracy": 0.7,
}
swanlab.log(metrics)
```

For more information on `swanlab.log`, refer to the [Logging Metrics](../experiment_track/log-experiment-metric) section.

## 5. Advanced Integration

You can also explore advanced SwanLab integrations in the following:

- [HuggingFace Transformers](../integration/integration-huggingface-transformers.md)
- [PyTorch Lightning](../integration/integration-pytorch-lightning.md)