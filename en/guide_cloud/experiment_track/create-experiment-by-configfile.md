# Create Experiment with Configuration File

This section will introduce how to create SwanLab experiments using configuration files in json or yaml format.

## Load Configuration File with swanlab.config

The `config` parameter of `swanlab.init` supports passing the path of a configuration file in json or yaml format, and parses the configuration file into a dictionary for experiment creation.

### Using json File

Below is an example of a configuration file in json format:

```json
{
    "epochs": 20,
    "learning-rate": 0.001
}
```

Pass the path of the configuration file to the `config` parameter, it will parse the configuration file into a dictionary:

```python
swanlab.init(config="swanlab-init-config.json")
# Equivalent to swanlab.init(config={"epochs": 20, "learning-rate": 0.001})
```

### Using yaml File

Below is an example of a configuration file in yaml format:

```yaml
epochs: 20
learning-rate: 0.001
```

Pass the path of the configuration file to the `config` parameter, it will parse the configuration file into a dictionary:

```python
swanlab.init(config="swanlab-init-config.yaml")
# Equivalent to swanlab.init(config={"epochs": 20, "learning-rate": 0.001})
```

## Load Configuration File with swanlab.init

The `load` parameter of `swanlab.init` supports passing the path of a configuration file in json or yaml format, and parses the configuration file for experiment creation.

### Using json File

Below is an example of a configuration file in json format:

```json
{
    "project": "cat-dog-classification",
    "experiment_name": "Resnet50",
    "description": "My first AI experiment",
    "config": {
        "epochs": 20,
        "learning-rate": 0.001
    }
}
```

Pass the path of the configuration file to the `load` parameter, it will parse the configuration file to initialize the experiment:

```python
swanlab.init(load="swanlab-config.json")
# Equivalent to
# swanlab.init(
#     project="cat-dog-classification",
#     experiment_name="Resnet50",
#     description="My first AI experiment",
#     config={
#         "epochs": 20,
#         "learning-rate": 0.001
#     }
# )
```

### Using yaml File

Below is an example of a configuration file in yaml format:

```yaml
project: cat-dog-classification
experiment_name: Resnet50
description: My first AI experiment
config:
  epochs: 20
  learning-rate: 0.001
```

Pass the path of the configuration file to the `load` parameter, it will parse the configuration file to initialize the experiment:

```python
swanlab.init(load="swanlab-config.yaml")
# Equivalent to
# swanlab.init(
#     project="cat-dog-classification",
#     experiment_name="Resnet50",
#     description="My first AI experiment",
#     config={
#         "epochs": 20,
#         "learning-rate": 0.001
#     }
# )
```

## FAQ

### 1. Is the configuration file naming fixed?

The naming of the configuration file is free, but it is recommended to use `swanlab-init` and `swanlab-init-config` as the configuration names.

### 2. What is the relationship between the parameters in the configuration file and the script?

The priority of the parameters in the script is higher than that in the configuration file, that is, the parameters in the script will override the parameters in the configuration file.

For example, there is a yaml configuration file and a code snippet below:

```yaml
project: cat-dog-classification
experiment_name: Resnet50
description: My first AI experiment
config:
  epochs: 20
  learning-rate: 0.001
```

```python
swanlab.init(
    experiment_name="resnet101",
    config={"epochs": 30},
    load="swanlab-init.yaml"
)
```

The final `experiment_name` is resnet101, and `config` is {"epochs": 30}.