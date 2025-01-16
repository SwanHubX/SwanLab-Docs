# Set Experiment Configuration

Use `swanlab.config` to save your training configurations, such as:
- Hyperparameters
- Input settings, such as dataset name or model type
- Any other variables for your experiment

`swanlab.config` allows you to easily analyze your experiments and reproduce your work in the future. You can also compare configurations of different experiments in the SwanLab application and see how different training configurations affect model outputs.

## Set Experiment Configuration

`config` is typically defined at the beginning of the training script. Of course, different AI workflows may vary, so `config` also supports defining at different locations in the script to meet flexible needs.

The following sections outline different scenarios for defining experiment configurations.

### Set in init

The following code snippet demonstrates how to define `config` using a Python dictionary and how to pass this dictionary as a parameter when initializing a SwanLab experiment:

```python
import swanlab

# Define a config dictionary
config = {
  "hidden_layer_sizes": [64, 128],
  "activation": "ELU",
  "dropout": 0.5,
  "num_classes": 10,
  "optimizer": "Adam",
  "batch_normalization": True,
  "seq_length": 100,
}

# Pass the config dictionary when you initialize SwanLab
run = swanlab.init(project="config_example", config=config)
```

Accessing values in `config` is similar to accessing values in other dictionaries in Python:

- Access values using the key name as an index
  ```python
  hidden_layer_sizes = swanlab.config["hidden_layer_sizes"]
  ```
- Access values using the `get()` method
  ```python
  activation = swanlab.config.get["activation"]
  ```
- Access values using dot notation
  ```python
  dropout = swanlab.config.dropout
  ```

### Set with argparse

You can set `config` using an `argparse` object. `argparse` is a very powerful module in the Python standard library (Python >= 3.2) for parsing program arguments from the command-line interface (CLI). This module allows developers to easily write user-friendly command-line interfaces.

You can directly pass the `argparse` object to set `config`:

```python
import argparse
import swanlab

# Initialize Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=20)
parser.add_argument('--learning-rate', default=0.001)
args = parser.parse_args()

swanlab.init(config=args)
```

This is equivalent to `swanlab.init(config={"epochs": 20, "learning-rate": 0.001})`

### Set at Different Locations in the Script

You can add more parameters to the `config` object at different locations throughout your script.

The following code snippet shows how to add new key-value pairs to the `config` object:

```python
import swanlab

# Define a config dictionary
config = {
  "hidden_layer_sizes": [64, 128],
  "activation": "ELU",
  "dropout": 0.5,
  # ... other configuration items
}

# Pass the config dictionary when you initialize SwanLab
run = swanlab.init(project="config_example", config=config)

# Update config after initializing SwanLab
swanlab.config["dropout"] = 0.8
swanlab.config.epochs = 20
swanlab.config.set["batch_size", 32]
```

### Set with Configuration Files

You can initialize `config` using json and yaml configuration files. For details, please refer to [Create Experiment with Configuration File](/en/guide_cloud/experiment_track/create-experiment-by-configfile).