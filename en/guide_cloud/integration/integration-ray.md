# Ray

[Ray](https://github.com/ray-project/ray) is a distributed computing framework specifically designed for large-scale parallel tasks and reinforcement learning applications. It was developed by a research team from the University of California, Berkeley, aiming to simplify the process of building high-performance, scalable distributed applications. Ray supports Python and Java, and can be easily integrated into existing machine learning, data processing, and reinforcement learning workflows.

![ray](./ray/logo.png)

SwanLab supports Ray experiment logging. You can conveniently record experiment metrics and hyperparameters through `SwanLabLoggerCallback`.

## 1. Import SwanLabCallback

```python
from swanlab.integration.ray import SwanLabLoggerCallback
```

`SwanLabLoggerCallback` is a log recording class adapted for `Ray`.

The parameters that can be defined in `SwanLabLoggerCallback` include:
- `project`: Project name
- `workspace`: Workspace name
- Other parameters consistent with `swanlab.init`

## 2. Integration with `tune.Tuner`

```python
tuner = tune.Tuner(
    ...
    run_config=tune.RunConfig(
        callbacks=[SwanLabLoggerCallback(project="Ray_Project")],
    ),
)
```

## 3. Complete Example

```python
import random
from ray import tune
from swanlab.integration.ray import SwanLabLoggerCallback

def train_func(config):
    offset = random.random() / 5
    for epoch in range(2, config["epochs"]):
        acc = 1 - (2 + config["lr"]) ** -epoch - random.random() / epoch - offset
        loss = (2 + config["lr"]) ** -epoch + random.random() / epoch + offset
        tune.report({"acc": acc, "loss": loss})


tuner = tune.Tuner(
    train_func,
    param_space={
        "lr": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
        "epochs": 10,
    },
    run_config=tune.RunConfig(
        callbacks=[SwanLabLoggerCallback(project="Ray_Project")],
    ),
)
results = tuner.fit()
```

![ray-tune](./ray/demo.png)