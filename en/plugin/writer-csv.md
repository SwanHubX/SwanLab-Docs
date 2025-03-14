# CSV Table Logger

If you wish to record some configuration information and metrics locally in a CSV file during training (in a format consistent with the "Table View" on the SwanLab webpage), we highly recommend using the `CSV Logger` plugin.

:::warning Improving the Plugin
All SwanLab plugins are open-source. You can view the [Github source code](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/plugin/writer.py). We welcome your suggestions and PRs!
:::

## Plugin Usage

**1. Initialize the CSV Logger:**

```python
from swanlab.plugin.writer import CSVWriter

csv_writer = CSVWriter(dir="logs")
```

The `dir` parameter specifies the save path for the CSV file. By default, it is saved in the current working directory.

**2. Pass the Plugin:**

```python
swanlab.init(
    ...
    callbacks=[csv_writer]
)
```

After executing the code, a `swanlab_run.csv` file will be generated in the `logs` directory, and data recording will begin. For each subsequent training session, a new row will be added to this CSV file.

If you want to specify a different file name, you can pass the `filename` parameter:

```python
csv_writer = CSVWriter(dir="logs", filename="my_csv_file.csv")
```

## Example Code

```python
import swanlab
from swanlab.plugin.writer import CSVWriter
import random

csv_writer = CSVWriter(dir="logs")

# Create a SwanLab project
swanlab.init(
    # Set the project name
    project="my-awesome-project",
    
    # Set hyperparameters
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
        "batch_size": 128
    },
    callbacks=[csv_writer]
)

# Simulate a training session
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
  acc = 1 - 2 ** -epoch - random.random() / epoch - offset
  loss = 2 ** -epoch + random.random() / epoch + offset

  # Log training metrics
  swanlab.log({"acc": acc, "loss2": loss})

# [Optional] Finish training, which is necessary in notebook environments
swanlab.finish()
```