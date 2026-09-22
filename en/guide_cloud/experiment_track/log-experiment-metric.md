# Log Experiment Metrics

Use the SwanLab Python library to log metrics and media data at each step (step) of training.

SwanLab collects metric names and data (key-value) in the training loop using `swanlab.log()`. The data collected from the script will be saved to a directory named `swanlog` in your local directory (the directory name can be set by the `logdir` parameter of `swanlab.init`) and then synchronized to the SwanLab cloud server.

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/en/guide_cloud/experiment_track/log-experiment-metric/line.png)

## Log Scalar Metrics

In the training loop, compose the metric name and data into a key-value dictionary and pass it to `swanlab.log()` to complete the logging of one metric:

```python
for epoch in range(num_epochs):
    for data, ground_truth in dataloader:
        predict = model(data)
        loss = loss_fn(predict, ground_truth)
        # Log metric, metric name is loss
        swanlab.log({"loss": loss})
```

When `swanlab.log` is used for logging, it will aggregate the dictionary `{metric name: metric}` to a unified location based on the metric name.

⚠️It is important to note that the value in `swanlab.log({key: value})` must be of type `int` / `float` / `BaseType` (if a `str` type is passed, it will first be attempted to be converted to `float`, and if the conversion fails, an error will be reported). The `BaseType` type mainly refers to multimedia data. For details, please refer to [Log Multimedia Data](./log-media.md).

Each time a record is made, a `step` is assigned to that record. By default, `step` starts from 0 and, with each subsequent logging under the same metric name, `step` equals the maximum `step` of historical records for that metric name + 1. For example:

```python
import swanlab
swanlab.init()

...

swanlab.log({"loss": loss, "acc": acc})
# In this record, loss has step 0, acc has step 0

swanlab.log({"loss": loss, "iter": iter})
# In this record, loss has step 1, iter has step 0, acc has step 0

swanlab.log({"loss": loss, "iter": iter})
# In this record, loss has step 2, iter has step 1, acc has step 0
```

## Metric Grouping

In the script, you can group charts by prefixing the metric name with a group name separated by "/" (slash). For example, `train/loss` will be grouped under the name "train", and `val/loss` will be grouped under the name "val":

```python
# Grouped under train
swanlab.log({"train/loss": loss})
swanlab.log({"train/batch_cost": batch_cost})

# Grouped under val
swanlab.log({"val/acc": acc})
```

:::tip
For metric names with multiple `/` separators, the current strategy uses the last separator.
For example, a metric named `a/b/c` is grouped under `a/b` by default.
If you need a custom group name, you can define it via [swanlab.define_metric()](../../api/py-define_metric.md) before the metric is logged.
:::

## Specify the Step for Logging

When the logging frequency of some metrics is inconsistent but you want their steps to be aligned, you can achieve alignment by setting the `step` parameter of `swanlab.log`:

```python
for iter, (data, ground_truth) in enumerate(train_dataloader):
    predict = model(data)
    train_loss = loss_fn(predict, ground_truth)
    swanlab.log({"train/loss": loss}, step=iter)

    # Validation part
    if iter % 1000 == 0:
        acc = val_trainer(model)
        swanlab.log({"val/acc": acc}, step=iter)
```

It is important to note that the same metric name is not allowed to have two identical step data. Once this happens, SwanLab will keep the first recorded data and discard the later recorded data.

## Print Metrics

You might want to print metrics during the training loop. You can control whether to print the metrics to the console (in the form of a `dict`) using the `print_to_console` parameter:

```python
swanlab.log({"acc": acc}, print_to_console=True)
```

Alternatively:

```python
print(swanlab.log({"acc": acc}))
```

## Automatically Log Environment Information

SwanLab automatically logs the following information during the experiment:

- **Command Line Output**: Standard output and standard error streams are automatically recorded and displayed in the "Logs" tab of the experiment page.
- **Experiment Environment**: Records dozens of environment information including operating system, hardware configuration, Python interpreter path, running directory, Python library dependencies, etc.
- **Training Time**: Records the start time and total duration of training.

## Asynchronous Logging

:::info
`swanlab.async_log()` requires SwanLab SDK **v0.8.0 or higher**.
:::

If you need to log metrics that require expensive computation or I/O, you can use `swanlab.async_log()` to execute the computation in the background without blocking the training loop. It accepts the same data format as `swanlab.log()` and returns a `Future` immediately.

```python
import swanlab
import time

swanlab.init(project="my-project")

def compute_metric():
    time.sleep(2)  # Simulate expensive computation
    return {"score": 0.95}

# Execute in background, training continues without blocking
future = swanlab.async_log(compute_metric, step=1)

swanlab.finish()
```

`swanlab.async_log()` supports multiple execution modes (`threading`, `asyncio`, `spawn`). For detailed usage and all mode options, see the [async_log API documentation](../../api/py-async-log.md).

## Custom X Axis

:::info
`swanlab.define_metric()` requires SwanLab SDK **v0.10.0 or higher**.
:::

By default, metric charts use step as the X axis. In some training scenarios (e.g., you want to view metric changes by epoch, learning rate, etc.), you can use `swanlab.define_metric()` to associate a chart's X axis with another metric:

```python
import swanlab

swanlab.init(project="my-project")

# Use train/epoch as the X axis of train/loss
swanlab.define_metric("train/loss", x_axis="train/epoch")

for epoch in range(num_epochs):
    # Log the X-axis metric first
    swanlab.log({"train/epoch": epoch})
    # ... training ...
    swanlab.log({"train/loss": loss})
```

X-axis and Y-axis metrics can be logged separately — the SDK automatically fills in the most recent X value for each Y value. The `key` also supports glob batch matching (e.g., `train/*`), making it easy to define the X axis for a group of metrics at once.

For detailed parameter descriptions and notes on custom X axes, see the [define_metric API documentation](../../api/py-define_metric.md).
