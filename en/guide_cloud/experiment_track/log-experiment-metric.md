# Log Experiment Metrics

Use the SwanLab Python library to log metrics and media data at each step (step) of training.

SwanLab collects metric names and data (key-value) in the training loop using `swanlab.log()`. The data collected from the script will be saved to a directory named `swanlog` in your local directory (the directory name can be set by the `logdir` parameter of `swanlab.init`) and then synchronized to the SwanLab cloud server.

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

⚠️It is important to note that the value in `swanlab.log({key: value})` must be of type `int` / `float` / `BaseType` (if a `str` type is passed, it will first be attempted to be converted to `float`, and if the conversion fails, an error will be reported). The `BaseType` type mainly refers to multimedia data. For details, please refer to [Log Multimedia Data](/guide_cloud/experiment_track/log-media.md).

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