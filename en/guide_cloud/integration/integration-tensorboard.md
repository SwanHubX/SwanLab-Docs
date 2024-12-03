# Tensorboard

[TensorBoard](https://github.com/tensorflow/tensorboard) is a visualization tool provided by Google TensorFlow, designed to help understand, debug, and optimize machine learning models. It presents various metrics and data from the training process through a graphical interface, allowing developers to gain a more intuitive understanding of the model's performance and behavior.

![TensorBoard](/assets/ig-tensorboard.png)

You can use `swanlab convert` to convert Tensorboard-generated Tfevent files into SwanLab experiments.

## Method 1: Command Line Conversion

```bash
swanlab convert -t tensorboard -tb_logdir [TFEVENT_LOGDIR]
```

Here, `[TFEVENT_LOGDIR]` refers to the path of the log files generated when you previously recorded experiments using Tensorboard.

The SwanLab Converter will automatically detect `tfevent` files in the specified path and its subdirectories (default subdirectory depth is 3) and create a SwanLab experiment for each `tfevent` file.

## Method 2: Conversion Within Code

```python
from swanlab.converter import TFBConverter

tfb_converter = TFBConverter(convert_dir="[TFEVENT_LOGDIR]")
tfb_converter.run()
```

This method achieves the same effect as the command line conversion.

## Parameter List

| Parameter | Corresponding CLI Argument | Description                  | 
| --------- | -------------------------- | ---------------------------- | 
| convert_dir | - | Path to Tfevent files       | 
| project    | -p, --project              | SwanLab project name         |
| workspace  | -w, --workspace            | SwanLab workspace name       |
| cloud      | --cloud                    | Whether to use the cloud version, default is True | 
| logdir     | -l, --logdir               | SwanLab log file save path   | 

Example:

```python
from swanlab.converter import TFBConverter

tfb_converter = TFBConverter(
    convert_dir="./runs",
    project="Tensorboard-Converter",
    workspace="SwanLab",
    logdir="./logs",
    )
tfb_converter.run()
```

The equivalent CLI command:
```bash
swanlab convert -t tensorboard --tb_logdir ./runs -p Tensorboard-Converter -w SwanLab -l ./logs
```

Executing the above script will create a project named `Tensorboard-Converter` under the `SwanLab` workspace, convert the tfevent files in the `./runs` directory into individual swanlab experiments, and save the logs generated during the swanlab runtime in the `./logs` directory.