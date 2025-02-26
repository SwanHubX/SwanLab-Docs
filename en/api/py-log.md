# log

[Github Source Code](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/sdk.py)

```python
log(
    data: Dict[str, DataType],
    step: int = None,
    print_to_console: bool = False,
)
```

| Parameter | Description |
|-----------|-------------|
| data      | (Dict[str, DataType]) Required. Pass in a dictionary of key-value pairs, where the key is the metric name and the value is the metric value. The value supports int, float, types that can be converted by float(), or any `BaseType` type. |
| step      | (int) Optional. This parameter sets the step number for the data. If step is not set, it will start from 0 and increment by 1 for each subsequent step. |
| print_to_console | (bool) Optional, default is False. When set to True, the key and value of data will be printed to the terminal in dictionary format. |

## Introduction

`swanlab.log` is the core API for metric logging. Use it to record data in experiments, such as scalars, images, audio, and text.

The most basic usage is as shown in the following code, which will record the accuracy and loss values into the experiment, generate visual charts, and update the summary values of these metrics:

```python
swanlab.log({"acc": 0.9, "loss":0.1462})
```

In addition to scalars, `swanlab.log` supports logging multimedia data, including images, audio, text, etc., and has a good display effect in the UI.

## Print the Passed Dictionary

`swanlab.log` supports printing the `key` and `value` of the passed `data` to the terminal. By default, this feature is disabled. To enable printing, you need to set `print_to_console=True`.

```python
swanlab.log({"acc": 0.9, "loss": 0.1462}, print_to_console=True)
```

Alternatively, you can also print it in this way:

```python
print(swanlab.log({"acc": 0.9, "loss": 0.1462}))
```

## More Usage

- Logging [Images](/en/api/py-Image.md)
- Logging [Audio](/en/api/py-Audio.md)
- Logging [Text](/en/api/py-Text.md)