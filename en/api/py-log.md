# log

[Github Source Code](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/sdk.py)

```python
log(
    data: Dict[str, DataType],
    step: int = None,
)
```

| Parameter | Description |
|-----------|-------------|
| data      | (Dict[str, DataType]) Required. Pass in a dictionary of key-value pairs, where the key is the metric name and the value is the metric value. The value supports int, float, types that can be converted by float(), or any `BaseType` type. |
| step      | (int) Optional. This parameter sets the step number for the data. If step is not set, it will start from 0 and increment by 1 for each subsequent step. |

## Introduction

`swanlab.log` is the core API for metric logging. Use it to record data in experiments, such as scalars, images, audio, and text.

The most basic usage is as shown in the following code, which will record the accuracy and loss values into the experiment, generate visual charts, and update the summary values of these metrics:

```python
swanlab.log({"acc": 0.9, "loss":0.1462})
```

In addition to scalars, `swanlab.log` supports logging multimedia data, including images, audio, text, etc., and has a good display effect in the UI.

## More Usage

- Logging [Images](/en/api/py-Image.md)
- Logging [Audio](/en/api/py-Audio.md)
- Logging [Text](/en/api/py-Text.md)