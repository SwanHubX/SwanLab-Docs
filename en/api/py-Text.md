# swanlab.Text

[Github Source Code](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/modules/text.py)

```python
Text(
    data: Union[str],
    caption: str = None,
) -> None
```

| Parameter | Description |
|-----------|-------------|
| data      | (Union[str]) Accepts a string. |
| caption   | (str) The label for the text. Used to mark the data in the experiment dashboard. |

## Introduction

Convert text data to be recorded by `swanlab.log()`.

### Logging String Text

Logging a single string text:

```python{4}
import swanlab

swanlab.init()
text = swanlab.Text("an awesome text.")
swanlab.log({"examples": text})
```

Logging multiple string texts:

```python
import swanlab

swanlab.init()

examples = []
for i in range(3):
    text = swanlab.Text("an awesome text.")
    examples.append(text)

swanlab.log({"examples": examples})
```