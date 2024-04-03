# swanlab.Text

[Github源代码](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/modules/text.py)

```python
Text(
    data: Union[str, List["Text"]],
    caption: str = None,
) -> None
```

| 参数    | 描述                                                              |
|-------|-----------------------------------------------------------------|
| data  | (Union[str, List["Text"]]) 接收字符串。                                      |
| caption | (str) 文本的标签。用于在实验看板中对data进行标记。                     |

## 介绍

对文本数据做转换，以被`swanlab.log()`记录。

### 记录字符串文本

记录单个字符串文本：

```python{4}
import swanlab

swanlab.init()
text = swanlab.Text("a awesome text.")
swanlab.log({"examples": text})
```

记录多个字符串文本：

```python
import swanlab

swanlab.init()

examples = []
for i in range(3):
    text = swanlab.Text("a awesome text.")
    examples.append(text)

swanlab.log({"examples": examples})
```