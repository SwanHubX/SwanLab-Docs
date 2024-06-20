# swanlab.Audio

[Github源代码](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/modules/audio.py)

```python
Audio(
    data_or_path: Union[str, np.ndarray],
    sample_rate: int = 44100,
    caption: str = None,
) -> None
```

| 参数          | 描述                                                                                                     |
|-------------|--------------------------------------------------------------------------------------------------------|
| data_or_path | (Union[str, np.ndarray]) 接收音频文件路径、numpy数组。Audio类将判断接收的数据类型做相应的转换。 |
| sample_rate | (int) 音频的采样率，默认为44100。                                             |
| caption     | (str) 音频的标签。用于在实验看板中展示音频时进行标记。                                                      |

## 介绍

对各种类型的音频数据做转换，以被`swanlab.log()`记录。

![](/assets/media-audio-1.jpg)

### 从numpy array创建

记录单个音频：

```python
import numpy as np
import swanlab

run = swanlab.init()

# 创建一个numpy array类型的音频
white_noise = np.random.randn(2, 100000)
# 传入swanlab.Audio，设置采样率
audio = swanlab.Audio(white_noise, caption="white_noise")

run.log({"examples": audio})
```

记录多个音频：

```python
import numpy as np
import swanlab

run = swanlab.init()

# 创建一个列表
examples = []
for i in range(3):
    white_noise = np.random.randn(100000)
    audio = swanlab.Audio(white_noise, caption="audio_{i}")
    # 列表中添加swanlab.Audio类型对象
    examples.append(audio)

run.log({"examples": examples})
```

### 从文件路径创建

```python
import swanlab

run = swanlab.init()
audio = swanlab.Audio("path/to/file")

run.log({"examples": audio})
```