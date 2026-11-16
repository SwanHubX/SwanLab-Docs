# swanlab.Video

[Github源代码](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/sdk/internal/run/transforms/video/__init__.py)

```python
Video(
    data_or_path: Union[str, Path, bytes, BytesIO, np.ndarray, Video],
    caption: str = None,
    fps: int = None,
    format: Literal["gif", "mp4", "webm", "ogg"] = None,
) -> None
```

| 参数         | 描述                                                                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| data_or_path | (Union[str, Path, bytes, BytesIO, np.ndarray, Video]) 接收视频文件路径、二进制数据（`bytes` / `BytesIO`）、`numpy` 数组或 `swanlab.Video` 实例。 |
| caption      | (str, optional) 视频说明文字，展示在实验看板的视频下方。                                                                                         |
| fps          | (int, optional) 将数组编码为视频时的帧率，默认为 `4`。传入文件、二进制数据或 `Video` 对象时忽略该参数。                                          |
| format       | (str, optional) 视频格式，可选 `"gif"`、`"mp4"`、`"webm"`、`"ogg"`。传入文件时由扩展名决定；传入数组或非 GIF 二进制数据时必填。                  |

## 介绍

封装视频数据以供 `swanlab.log()` 记录，支持 **GIF**、**MP4**、**WebM**、**OGG** 四种格式：

- **视频文件与二进制数据**：原样保存，保留原有画质与音轨；
- **NumPy 数组 / Tensor**：在本地编码为指定格式的无声视频。

::: warning 环境依赖
**⚠️注意**：传入`numpy`数组或 Tensor 需安装扩展依赖：`pip install "swanlab[media]"`
:::

### 从文件路径创建

支持传入路径字符串或 `pathlib.Path`，格式由扩展名（`.gif`、`.mp4`、`.webm`、`.ogg`）自动推断：

```python
import swanlab

swanlab.init()

video = swanlab.Video("path/to/video.mp4", caption="video example")
swanlab.log({"video": video})
```

### 从二进制数据创建

支持 `bytes` 和 `io.BytesIO`。GIF 可自动识别，其他格式需显式指定 `format`：

```python
import swanlab

with open("path/to/video.webm", "rb") as f:
    video = swanlab.Video(f.read(), format="webm", caption="video example")

swanlab.init()
swanlab.log({"video": video})
```

### 从 NumPy 数组创建

支持形状为 `(T, C, H, W)` 或 `(B, T, C, H, W)` 的数组，其中 `B` 为批次大小、`T` 为帧数、`C` 为通道数（`1`、`3` 或 `4`）、`H`/`W` 为高宽：

- `dtype` 必须为 `uint8`（`0~255`），其他类型请先转换；
- 4 通道（RGBA）编码时会忽略 Alpha 通道，按 RGB 输出；
- 必须指定 `format`，`fps` 默认为 `4`。

```python
import numpy as np
import swanlab

swanlab.init()

# 64 帧 128x128 的 RGB 视频
frames = np.random.randint(0, 256, (64, 3, 128, 128), dtype=np.uint8)
video = swanlab.Video(frames, format="gif", fps=8, caption="random frame")
swanlab.log({"video": video})
```

传入 5D 数组 `(B, T, C, H, W)` 时，`B` 个样本会自动拼接为网格画面（如 `B=4` 时为 `2×2` 网格），画面尺寸随之放大：

```python
# 4 个样本，每个 32 帧 64x64
batch = np.random.randint(0, 256, (4, 32, 3, 64, 64), dtype=np.uint8)
video = swanlab.Video(batch, format="mp4", fps=8)
swanlab.log({"video": video})
```

### 从 PyTorch Tensor 创建

`swanlab.Video` 通过 `.numpy()` 方法读取视频帧，传入 Tensor 前需先 `detach`、转到 CPU 并转为 `uint8`：

```python
# tensor: (T, 3, H, W) 的 float 张量，取值 0~1
frames = (tensor * 255).clamp(0, 255).to(torch.uint8).detach().cpu()
video = swanlab.Video(frames, format="mp4", fps=8)
swanlab.log({"video": video})
```

### 复用 Video 对象

传入已有的 `swanlab.Video` 实例可直接复用其数据、格式和 `caption`：

```python
base = swanlab.Video("path/to/video.mp4", caption="raw video")
copy = swanlab.Video(base)
```

### 单步记录多个视频

```python
import swanlab

swanlab.init()

videos = [swanlab.Video(f"video-{i}.mp4", caption=f"video-{i}") for i in range(3)]
swanlab.log({"examples": videos})
```

## 注意事项

1. 入参为数组时，画面单边上限为 `4096` 像素（批量拼接后按总宽高计算，如 `B=4` 时宽高各翻倍）。MP4 宽高会自动补齐为偶数，OGG 自动补齐为 16 的倍数。
2. 单个视频默认上限 `50 MiB`，可通过 [`core.media_max_size`](./py-settings.md#核心行为配置-coresettings) 调整，超出范围限制的视频会被跳过并告警。
3. 传入文件时校验扩展名，传入二进制数据时校验文件头（Magic Bytes），格式不匹配或不受支持会直接报错。
