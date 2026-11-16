# swanlab.Video

[Github Source Code](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/sdk/internal/run/transforms/video/__init__.py)

```python
Video(
    data_or_path: Union[str, Path, bytes, BytesIO, np.ndarray, Video],
    caption: str = None,
    fps: int = None,
    format: Literal["gif", "mp4", "webm", "ogg"] = None,
) -> None
```

| Parameter    | Description                                                                                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| data_or_path | (Union[str, Path, bytes, BytesIO, np.ndarray, Video]) Accepts a video file path, binary data (`bytes` / `BytesIO`), a `numpy` array, or a `swanlab.Video` instance. |
| caption      | (str, optional) Video caption displayed below the video in the dashboard.                                                                                           |
| fps          | (int, optional) Frame rate (frames per second) when encoding an array into a video. Defaults to `4`. Ignored for file paths, binary data, or `Video` instances.     |
| format       | (str, optional) Video format: `"gif"`, `"mp4"`, `"webm"`, or `"ogg"`. Inferred from the file extension for file paths; required for arrays and non-GIF binary data. |

## Introduction

<video controls src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/videos/swan-video-show-v2.mp4" />

Wraps video data to be logged via `swanlab.log()`. Supports **GIF**, **MP4**, **WebM**, and **OGG**:

- **Video files and binary data**: Stored as-is, preserving the original quality and audio track.
- **NumPy arrays / Tensors**: Encoded locally into a silent video in the specified format.

::: warning Dependencies
**⚠️ Note**: Passing a `numpy` array or Tensor requires the media extras: `pip install "swanlab[media]"`
:::

### From a File Path

Accepts a path string or `pathlib.Path`. The format is inferred from the file extension (`.gif`, `.mp4`, `.webm`, `.ogg`):

```python
import swanlab

swanlab.init()

video = swanlab.Video("path/to/video.mp4", caption="video example")
swanlab.log({"video": video})
```

### From Binary Data

Accepts `bytes` or `io.BytesIO`. GIF is detected automatically; other formats require `format` to be specified explicitly:

```python
import swanlab

with open("path/to/video.webm", "rb") as f:
    video = swanlab.Video(f.read(), format="webm", caption="video example")

swanlab.init()
swanlab.log({"video": video})
```

### From a NumPy Array

Accepts a 4D array `(T, C, H, W)` or a 5D batch array `(B, T, C, H, W)`, where `B` is batch size, `T` is frame count, `C` is channels (`1`, `3`, or `4`), and `H`/`W` are height and width:

- `dtype` must be `uint8` (`0~255`); convert other dtypes beforehand.
- 4-channel (RGBA) inputs drop the alpha channel and are encoded as RGB.
- `format` is required; `fps` defaults to `4`.

```python
import numpy as np
import swanlab

swanlab.init()

# 64-frame 128x128 RGB video
frames = np.random.randint(0, 256, (64, 3, 128, 128), dtype=np.uint8)
video = swanlab.Video(frames, format="gif", fps=8, caption="random frame")
swanlab.log({"video": video})
```

When a 5D array `(B, T, C, H, W)` is passed, the `B` samples are tiled into a single grid frame (e.g., a `2×2` grid when `B=4`), scaling the output resolution accordingly:

```python
# 4 samples, each 32 frames at 64x64
batch = np.random.randint(0, 256, (4, 32, 3, 64, 64), dtype=np.uint8)
video = swanlab.Video(batch, format="mp4", fps=8)
swanlab.log({"video": video})
```

### From a PyTorch Tensor

`swanlab.Video` reads frames via the object's `.numpy()` method. Detach the Tensor, move it to CPU, and cast it to `uint8` before passing it in:

```python
# tensor: float tensor of shape (T, 3, H, W) with values in 0~1
frames = (tensor * 255).clamp(0, 255).to(torch.uint8).detach().cpu()
video = swanlab.Video(frames, format="mp4", fps=8)
swanlab.log({"video": video})
```

### Reusing a Video Instance

Pass an existing `swanlab.Video` instance to reuse its underlying data, format, and `caption`:

```python
base = swanlab.Video("path/to/video.mp4", caption="raw video")
copy = swanlab.Video(base)
```

### Logging Multiple Videos in One Step

```python
import swanlab

swanlab.init()

videos = [swanlab.Video(f"video-{i}.mp4", caption=f"video-{i}") for i in range(3)]
swanlab.log({"examples": videos})
```

## Notes

1. For array inputs, the maximum resolution per side is `4096` pixels (measured after grid tiling for batched inputs, e.g., width and height double when `B=4`). Frame dimensions are automatically padded to multiples of 2 for MP4 and multiples of 16 for OGG.
2. The default size limit per video is `50 MiB` (configurable via [`core.media_max_size`](./py-settings.md#core-behavior-configuration-coresettings)). Videos exceeding the limit are skipped with a warning.
3. File paths are validated by extension, and binary inputs are validated by file header (magic bytes). Mismatched or unsupported formats raise an error.
