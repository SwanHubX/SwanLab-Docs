# swanlab.Video

[Github源代码](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/modules/video/__init__.py)

```python
Video(
    data_or_path: str,
    caption: str = None,
) -> None
```

| 参数          | 描述                                                                                                     |
|-------------|--------------------------------------------------------------------------------------------------------|
| data_or_path | (str) 接收视频文件路径。目前仅支持GIF格式文件。 |
| caption     | (str) 视频的标签。用于在实验看板中展示视频时进行标记。                                                      |

## 介绍

对视频数据做转换，以被`swanlab.log()`记录。目前仅支持GIF格式的视频文件。

::: warning 格式限制
目前 `swanlab.Video` 仅支持 GIF 格式的文件路径。其他视频格式暂不支持。
:::

### 记录GIF视频文件

记录单个GIF视频：

```python
import swanlab

swanlab.init()
video = swanlab.Video("path/to/video.gif", caption="训练过程演示")
swanlab.log({"video": video})
```

记录多个GIF视频：

```python
import swanlab

swanlab.init()

examples = []
for i in range(3):
    video = swanlab.Video(f"video_{i}.gif", caption=f"视频示例 {i}")
    examples.append(video)

swanlab.log({"examples": examples})
```

## 创建GIF视频示例

以下是一个创建GIF动画并记录的完整示例：

```python
import os.path
import random
from PIL import Image as PILImage
from PIL import ImageDraw
import swanlab

swanlab.init()

# 创建一个GIF动画
def create_mock_gif(output_path, width=200, height=200, frames=10, duration=100):
    """
    创建一个简单的mock GIF动画
    
    参数:
        output_path: 输出GIF文件路径
        width: 图像宽度(像素)
        height: 图像高度(像素)
        frames: 动画帧数
        duration: 每帧显示时间(毫秒)
    """
    images = []
    
    for i in range(frames):
        # 创建一个新的RGB图像
        img = PILImage.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # 随机生成颜色
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        
        # 在图像上绘制一个随机的圆形
        x = random.randint(0, width)
        y = random.randint(0, height)
        radius = random.randint(10, min(width, height) // 2)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(r, g, b))
        
        # 将当前帧添加到列表中
        images.append(img)
    
    # 保存为GIF动画
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)

# 创建GIF文件
gif_path = "test.gif"
create_mock_gif(gif_path, width=300, height=300, frames=15, duration=200)

# 记录到SwanLab
swanlab.log({"video": swanlab.Video(gif_path, caption="这是一个测试视频")})
```

## 使用场景

`swanlab.Video` 适用于以下场景：

1. **训练过程可视化**：记录模型训练过程中的动态变化
2. **实验结果展示**：展示实验结果的动态演示
3. **数据可视化**：将时间序列数据转换为动画展示
4. **模型行为分析**：记录模型在不同输入下的动态响应

## 注意事项

1. **文件格式**：目前仅支持 GIF 格式，不支持 MP4、AVI 等其他视频格式
2. **文件大小**：建议控制 GIF 文件大小，过大的文件可能影响加载速度
3. **帧率控制**：创建 GIF 时可以通过 `duration` 参数控制播放速度
4. **文件路径**：确保提供的文件路径正确且文件存在