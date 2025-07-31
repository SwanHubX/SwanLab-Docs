# swanlab.Video

[Github Source Code](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/modules/video/__init__.py)

```python
Video(
    data_or_path: str,
    caption: str = None,
) -> None
```

| Parameter    | Description                                                                                            |
|--------------|--------------------------------------------------------------------------------------------------------|
| data_or_path | (str) Accepts a video file path. Currently only supports GIF format files. |
| caption      | (str) The label for the video. Used to mark the video when displayed in the experiment dashboard.       |

## Introduction

Convert video data to be recorded by `swanlab.log()`. Currently only supports GIF format video files.

::: warning Format Limitation
Currently `swanlab.Video` only supports GIF format file paths. Other video formats are not supported yet.
:::

### Logging GIF Video Files

Logging a single GIF video:

```python
import swanlab

swanlab.init()
video = swanlab.Video("path/to/video.gif", caption="Training Process Demo")
swanlab.log({"video": video})
```

Logging multiple GIF videos:

```python
import swanlab

swanlab.init()

examples = []
for i in range(3):
    video = swanlab.Video(f"video_{i}.gif", caption=f"Video Example {i}")
    examples.append(video)

swanlab.log({"examples": examples})
```

## Creating GIF Video Example

Here's a complete example of creating a GIF animation and logging it:

```python
import os.path
import random
from PIL import Image as PILImage
from PIL import ImageDraw
import swanlab

swanlab.init()

# Create a GIF animation
def create_mock_gif(output_path, width=200, height=200, frames=10, duration=100):
    """
    Create a simple mock GIF animation
    
    Parameters:
        output_path: Output GIF file path
        width: Image width (pixels)
        height: Image height (pixels)
        frames: Number of animation frames
        duration: Display time per frame (milliseconds)
    """
    images = []
    
    for i in range(frames):
        # Create a new RGB image
        img = PILImage.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Generate random colors
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        
        # Draw a random circle on the image
        x = random.randint(0, width)
        y = random.randint(0, height)
        radius = random.randint(10, min(width, height) // 2)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(r, g, b))
        
        # Add current frame to the list
        images.append(img)
    
    # Save as GIF animation
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)

# Create GIF file
gif_path = "test.gif"
create_mock_gif(gif_path, width=300, height=300, frames=15, duration=200)

# Log to SwanLab
swanlab.log({"video": swanlab.Video(gif_path, caption="This is a test video")})
```

## Use Cases

`swanlab.Video` is suitable for the following scenarios:

1. **Training Process Visualization**: Record dynamic changes during model training
2. **Experiment Result Display**: Show dynamic demonstrations of experiment results
3. **Data Visualization**: Convert time series data into animated displays
4. **Model Behavior Analysis**: Record dynamic responses of models under different inputs

## Notes

1. **File Format**: Currently only supports GIF format, does not support other video formats like MP4, AVI, etc.
2. **File Size**: It's recommended to control GIF file size, as overly large files may affect loading speed
3. **Frame Rate Control**: When creating GIFs, you can control playback speed through the `duration` parameter
4. **File Path**: Ensure the provided file path is correct and the file exists
