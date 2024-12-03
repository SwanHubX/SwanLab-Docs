# swanlab.Image

[Github Source Code](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/modules/image.py)

```python
Image(
    data_or_path: Union[str, np.ndarray, PILImage.Image],
    mode: str = "RGB",
    caption: str = None,
    file_type: str = None,
    size: Union[int, list, tuple] = None,
) -> None
```

| Parameter    | Description                                                                                                                                                        |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| data_or_path | (Union[str, np.ndarray, PILImage.Image]) Accepts an image file path, numpy array, or PIL image. The Image class will determine the received data type and perform the appropriate conversion. |
| mode         | (str) The PIL mode of the image. The most common are "L", "RGB", "RGBA". For a complete explanation, see: [Pillow mode](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes) |
| caption      | (str) The label for the image. Used to mark the image when displayed in the experiment dashboard.                                                                  |
| file_type    | (str) Set the image format, options include ['png', 'jpg', 'jpeg', 'bmp'], default is 'png'.                                                                       |
| size         | (Union[int, list, tuple]) Set the image size, default is to keep the original size. If size is set to an int type, such as 512, the image will be scaled based on the longest side not exceeding 512. [More usage for size](#resize-the-input-image) |

## Introduction

Convert various types of image data to be recorded by `swanlab.log()`.

![](/assets/media-image-1.jpg)

### Creating from numpy array

Logging a single image:

```python
import numpy as np
import swanlab

run = swanlab.init()

# 1. Create a numpy array
random_image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
# 2. Pass it to swanlab.Image
image = swanlab.Image(random_image, caption="random image")

run.log({"examples": image})
```

Logging multiple images:

```python
import numpy as np
import swanlab

run = swanlab.init()

# Create a list
examples = []
for i in range(3):
    random_image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    image = swanlab.Image(random_image, caption="random image")
    # Add swanlab.Image type objects to the list
    examples.append(image)

# Log the list of images
run.log({"examples": examples})
```

### Creating from PyTorch Tensor

`swanlab.Image` supports tensors with dimensions [B, C, H, W] and [C, H, W].

```python
import torch
import swanlab

run = swanlab.init()
···
for batch, ground_truth in train_dataloader():
    # Assume batch is a tensor with dimensions [16, 3, 256, 256]
    tensors = swanlab.Image(batch)
    run.log({"examples": tensors})
```

### Creating from PIL Image

```python
import numpy as np
from PIL import Image
import swanlab

run = swanlab.init()

# Create a list
examples = []
for i in range(3):
    random_image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    pil_image = Image.fromarray(random_image)
    image = swanlab.Image(pil_image, caption="random image")
    examples.append(image)

run.log({"examples": examples})
```

### Creating from file path

```python
import swanlab

run = swanlab.init()
image = swanlab.Image("path/to/file", caption="random image")

run.log({"examples": image})
```

By default, `swanlab.Image` converts and stores images in `png` format.

If you want to use `jpg` format:

```python{3}
image = swanlab.Image("path/to/file",
                      caption="random image",
                      file_type="jpg")
```

### Resize the Input Image

By default, `swanlab.Image` does not resize the image.  

If you need to resize the image, you can adjust the image size by setting the `size` parameter.

The resizing rules are:  

1. Default: No resizing of the image.

2. `size` as an int type: If the longest side exceeds `size`, set the longest side to `size` and scale the other side proportionally; otherwise, no resizing.

3. `size` as a list/tuple type:

    - (int, int): Resize the image to width `size[0]` and height `size[1]`.
    - (int, None): Resize the image to width `size[0]` and scale the height proportionally.
    - (None, int): Resize the image to height `size[1]` and scale the width proportionally.

```python
print(im_array.shape)
# [1024, 512, 3]

im1 = swanlab.Image(im_array, size=512)
# [512, 256, 3]

im2 = swanlab.Image(im_array, size=(512, 512))
# [512, 512, 3]

im3 = swanlab.Image(im_array, size=(None, 1024))
# [2048, 1024, 3]

im4 = swanlab.Image(im_array, size=(256, None))
# [256, 128, 3]
```

### Logging Matplotlib Plots

```python
import swanlab
import matplotlib.pyplot as plt

# Define the data for the x and y axes
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Create a line plot with plt
plt.plot(x, y)

# Add title and labels
plt.title("Examples")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

swanlab.init()

# Log the plt plot
swanlab.log({"example": swanlab.Image(plt)})
```