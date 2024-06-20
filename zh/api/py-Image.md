# swanlab.Image

[Github源代码](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/modules/image.py)

```python
Image(
    data_or_path: Union[str, np.ndarray, PILImage.Image],
    mode: str = "RGB",
    caption: str = None,
    file_type: str = None,
    size: Union[int, list, tuple] = None,
) -> None
```

| 参数        | 描述                                                                                                                                                                   |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| data_or_path | (Union[str, np.ndarray, PILImage.Image]) 接收图像文件路径、numpy数组、或者PIL图像。Image类将判断接收的数据类型做相应的转换。                                      |
| mode      | (str) 图像的 PIL 模式。最常见的是 "L"、"RGB"、"RGBA"。完整解释请参阅：[Pillow mode](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes)                         |
| caption   | (str) 图像的标签。用于在实验看板中展示图像时进行标记。                                                                                                                 |
| file_type | (str) 设置图片的格式，可选['png', 'jpg', 'jpeg', 'bmp']，默认为'png'                                                                                                   |
| size      | (Union[int, list, tuple]) 设置图像的尺寸，默认保持原图尺寸。如果size设置为int类型，如512，将根据最长边不超过512的标准做图像缩放, [size更多用法](#对传入图像做resize)|


## 介绍

对各种类型的图像数据做转换，以被`swanlab.log()`记录。

![](/assets/media-image-1.jpg)

### 从numpy array创建

记录单张图像：

```python
import numpy as np
import swanlab

run = swanlab.init()

# 1. 创建一个numpy array
random_image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
# 2. 传入swanlab.Image
image = swanlab.Image(random_image, caption="random image")

run.log({"examples": image})
```

记录多张图像：

```python
import numpy as np
import swanlab

run = swanlab.init()

# 创建一个列表
examples = []
for i in range(3):
    random_image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    image = swanlab.Image(random_image, caption="random image")
    # 列表中添加swanlab.Image类型对象
    examples.append(image)

# 记录图列
run.log({"examples": examples})
```

### 从PyTorch Tensor创建

`swanlab.Image`支持传入尺寸为[B, C, H, W]与[C, H, W]的Tensor。

```python
import torch
import swanlab

run = swanlab.init()
···
for batch, ground_truth in train_dataloader():
    # 假设batch是尺寸为[16, 3, 256, 256]的tensor
    tensors = swanlab.Image(batch)
    run.log({"examples": tensors})
```


### 从PIL Image创建

```python
import numpy as np
from PIL import Image
import swanlab

run = swanlab.init()

# 创建一个列表
examples = []
for i in range(3):
    random_image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    pil_image = Image.fromarray(random_image)
    image = swanlab.Image(pil_image, caption="random image")
    examples.append(image)

run.log({"examples": examples})
```

### 从文件路径创建

```python
import swanlab

run = swanlab.init()
image = swanlab.Image("path/to/file", caption="random image")

run.log({"examples": image})
```

`swanlab.Image`在默认情况下，是以`png`的格式做图像转换与存储。

如果想要用`jpg`格式：

```python{3}
image = swanlab.Image("path/to/file",
                      caption="random image",
                      file_type="jpg")
```

### 对传入图像做Resize

在默认情况，`swanlab.Image`不对图像做任何尺寸缩放。  

如果需要放缩图像，我们可以通过设置`size`参数，来调节图像尺寸。

放缩规则为：  

1. 默认: 不对图像做任何缩放

2. `size`为int类型: 如果最长边超过`size`, 则将最长边设为`size`, 另一边等比例缩放; 否则不缩放

3. `size`为list/tuple类型: 

    - (int, int): 将图像缩放到宽为size[0], 高为size[1]
    - (int, None): 将图像缩放到宽为size[0], 高等比例缩放
    - (None, int): 将缩放缩放到高为size[1], 宽等比例缩放

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

### 记录Matplotlib图表

```python
import swanlab
import matplotlib.pyplot as plt

# 定义横纵坐标的数据
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# plt创建折线图
plt.plot(x, y)

# 添加标题和标签
plt.title("Examples")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

swanlab.init()

# 记录plt
swanlab.log({"example": swanlab.Image(plt)})
```