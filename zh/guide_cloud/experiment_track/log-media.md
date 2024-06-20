# 记录多媒体数据

SwanLab 支持记录多媒体数据（图像、音频、文本等）以直观地探索你的实验结果，实现模型的主观评估。

## 图片

`swanlab.Image` 支持记录多种图片类型，包括 numpy、PIL、Tensor、读取文件等。[API文档](/zh/api/py-Image)。

![](/assets/media-image-1.jpg)

::: warning
建议每个 step 记录少于 50 个图像，以防止日志记录成为训练期间的耗时瓶颈，以及图像加载成为查看结果时的耗时瓶颈。
:::

### 记录 Array 型图片

Array型包括numpy和tensor。直接将 Array 传入 `swanlab.Image`，它将根据类型自动做相应处理：

- 如果是 `numpy.ndarray`：SwanLab 会使用 pillow (PIL) 对其进行读取 。
- 如果是 `tensor`：SwanLab 会使用 `torchvision` 的 `make_grid`函数做转换，然后使用 pillow 对其进行读取。

示例代码：

```python
image = swanlab.Image(image_array, caption="左图: 输入, 右图: 输出")
swanlab.log({"examples": image})
```

### 记录 PIL 型图片

直接传入 `swanlab.Image`：

```python
image = PIL.Image.fromarray(image_array)
swanlab.log({"examples": image})
```

### 记录文件图片

提供文件路径给 `swanlab.Image`：

```python
image = swanlab.Image("myimage.jpg")
swanlab.log({"example": image})
```

### 记录 Matplotlib

将 `matplotlib.pyplot` 的 `plt` 对象传入 `swanlab.Image`：

```python
import matplotlib.pyplot as plt

# 数据
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
# 创建折线图
plt.plot(x, y)
# 添加标题和标签
plt.title("Examples")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

swanlab.log({"example": swanlab.Image(plt)})
```

### 单步记录多个图像

单步记录多个图像即在一次 `swanlab.log` 中，传递一个由 `swanlab.Image` 类型对象组成的列表。

```python
# 创建一个空列表
image_list = []
for i in range(3):
    random_image = np.random.randint(low=0, high=256, size=(100, 100, 3))
    image = swanlab.Image(random_image, caption=f"随机图像{i}")
    # 将 swanlab.Image 类型对象添加到列表中
    image_list.append(image)

swanlab.log({"examples": image_list})
```

关于图像的更多细节，可参考[API文档](/zh/api/py-Image)

## 音频

[API文档](/zh/api/py-Audio)

![](/assets/media-audio-1.jpg)

### 记录 Array 型音频

```python
audio = swanlab.Audio(np_array, sample_rate=44100, caption="white_noise")
swanlab.log({"white_noise": audio})
```

### 记录音频文件

```python
swanlab.log({"white_noise": swanlab.Audio("white_noise.wav")})
```

### 单步记录多个音频

```python
examples = []
for i in range(3):
    white_noise = np.random.randn(100000)
    audio = swanlab.Audio(white_noise, caption="audio_{i}")
    # 列表中添加swanlab.Audio类型对象
    examples.append(audio)

run.log({"examples": examples})
```

## 文本

[API文档](/zh/api/py-Text)

### 记录字符串

```python
swanlab.log({"text": swanlab.Text("A example text.")})
```

### 单步记录多个文本

```python
# 创建一个空列表
image_list = []
for i in range(3):
    text = swanlab.Text("A example text.", caption=f"{i}")
    text_list.append(text)

swanlab.log({"examples": text_list})
```