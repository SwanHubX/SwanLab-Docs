# 记录媒体数据

SwanLab 支持记录媒体数据（图像、音频、文本、三维点云等）以直观地探索你的实验结果，实现模型的主观评估。

## 1.图像

`swanlab.Image` 支持记录多种图像类型，包括 numpy、PIL、Tensor、读取文件等。[API文档](/api/py-Image)。

![](/assets/media-image-1.jpg)

### 1.1 记录 Array 型图像

Array型包括numpy和tensor。直接将 Array 传入 `swanlab.Image`，它将根据类型自动做相应处理：

- 如果是 `numpy.ndarray`：SwanLab 会使用 pillow (PIL) 对其进行读取 。
- 如果是 `tensor`：SwanLab 会使用 `torchvision` 的 `make_grid`函数做转换，然后使用 pillow 对其进行读取。

示例代码：

```python
image = swanlab.Image(image_array, caption="左图: 输入, 右图: 输出")
swanlab.log({"examples": image})
```

### 1.2 记录 PIL 型图像

直接传入 `swanlab.Image`：

```python
image = PIL.Image.fromarray(image_array)
swanlab.log({"examples": image})
```

### 1.3 记录文件图像

提供文件路径给 `swanlab.Image`：

```python
image = swanlab.Image("myimage.jpg")
swanlab.log({"example": image})
```

### 1.4 记录 Matplotlib

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

### 1.5 单步记录多个图像

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

关于图像的更多细节，可参考[API文档](/api/py-Image)

## 2. 音频

[API文档](/api/py-Audio)

![](/assets/media-audio-1.jpg)

### 2.1 记录 Array 型音频

```python
audio = swanlab.Audio(np_array, sample_rate=44100, caption="white_noise")
swanlab.log({"white_noise": audio})
```

### 2.2 记录音频文件

```python
swanlab.log({"white_noise": swanlab.Audio("white_noise.wav")})
```

### 2.3 单步记录多个音频

```python
examples = []
for i in range(3):
    white_noise = np.random.randn(100000)
    audio = swanlab.Audio(white_noise, caption="audio_{i}")
    # 列表中添加swanlab.Audio类型对象
    examples.append(audio)

run.log({"examples": examples})
```

## 3. 文本

[API文档](/api/py-Text)

### 3.1 记录字符串

```python
swanlab.log({"text": swanlab.Text("A example text.")})
```

### 3.2 单步记录多个文本

```python
# 创建一个空列表
text_list = []
for i in range(3):
    text = swanlab.Text("A example text.", caption=f"{i}")
    text_list.append(text)

swanlab.log({"examples": text_list})
```

![alt text](/assets/log-media-text.png)

## 4. 3D点云

![](/zh/api/py-object3d/demo.png)

请参考此文档：[API-Oject3D](/api/py-object3d)

## 5. 生物化学分子

![](/assets/molecule.gif)

请参考此文档：[API-Molecule](/api/py-molecule)

## 6. 视频

请参考此文档：[API-Video](/api/py-Video)

## Q&A

### 1. caption参数有什么作用？

每一个媒体类型都会有1个`caption`参数，它的作用是对该媒体数据的文字描述，比如对于图像：

```python
apple_image = swanlab.Image(data, caption="苹果")
swanlab.log({"im": apple_image})
```
<img src="/assets/log-media-image.png" width=400, height=400>


### 2. 想要媒体数据和epoch数同步，怎么办？

在用swanlab.log记录媒体数据时，指定`step`参数为epoch数即可。

```python
for epoch in epochs:
    ···
    swanlab.log({"im": sw_image}, step=epoch)
```