# PyTorch

在学术研究者当中，[PyTorch](https://pytorch.org/) 是最流行的 Python 深度学习框架。  

![PyTorch](/assets/ig-pytorch.png)

你可以使用PyTorch进行深度学习模型训练，同时使用SwanLab进行实验跟踪与可视化。

::: warning Pytorch生态的其他集成

- [Lightning](/zh/guide_cloud/integration/integration-pytorch-lightning.md)

:::


## 记录Tensor图像

你可以将带有图像数据的PyTorch `Tensors`传递给`swanlab.Image`，`swanlab.Image`将使用`torchvision`把它们转换成图像：

```python
image_tensors = ...  # shape为[B, C, H, W]的Tensor图像
swanlab.log({"examples": [swanlab.Image(im) for im in image_tensors]})
```