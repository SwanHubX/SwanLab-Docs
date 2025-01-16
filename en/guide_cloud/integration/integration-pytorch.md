# PyTorch

[![](/assets/colab.svg)](https://colab.research.google.com/drive/1RWsrY_1bS8ECzaHvYtLb_1eBkkdzekR3?usp=sharing)

Among academic researchers, [PyTorch](https://pytorch.org/) is the most popular Python deep learning framework.

![PyTorch](/assets/ig-pytorch.png)

You can use PyTorch to train deep learning models while using SwanLab for experiment tracking and visualization.

::: warning Integration with Other PyTorch Ecosystems

- [Lightning](/en/guide_cloud/integration/integration-pytorch-lightning.md)
- [Torchtune](/en/guide_cloud/integration/integration-pytorch-torchtune.md)

:::

## Log Tensor Images

You can pass PyTorch `Tensors` with image data to `swanlab.Image`, and `swanlab.Image` will use `torchvision` to convert them into images:

```python
image_tensors = ...  # Tensor images with shape [B, C, H, W]
swanlab.log({"examples": [swanlab.Image(im) for im in image_tensors]})
```