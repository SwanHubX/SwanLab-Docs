
# 🚀快速开始

安装 SwanLab 并在几分钟内开始跟踪你的人工智能实验。

::: info 提示

如果你想在本地或无网络环境下查看实验，请看[自托管部分](/zh/guide_cloud/self_host/offline-board.md)。

:::

## 1. 创建账号并安装SwanLab

在开始之前，请确保你创建一个帐户并安装 SwanLab：

1. 在 [SwanLab官网](https://dev101.swanlab.cn) 免费注册账号，然后登录你的SwanLab账户。
2. 使用 pip 在 Python3 环境的计算机上安装swanlab库

打开命令行，输入：

```bash
pip install swanlab
```

## 2. 登录到SwanLab

下一步，你需要在你的编程环境上登录SwanLab。

打开命令行，输入：

```bash
swanlab login
```

出现如下提示时，复制粘贴您的[API Key](https://dev101.swanlab.cn/settings)：

```bash
swanlab: Logging into swanlab cloud.
swanlab: You can find your API key at: https://dev101.swanlab.cn/settings
swanlab: Paste an API key from your profile and hit enter, or press 'CTRL-C' to quit:
```

输入完成后，按下回车，完成登录。

## 3. 开启一个实验并跟踪超参数

在Python脚本中，我们用`swanlab.init`创建一个SwanLab实验，并向`config`参数传递将一个包含超参数键值对的字典：

```python
run = swanlab.init(
    # 设置项目
    project="my-project",
    # 跟踪超参数与实验元数据
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```

`run`是SwanLab的基本组成部分，你将经常使用它来记录与跟踪实验指标。

## 4. 记录实验指标

在Python脚本中，用`swanlab.log`记录实验指标（比如准确率acc和损失值loss）。

用法是将一个包含指标的字典传递给`swanlab.log`：

```python
swanlab.log({"accuracy": acc, "loss": loss})
```

## 5. 完整代码，在线查看可视化看板

我们将上面的步骤整合为下面所示的完整代码：

```python
import swanlab
import random

# 登陆SwanLab
swanlab.login()

# 初始化SwanLab
run = swanlab.init(
    # 设置项目
    project="my-project",
    # 跟踪超参数与实验元数据
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    },
)

print(f"学习率为{run.config.learning_rate}")

offset = random.random() / 5

# 模拟一次训练过程
for epoch in range(2, run.config.epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    # 记录指标
    swanlab.log({"accuracy": acc, "loss": loss})
```

运行代码，访问[SwanLab网站](swanlab.pro），查看在每个训练步骤中，你使用SwanLab记录的指标（准确率和损失值）的改进情况。

## 下一步是什么

1. 查看SwanLab如何[记录多媒体内容](/zh/guide_cloud/experiment_track/log-media)（图片、音频、文本、...）
2. 查看如何通过SwanLab与团队协作

## 常见问题

### 在哪里可以找到我的API Key？

登陆SwanLab网站后，API Key将显示在[用户设置]页面上。

### 我可以离线使用SwanLab吗？

可以，具体流程请查看[自托管部分](/zh/guide_cloud/self_host/offline-board.md)。