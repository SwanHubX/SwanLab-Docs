
# 🚀快速开始

安装 SwanLab 并在几分钟内开始跟踪你的人工智能实验。

![quick-start-1](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/zh/guide_cloud/general/quick_start/quick-start.png)

## 1. 安装SwanLab

使用 [pip](https://pip.pypa.io/en/stable/) 在Python3环境的计算机上安装swanlab库。

打开命令行，输入：

```bash
pip install swanlab
```

按下回车，等待片刻完成安装。

> 如果遇到安装速度慢的问题，可以指定国内源安装：  
> `pip install swanlab -i https://mirrors.cernet.edu.cn/pypi/web/simple`

## 2. 登录账号

> 如果你还没有SwanLab账号，请在 [官网](https://swanlab.cn) 免费注册。

打开命令行，输入：

```bash
swanlab login
```

当你看到如下提示时：

```bash
swanlab: Logging into swanlab cloud.
swanlab: You can find your API key at: https://swanlab.cn/settings
swanlab: Paste an API key from your profile and hit enter, or press 'CTRL-C' to quit:
```

在[用户设置](https://swanlab.cn/settings)页面复制您的 **API Key**，粘贴后按下回车（你不会看到粘贴后的API Key，请放心这是正常的），即可完成登录。之后无需再次登录。

::: info

如果你的电脑不太适合命令行粘贴API Key（比如一些Windows CMD）的方式登录，可以使用：

```shell
swanlab login -k your-api-key
```

亦可使用python脚本登录：

```python
import swanlab
swanlab.login(api_key="你的API Key", save=True)
```

若要在Kaggle等Notebook环境下使用Swanlab，参见[用 Notebook 跟踪实验](/guide_cloud/experiment_track/jupyter-notebook.md)

:::

## 3. 开启一个实验并跟踪超参数

在Python脚本中，我们用`swanlab.init`创建一个SwanLab实验，并向`config`参数传递将一个包含超参数键值对的字典：

```python
import swanlab

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

```python (5,25)
import swanlab
import random

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

# 模拟训练过程
for epoch in range(2, run.config.epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    # 记录指标
    swanlab.log({"accuracy": acc, "loss": loss})
```

运行代码，访问[SwanLab](https://swanlab.cn)，查看在每个训练步骤中，你使用SwanLab记录的指标（准确率和损失值）的改进情况。

![quick-start-1](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/zh/guide_cloud/general/quick_start/line-chart.png)

## 下一步是什么

1. 查看SwanLab如何[记录多媒体内容](/guide_cloud/experiment_track/log-media)（图片、音频、文本、...）
2. 查看SwanLab记录[MNIST手写体识别](/examples/mnist.md)的案例
3. 查看与其他框架的[集成](/guide_cloud/integration/index.md)
4. 查看如何通过SwanLab与[团队协作](/guide_cloud/general/organization.md)

## 常见问题

### 1. 在哪里可以找到我的API Key？

登陆SwanLab网站后，API Key将显示在[用户设置](https://swanlab.cn/settings)页面上。

### 2. 我可以离线使用SwanLab吗？

可以，具体流程请查看[自托管部分](/self_host/docker/deploy.md)。
