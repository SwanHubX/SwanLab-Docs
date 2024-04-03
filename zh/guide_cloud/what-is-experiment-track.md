# AI实验跟踪

使用几行代码跟踪人工智能实验，然后你可以在交互式仪表板中查看与比较结果。  
下图显示了一个示例仪表板，您可以在其中查看和比较多次实验的指标。

![](/assets/swanlab-overview.png)

## 它是如何工作的？

使用几行代码跟踪机器学习实验：

1. 创建SwanLab实验。
2. 将超参数字典（例如学习率或模型类型）存储到您的配置中 ( swanlab.config)。
3. 在训练循环中随时间记录指标 ( swanlab.log())，例如准确性acc和损失loss。

接下来的伪代码演示了常见的SwanLab实验跟踪工作流：

```python
# 1. 创建1个SwanLab实验
swanlab.init(organization="my-organization", project="my-project-name")

# 2. 存储模型的输入或超参数
swanlab.config.learning_rate = 0.01

# 这里写模型的训练代码

# 3. 记录随时间变化的指标以可视化表现
swanlab.log({"loss": loss})
```

## 如何开始？

探索以下资源以了解SwanLab实验跟踪：

- 阅读[快速开始](/zh/guide_cloud/quick-start)
- 探索本章以了解如何：
  - [创建一个实验](/zh/guide_cloud/create-experiment)
  - 配置实验
  - 记录实验数据
  - 查看实验结果
- 在[API文档](/zh/api/api-index)中探索SwanLab Python 库。