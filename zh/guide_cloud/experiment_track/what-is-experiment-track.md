# 实验跟踪

**实验跟踪（Experiment Tracking）** 是指在机器学习模型开发过程中，记录每个实验的元数据（超参数、配置项）和指标（loss、acc...），并进行组织和呈现的过程。可以理解为在进行机器学习实验时，记录下实验的各个关键信息。

实验跟踪的目的是帮助研究人员更有效地管理和分析实验结果，以便更好地理解模型性能的变化，进而优化模型开发过程。它的作用包括:

1. **细节和可重复性**: 记录实验细节并方便重复实验。
2. **比较**: 可以比较不同实验结果，分析哪些变化导致了性能提升。
3. **团队协作**: 方便团队成员之间共享和比较实验结果，提高协作效率。 

![](/assets/swanlab-overview.png)

**SwanLab**帮助你只需使用几行代码，便可以跟踪机器学习实验，并在交互式仪表板中查看与比较结果。上图显示了一个示例仪表板，您可以在其中查看和比较多次实验的指标，分析导致性能变化的关键要素。


## SwanLab是如何进行实验跟踪的？

通过SwanLab,使用几行代码跟踪机器学习实验。跟踪流程：

1. 创建SwanLab实验。
2. 将超参数字典（例如学习率或模型类型）存储到您的配置中 (swanlab.config)。
3. 在训练循环中随时间记录指标 (swanlab.log)，例如准确性acc和损失loss。

下面的伪代码演示了常见的**SwanLab实验跟踪工作流**：

```python
# 1. 创建1个SwanLab实验
swanlab.init(project="my-project-name")

# 2. 存储模型的输入或超参数
swanlab.config.learning_rate = 0.01

# 这里写模型的训练代码
...

# 3. 记录随时间变化的指标以可视化表现
swanlab.log({"loss": loss})
```

## 如何开始？

探索以下资源以了解SwanLab实验跟踪：

- 阅读[快速开始](/zh/guide_cloud/general/quick-start)
- 探索本章以了解如何：
  - [创建一个实验](/zh/guide_cloud/experiment_track/create-experiment)
  - [配置实验](/zh/guide_cloud/experiment_track/set-experiment-config.md)
  - [记录指标](/zh/guide_cloud/experiment_track/log-experiment-metric.md)
  - [查看实验结果](/zh/guide_cloud/experiment_track/view-result.md)
- 在[API文档](/zh/api/api-index)中探索SwanLab Python 库。