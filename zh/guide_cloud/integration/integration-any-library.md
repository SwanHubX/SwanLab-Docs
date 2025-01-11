# 将SwanLab集成到你的库

本指南提供了如何将W&B集成到您的Python库中的最佳实践，以获得强大的实验跟踪、GPU和系统监控、模型检查点等功能。

下面我们将介绍您正在处理的代码库比单个 Python 训练脚本或 Jupyter 笔记本更复杂时的最佳提示和最佳实践。

## Requirements

在开始之前，请决定是否在您的库的依赖项中要求 W&B：

### 安装时需要SwanLab

```plaintext
torch==2.5.0
...
swanlab==0.4.*
```

### 将swanlab设置为可选安装

