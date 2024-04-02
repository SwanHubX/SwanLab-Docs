# 创建一个实验

使用 SwanLab Python SDK 跟踪人工智能实验，然后您可以在交互式仪表板中查看结果。  
本节将介绍如何创建一个SwanLab实验。

## 1. 如何创建一个SwanLab实验?

创建一个SwanLab实验分为3步：
1. 初始化SwanLab
2. 传递一个超参数字典
3. 在你的训练循环中记录指标

### 初始化SwanLab

`swanlab.init()`的作用是初始化一个SwanLab实验，它将启动后台进程以同步和记录数据。  
下面的代码片段展示了如何创建一个名为 **cat-dog-classification** 的新SwanLab项目。并为其添加了：

1. **project**：项目名。
1. **experiment**：实验名。实验名为当前实验的标识，以帮助您识别此实验。  
2. **description**：描述。描述是对实验的详细介绍。

```python
# 导入SwanLab Python库
import swanlab

# 1. 开启一个SwanLab实验
run = swanlab.init(
    project="cat-dog-classification",
    experiment_name="Resnet50",
    description="我的第一个人工智能实验",
)
```

当你初始化SwanLab时，`swanlab.init()`将返回一个对象。  
此外，SwanLab会创建一个本地目录（默认名称为“swanlog”），所有日志和文件都保存在其中，并异步传输到 SwanLab 服务器。（该目录也可以被`swanlab watch -l [logdir]`命令打开本地实验看板。）

::: info
**注意**：如果调用 `swanlab.init` 时该项目已存在，则实验会添加到预先存在的项目中。  
例如，如果您已经有一个名为`"cat-dog-classification"`的项目，那么新的实验会添加到该项目中。
:::

### 传递超参数字典

传递超参数字典，例如学习率或模型类型。  
你在`config`中传入的字典将被保存并用于后续的实验对比与结果查询。

```python
# 2. 传递一个超参数字典
swanlab.config={"epochs": 20, "learning_rate": 1e-4, "batch_size": 32, "model_type": "CNN"}
```

有关如何配置实验的更多信息，请参阅[设置实验配置]()。