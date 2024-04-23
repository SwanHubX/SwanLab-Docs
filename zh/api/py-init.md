# swanlab.init

```python
init(
    project: str = None,
    experiment_name: str = None,
    description: str = None,
    config: dict = None,
    logdir: str = None,
    suffix: str = "default",
    log_level: str = None,
)
```

| 参数         | 描述 |
|-------------|------|
| project |(str)项目名。 |
| experiment_name | (str) 实验名称, 如果不指定则取"exp"。完整的实验名称由 `experiment_name+"_"+suffix` 构成。 |
| description   | (str) 实验描述, 如果不指定默认为None                                   |
| config       | (dict) 实验配置，在此处可以记录一些实验的超参数等信息。                   |
| logdir       | (str) 日志文件存储路径，默认为swanlog                                  |
| suffix       | (str) experiment_name的后缀。完整的实验名称由`experiment_name`和`suffix`共同构成。默认值为"default"，代表默认的后缀规则为`'%b%d-%h-%m-%s' (example:'Feb03_14-45-37')` |
| log_level    | (str) 打印日志等级。                                                |

## 介绍

- 在机器学习训练流程中，我们可以将`swandb.init()`添加到训练脚本和测试脚本的开头，SwanLab将跟踪机器学习流程的每个环节。

- `swanlab.init()`会生成一个新的后台进程来将数据记录到实验中，默认情况下，它还会将数据同步到swanlab.pro，以便你可以在线实时看到可视化结果。

- 在使用`swanlab.log()`记录数据之前，需要先调用`swanlab.init()`：

```python
import swanlab

swanlab.init()
swanlab.log({"loss": 0.1846})
```

- 调用`swanlab.init()`会返回一个`SwanLabRun`类型的对象，同样可以执行`log`操作：

```python
import swanlab

run = swanlab.init()
run.log({"loss": 0.1846})
```

- 在脚本运行结束时，我们将自动调用`swanlab.finish`来结束SwanLab实验。但是，如果从子进程调用`swanlab.init()`，如在jupyter notebook中，则必须在子进程结束时显式调用`swanlab.finish`。

```python
import swanlab

swanlab.init()
swanlab.finish()
```


## 更多用法

### 设置项目、实验名、描述

```python
swanlab.init(
    project="cats-detection",
    experiment_name="YoloX-baseline",
    description="YoloX检测模型的基线实验，主要用于后续对比。",
)
```


### 设置日志文件保存位置

下面的代码展示了如何将日志文件保存到自定义的目录下：

```python
swanlab.init(
    logdir="path/to/my_custom_dir"
)
```

### 将实验相关的元数据添加到实验配置中

```python
swanlab.init(
    config={
        "learning-rate": 1e-4,
        "model": "CNN",
    }
)

```