# swanlab.init

```python
init(
    project: str = None,
    workspace: str = None,
    experiment_name: str = None,
    description: str = None,
    config: Union[dict, str] = None,
    logdir: str = None,
    mode: str = "cloud",
    load: str = None,
    public: bool = None,
    callbacks: list = None,
    **kwargs,
)
```

| 参数         | 描述 |
|-------------|------|
| project |(str)项目名，如果不指定则取运行目录的名称。|
| workspace |(str)工作空间，默认将实验同步到你的个人空间下，如果要上传到组织，则填写组织的username。|
| experiment_name | (str) 实验名称, 如果不指定则取"swan-1"这样的`动物名+序号`作为实验名。 |
| description   | (str) 实验描述, 如果不指定默认为None。                                   |
| config       | (dict, str) 实验配置，在此处可以记录一些实验的超参数等信息。支持传入配置文件路径，支持yaml和json文件。                   |
| logdir       | (str) 离线看板日志文件存储路径，默认为`swanlog `。                                 |
| mode       | (str) 设置swanlab实验创建的模式，可选"cloud"、"local"、"disabled"，默认设置为"cloud"。<br>`cloud`：将实验上传到云端。（公有云和私有化部署）<br>`local`：不上传到云端，但会记录实验信息到本地。<br>`disabled`：不上传也不记录。|
| load       | (str) 加载的配置文件路径，支持yaml和json文件。|
| public       | (bool) 设置使用代码直接创建SwanLab项目的可见性，默认为False即私有。|
| callbacks       | (list) 设置实验回调函数，支持`swankit.callback.SwanKitCallback`的子类。|
| name       | (str) 与experiment_name效果一致，优先级低于experiment_name。|
| notes       | (str) 与description效果一致，优先级低于description。|

## 介绍

- 在机器学习训练流程中，我们可以将`swandb.init()`添加到训练脚本和测试脚本的开头，SwanLab将跟踪机器学习流程的每个环节。

- `swanlab.init()`会生成一个新的后台进程来将数据记录到实验中，默认情况下，它还会将数据同步到swanlab.cn，以便你可以在线实时看到可视化结果。

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

> 仅在mode="local"时有效

下面的代码展示了如何将日志文件保存到自定义的目录下：

```python
swanlab.init(
    logdir="path/to/my_custom_dir",
    mode="local",
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

### 上传到组织

```python
swanlab.init(
    workspace="[组织的username]"
)
```

### 插件

关于插件的更多信息，请参考[插件](/zh/plugin/plugin-index.md)。

```python
from swanlab.plugin.notification import EmailCallback

email_callback = EmailCallback(...)

swanlab.init(
    callbacks=[email_callback]
)
```



## 过期参数

- `cloud`：在v0.3.4被`mode`参数取代。参数仍然可用，且会覆盖掉`mode`的设置。
