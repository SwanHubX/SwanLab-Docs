# swanlab.init

```python
init(
    project: str = None,
    workspace: str = None,
    experiment_name: str = None,
    description: str = None,
    job_type: str = None,
    group: str = None,
    tags: List[str] = None,
    config: Union[dict, str] = None,
    logdir: str = None,
    mode: MODES = None,
    load: str = None,
    public: bool = None,
    callbacks: List[SwanKitCallback] = None,
    settings: Settings = None,
    id: str = None,
    resume: Union[Literal['must', 'allow', 'never'], bool] = None,
    reinit: bool = None,
    **kwargs,
)
```

| 参数         | 描述 |
|-------------|------|
| project |(str)项目名，如果不指定则取运行目录的名称。|
| workspace |(str)工作空间，默认将实验同步到你的个人空间下，如果要上传到组织，则填写组织的username。|
| experiment_name | (str) 实验名称, 如果不指定则取"swan-1"这样的`动物名+序号`作为实验名。 |
| job_type | (str) 任务类型，默认为空。 |
| group | (str) 实验分组，默认为空。 |
| tags       | (list) 实验标签。可以传入多个字符串组成的列表，标签会显示在实验顶部的标签栏。|
| description   | (str) 实验描述, 如果不指定默认为None。                                   |
| config       | (dict, str) 实验配置，在此处可以记录一些实验的超参数等信息。支持传入配置文件路径，支持yaml和json文件。                   |
| logdir       | (str) 离线看板日志文件存储路径，默认为`swanlog `。                                 |
| mode       | (str) 设置swanlab实验创建的模式，可选"cloud"、"local"、"offline"、"disabled"，默认设置为"cloud"。<br>`cloud`：将实验上传到云端。（公有云和私有化部署）<br>`offline`：仅将实验数据保存到本地。<br>`local`：不上传到云端，但会记录实验数据和一些可被`swanlab watch`打开的数据到本地。<br>`disabled`：不上传也不记录。|
| load       | (str) 加载的配置文件路径，支持yaml和json文件。|
| public       | (bool) 设置使用代码直接创建SwanLab项目的可见性，默认为False即私有。|
| callbacks       | (list) 设置实验回调函数，支持`swanlab.toolkit.callback.SwanKitCallback`的子类。|
| name       | (str) 与experiment_name效果一致，优先级低于experiment_name。|
| notes       | (str) 与description效果一致，优先级低于description。|
| settings       | (dict) 实验配置。支持传入1个`swanlab.Settings`对象。|
| id       | (str) 上次实验的运行ID，用于恢复上次实验。ID必须为21位字符串。|
| resume       | (str) 断点续训模式，可选True、False、"must"、"allow"、"never"，默认取None。<br>`True`： 效果同`resume="allow"`。<br>`False`：效果同`resume="never"`。<br>`must`：你必须传递 `id` 参数，并且实验必须存在。<br>`allow`：如果存在实验，则会resume该实验，否则将创建新的实验。<br>`never`：你不能传递 `id` 参数，将会创建一个新的实验。(即不开启resume的效果)|
| reinit       | (bool) 是否重新创建实验，如果为True，则每次调用`swanlab.init()`时，会把上一次实验`finish`掉；默认取None。|

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

### 设置标签

```python
swanlab.init(
    tags=["yolo", "detection", "baseline"]
)
```

### 设置日志文件保存位置

下面的代码展示了如何将日志文件保存到自定义的目录下：

```python
swanlab.init(
    logdir="path/to/my_custom_dir",
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

关于插件的更多信息，请参考[插件](/plugin/plugin-index.md)。

```python
from swanlab.plugin.notification import EmailCallback

email_callback = EmailCallback(...)

swanlab.init(
    callbacks=[email_callback]
)
```

### 断点续训

断点续训的意思是，如果你之前有一个状态为`完成`或`中断`的实验，需要补一些实验数据，那么你可以通过`resume`和`id`参数来恢复这个实验。

```python
swanlab.init(
    resume=True,
    id="14pk4qbyav4toobziszli",  # id必须为21位字符串
)
```

实验id可以在实验的「环境」选项卡或URL中找到，必须为1个21位字符串。


:::tip resume使用场景

1. 之前的训练进程断了，基于checkpoint继续训练时，希望实验图表能和之前的swanlab实验续上，而非创建1个新swanlab实验
2. 训练和评估分为了两个进程，但希望评估和训练记录在同一个swanlab实验中
3. config中有一些参数填写有误，希望更新config参数

:::

:::warning ⚠️注意

1. 由项目克隆产生的实验，不能被resume

:::


断点续训可以选择三种模式：

1. `allow`：如果项目下存在`id`对应的实验，则会resume该实验，否则将创建新的实验。
2. `must`：如果项目下存在`id`对应的实验，则会resume该实验，否则将报错
3. `never`：不能传递 `id` 参数，将会创建一个新的实验。(即不开启resume的效果)

::: info
`resume=True` 效果同 `resume="allow"`。<br>
`resume=False` 效果同 `resume="never"`。
:::

测试代码：

```python
import swanlab

run = swanlab.init()
swanlab.log({"loss": 2, "acc":0.4})
run.finish()

run = swanlab.init(resume=True, id=run.id)
swanlab.log({"loss": 0.2, "acc": 0.9})
```


## 过期参数

- `cloud`：在v0.3.4被`mode`参数取代。参数仍然可用，且会覆盖掉`mode`的设置。
