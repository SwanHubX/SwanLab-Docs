# swanlab.Settings

```python
Settings(
    model_config = ConfigDict(frozen=True),
    metadata_collect: StrictBool = True,
    collect_hardware: StrictBool = True,
    collect_runtime: StrictBool = True,
    security_mask: StrictBool = True,
    requirements_collect: StrictBool = True,
    conda_collect: StrictBool = False,
    hardware_monitor: StrictBool = True,
    disk_io_dir: DirectoryPath = Field(...),
    upload_interval: PositiveInt = 1,
    max_log_length: int = Field(ge=500, le=4096, default=1024),
    log_proxy_type: Literal["all", "stdout", "stderr", "none"] = "all",
)
```

| 参数                     | 类型            | 描述                                                                              |
|:-----------------------|:--------------|:--------------------------------------------------------------------------------|
| `metadata_collect`     | StrictBool    | 是否开启元数据采集。默认值为 `True`。                                                          |
| `collect_hardware`     | StrictBool    | 是否采集当前系统环境的硬件信息。默认值为 `True`。                                                    |
| `collect_runtime`      | StrictBool    | 是否采集运行时信息。默认值为 `True`。                                                          |
| `security_mask`        | StrictBool    | 是否自动隐藏隐私信息，如 api_key 等。开启后将在检测到隐私信息时，自动将其替换为加密字符（****）。默认值为 `True`。             |
| `requirements_collect` | StrictBool    | 是否采集 Python 环境信息 (`pip list`)。默认值为 `True`。                                      |
| `conda_collect`        | StrictBool    | 是否采集 Conda 环境信息。默认值为 `False`。                                                   |
| `hardware_monitor`     | StrictBool    | 是否开启硬件监控。如果 `metadata_collect` 关闭，则此项无效。默认值为 `True`。                            |
| `disk_io_dir`          | DirectoryPath | 磁盘 IO 监控的路径。默认值为系统根目录 (`/` 或 `C:\`)。                                            |
| `hardware_interval`    | PositiveInt   | 硬件监控采集间隔，以秒为单位，最小值为5秒。                                                          |
| `backup`               | PositiveInt   | 日志备份开启功能，默认值为 `True`。开启后，日志将被备份到本地（默认为`swanlog`目录）。      |
| `upload_interval`      | PositiveInt   | 日志上传间隔（单位：秒）。默认值为 `1`。                                                          |
| `max_log_length`       | int           | 终端日志上传单行最大字符数（范围：500-4096）。默认值为 `1024`。                                         |
| `log_proxy_type`       | Literal       | 日志代理类型，会影响实验的日志选项卡记录的内容。默认值为 `"all"`。"stdout" 表示只代理标准输出流，"stderr" 表示只代理标准错误流，"all" 表示代理标准输出流和标准错误流，"none" 表示不代理日志。|

## 介绍

- `swanlab.Settings`类用于和管理 SwanLab 的全局功能开关和设置。

- 在`import swanlab`时，会创建一个默认的全局设置，各个设置及其默认值详见上表。

- 如果我们要对某些设置进行调整，需要通过新建一个`Settings`实例如`new_settings`，在实例化时传入想要修改的配置参数，然后要通过运行`swanlab.merge_settings(new_settings)`来对全局设置进行更新。

- 值得注意的是，`merge_settings()`方法只在`swanlab.init()`被调用之前可用，这意味着，在使用`swanlab`的过程中，一旦`swanlab.init()`被调用，全局设置将不再能被更改。

## 更多用法

### 更新全局设置

::: code-group

```python [方式一]
import swanlab

# 创建新的设置对象
new_settings = swanlab.Settings(
    metadata_collect=False,
    hardware_monitor=False,
    upload_interval=5
)

swanlab.init(settings=new_settings)
...
```

```python [方式二]
import swanlab

# 创建新的设置对象
new_settings = swanlab.Settings(
    metadata_collect=False,
    hardware_monitor=False,
    upload_interval=5
)

# 更新全局设置
swanlab.merge_settings(new_settings)

swanlab.init()
...
```

:::

### 记录 conda 环境信息

```python
import swanlab
from swanlab import Settings

# 创建新的设置对象
new_settings = Settings(
    conda_collect=True  # 默认不开启
)

# 更新全局设置
swanlab.merge_settings(new_settings)

swanlab.init()
...
```
