# swanlab.Settings

```python
Settings(
    model_config = SettingsConfigDict(frozen=True),
    interactive: bool = True,
    mode: Literal["disabled", "local", "online", "offline"] = "online",
    log_dir: DirectoryPath = Field(default_factory=log_dir_factory),
    api_key: Optional[str] = Field(default=None),
    api_host: str = Field(default="https://api.swanlab.cn"),
    web_host: str = Field(default="https://swanlab.cn"),
    project: ProjectSettings = Field(default_factory=ProjectSettings),
    experiment: ExperimentSettings = Field(default_factory=ExperimentSettings),
    run: RunSettings = Field(default_factory=RunSettings),
    terminal: TerminalSettings = Field(default_factory=TerminalSettings),
    integration: IntegrationSettings = Field(default_factory=IntegrationSettings),
    core: CoreSettings = Field(default_factory=CoreSettings),
    probe: ProbeSettings = Field(default_factory=ProbeSettings),
)
```

| 参数          | 类型          | 描述                                                                                                                                                                                |
| :------------ | :------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `interactive` | StrictBool    | 是否开启交互模式。关闭后，所有用户输入提示和相关交互将被禁用。适用于 CI/CD 环境或后台批量任务。默认值为 `True`。                                                                    |
| `mode`        | Literal       | SwanLab 运行模式。可选值：`disabled`（禁用 SwanLab）、`local`（本地运行）、`online`（云端同步，`cloud` 为其别名）、`offline`（离线模式）。**注意：大小写敏感**。默认值为 `online`。 |
| `log_dir`     | DirectoryPath | SwanLab 日志文件保存的目录。默认值为当前运行目录下的 `swanlog` 文件夹。                                                                                                             |
| `api_key`     | str           | SwanLab 服务的 API 密钥。                                                                                                                                                           |
| `api_host`    | str           | SwanLab API 服务的基础 URL。默认值为 `https://api.swanlab.cn`。                                                                                                                     |
| `web_host`    | str           | SwanLab Web 服务的基础 URL，仅用于展示，不影响 SDK 行为。默认值为 `https://swanlab.cn`。                                                                                            |

## 嵌套配置

### 项目配置 `ProjectSettings`

通过 `project` 字段传入，例如 `Settings(project=ProjectSettings(workspace="my-ws"))`。

| 参数        | 类型       | 描述                                                                              |
| :---------- | :--------- | :-------------------------------------------------------------------------------- |
| `name`      | str        | 项目名称。                                                                        |
| `workspace` | str        | 项目所属的工作空间名称。                                                          |
| `public`    | StrictBool | 是否将实验设为公开。可选值为 `true`、`yes`、`1`，留空则为私有。默认值为 `False`。 |

### 实验配置 `ExperimentSettings`

通过 `experiment` 字段传入，例如 `Settings(experiment=ExperimentSettings(tags=["tag1", "tag2"]))`。

| 参数          | 类型      | 描述                                                                                           |
| :------------ | :-------- | :--------------------------------------------------------------------------------------------- |
| `name`        | str       | 实验名称。                                                                                     |
| `color`       | str       | 实验展示颜色，支持预设颜色名称、RGB 字符串或十六进制颜色码。                                   |
| `description` | str       | 实验描述。                                                                                     |
| `tags`        | List[str] | 实验标签列表。最多 50 个标签。也可传入逗号分隔的字符串（如 `"tag1,tag2"`）或 JSON 数组字符串。 |
| `group`       | str       | 实验分组名称，用于将实验分组以便管理和区分。                                                   |
| `job_type`    | str       | 实验任务类型，用于标识当前实验的任务类型（如分类、回归等）。                                   |

### 运行配置 `RunSettings`

通过 `run` 字段传入，例如 `Settings(run=RunSettings(resume="allow"))`。

| 参数                 | 类型    | 描述                                                                                                                                                                                            |
| :------------------- | :------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                 | str     | 实验运行 ID。留空则自动生成。                                                                                                                                                                   |
| `resume`             | Literal | 断点续训策略。可选值：`must`（必须续训，找不到历史记录则报错）、`allow`（允许续训）、`never`（不允许续训）。布尔值也可接受（`True` 映射为 `allow`，`False` 映射为 `never`）。默认值为 `never`。 |
| `parallel`           | Literal | 并行执行策略。可选值：`none`（默认）、`shared`（共享运行目录）。设置为 `shared` 时，`resume` 会被自动强制为 `allow`。                                                                           |
| `history`            | Path    | 历史记录路径，用于从之前的运行中恢复。                                                                                                                                                          |
| `config`             | Path    | 配置文件路径或字典，用于加载实验配置。                                                                                                                                                          |
| `dir`                | str     | 自定义运行文件夹名称。设置后将跳过目录冲突重试，直接创建该目录。                                                                                                                                |
| `dir_max_length`     | int     | 自动生成的运行文件夹名称的最大长度。范围为 50-255，默认值为 `255`。                                                                                                                             |
| `dir_create_retries` | int     | 创建唯一运行文件夹的最大重试次数。当生成目录名与已有目录冲突时，会在短暂延迟后重新生成，最多重试此次数。最小值为 1，默认值为 `10`。                                                             |

### 终端日志配置 `TerminalSettings`

通过 `terminal` 字段传入，例如 `Settings(terminal=TerminalSettings(proxy_type="stdout"))`。

| 参数         | 类型    | 描述                                                                                                                                                            |
| :----------- | :------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `proxy_type` | Literal | 终端日志代理策略。可选值：`all`（代理标准输出和标准错误流）、`stdout`（仅代理标准输出流）、`stderr`（仅代理标准错误流）、`none`（不代理日志）。默认值为 `all`。 |
| `max_length` | int     | 单行终端日志的最大字符数。范围为 500-4096，默认值为 `1024`。                                                                                                    |

### 集成配置 `IntegrationSettings`

通过 `integration` 字段传入，用于配置 Webhook 和本地看板。

#### Webhook `WebhookSettings`

| 参数      | 类型 | 描述                                       |
| :-------- | :--- | :----------------------------------------- |
| `url`     | str  | Webhook 通知地址。                         |
| `value`   | str  | Webhook 回调结构体传递的 value 值。        |
| `timeout` | int  | Webhook 请求超时时间（秒）。默认值为 `5`。 |

#### 本地看板 `DashBoardSettings`

| 参数   | 类型 | 描述                                                        |
| :----- | :--- | :---------------------------------------------------------- |
| `host` | str  | 本地 SwanLab 看板服务绑定的主机地址。默认值为 `127.0.0.1`。 |
| `port` | int  | 本地 SwanLab 看板服务的端口号。默认值为 `5092`。            |

### 核心行为配置 `CoreSettings`

通过 `core` 字段传入，用于控制上传和保存的核心行为。

| 参数              | 类型  | 描述                                                                                                                                                                            |
| :---------------- | :---- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `section_rule`    | int   | 指标 key 的分段规则，指定用第几个 `/` 分割 section 和 metric 名：`0` 表示第一个 `/`，`1` 表示第二个，`-1` 表示最后一个。超出范围时通过 `idx % slash_count` 回绕。默认值为 `0`。 |
| `record_batch`    | int   | 单次 HTTP 请求上传的记录条数上限。范围为 1-100000，默认值为 `10000`。                                                                                                           |
| `record_interval` | float | 上传线程的批量上传间隔（秒）。默认值为 `5.0`。在 Converter 等高吞吐场景下可调小（如 `0.5`）。                                                                                   |
| `save_split`      | int   | 分片上传的文件大小阈值（字节）。默认值为 `100 MiB`（104857600 字节）。                                                                                                          |
| `save_size`       | int   | 单文件最大保存大小（字节）。默认值为 `50 GiB`（53687091200 字节）。                                                                                                             |
| `save_part`       | int   | 分片上传的每个分片大小（字节）。默认值为 `32 MiB`（33554432 字节）。                                                                                                            |
| `save_batch`      | int   | 每次保存上传的最大文件数量。默认值为 `100`。                                                                                                                                    |

### 探针配置 `ProbeSettings`

通过 `probe` 字段传入，控制 SwanLab 在启动时收集哪些环境信息。

| 参数               | 类型          | 描述                                                                                                                                                                                                                                                                                                |
| :----------------- | :------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hardware`         | StrictBool    | 是否收集静态硬件元数据（如 GPU 型号、CPU 核心数、总内存等）并在启动时上报。**注意**：如果 `monitor` 为 `True` 而 `hardware` 为 `False`，监控模块仍会访问硬件信息来计算动态指标（如利用率百分比），但静态硬件快照本身会被丢弃，既不会包含在遥测数据中，也不会保存到本地持久化存储。默认值为 `True`。 |
| `runtime`          | StrictBool    | 是否收集软件运行环境信息。开启后会记录操作系统、Python 版本、主机名、当前工作目录和启动脚本的精确命令。默认值为 `True`。                                                                                                                                                                            |
| `requirements`     | StrictBool    | 是否收集 Python 依赖项快照。开启后会记录已安装的 pip 包及其精确版本（类似 `pip freeze`），以确保环境可复现。默认值为 `True`。                                                                                                                                                                       |
| `conda`            | StrictBool    | 是否收集 Conda 环境配置。开启后会记录当前活跃的 Conda 环境详情和导出的依赖。默认关闭以避免额外的启动开销。默认值为 `False`。                                                                                                                                                                        |
| `git`              | StrictBool    | 是否收集 Git 仓库元数据。开启后会记录当前分支、最新 commit 哈希和远程 URL，帮助将实验运行与代码的特定版本紧密关联。默认值为 `True`。                                                                                                                                                                |
| `swanlab`          | StrictBool    | 是否收集 SwanLab 自身元数据。开启后会记录 SwanLab 版本、运行目录等信息。默认值为 `True`。                                                                                                                                                                                                           |
| `monitor`          | StrictBool    | 是否启用周期性的硬件监控（CPU 使用率、GPU 利用率、内存等）。默认值为 `True`。                                                                                                                                                                                                                       |
| `monitor_interval` | int           | 周期性硬件监控的采集间隔（秒）。最小值为 5，默认值为 `10`。                                                                                                                                                                                                                                         |
| `monitor_disk_dir` | DirectoryPath | 磁盘 I/O 监控的基准目录，用于计算磁盘使用率。默认为系统根目录（Linux/macOS 为 `/`，Windows 为 `C:\`）。                                                                                                                                                                                             |

## 介绍

- `swanlab.Settings` 类用于管理 SwanLab 的全局功能开关和设置。

- 在 `import swanlab` 时，会创建一个默认的全局设置，各个设置及其默认值详见上表。

- 如果要对某些设置进行调整，需要新建一个 `Settings` 实例（如 `new_settings`），在实例化时传入想要修改的配置参数，然后通过运行 `swanlab.merge_settings(new_settings)` 来更新全局设置。

- 值得注意的是，`merge_settings()` 方法只在 `swanlab.init()` 被调用之前可用。这意味着，在使用 `swanlab` 的过程中，一旦 `swanlab.init()` 被调用，全局设置将不再能被更改。

## 更多用法

### 更新全局设置

::: code-group

```python [方式一]
import swanlab

# 创建新的设置对象
new_settings = swanlab.Settings(
    interactive=False,
    mode="offline",
    probe=swanlab.Settings.Probe(
        hardware=False,
        monitor=False,
        monitor_interval=5,
    ),
    core=swanlab.Settings.Core(
        record_batch=5000,
        record_interval=2.0,
    ),
)

swanlab.init(settings=new_settings)
...
```

```python [方式二]
import swanlab

# 创建新的设置对象
new_settings = swanlab.Settings(
    interactive=False,
    mode="offline",
    probe=swanlab.Settings.Probe(
        hardware=False,
        monitor=False,
        monitor_interval=5,
    ),
    core=swanlab.Settings.Core(
        record_batch=5000,
        record_interval=2.0,
    ),
)

# 更新全局设置
swanlab.merge_settings(new_settings)

swanlab.init()
...
```

:::

### 禁用交互模式

适用于 CI/CD 环境或后台批量任务：

```python
import swanlab
from swanlab import Settings

# 创建新的设置对象
new_settings = Settings(
    interactive=False,
)

# 更新全局设置
swanlab.merge_settings(new_settings)

swanlab.init()
...
```

### 配置探针

```python
import swanlab
from swanlab import Settings

# 创建新的设置对象
new_settings = Settings(
    probe=Settings.Probe(
        hardware=False,     # 不收集静态硬件信息
        conda=True,         # 收集 Conda 环境配置
        git=False,          # 不收集 Git 信息
        monitor=True,       # 启用硬件监控
        monitor_interval=5, # 监控间隔 5 秒
        monitor_disk_dir="/data",  # 磁盘监控目录
    ),
)

# 更新全局设置
swanlab.merge_settings(new_settings)

swanlab.init()
...
```
