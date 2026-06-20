# 环境变量

[⚙️完整环境变量配置 -> Github](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/sdk/internal/settings/__init__.py)

## 调试配置

| 环境变量        | 描述                                                                    | 默认值  |
| --------------- | ----------------------------------------------------------------------- | ------- |
| `SWANLAB_DEBUG` | 是否开启调试模式。开启后会在终端打印 debug 信息，并同步写入诊断日志文件 | `false` |

## 全局配置

| 环境变量           | 描述                                                                                                                                           | 默认值                           |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `SWANLAB_ROOT`     | SwanLab 全局文件夹保存的路径，是 `SWANLAB_SAVE_DIR` 的新版环境变量                                                                             | 用户主目录下的 `.swanlab` 文件夹 |
| `SWANLAB_PUBLIC`   | 是否将实验设为公开，可选值为 `true`、`yes`、`1`，留空则为私有                                                                                  | 私有                             |
| `SWANLAB_LOGDIR`   | SwanLab 解析日志文件保存的路径                                                                                                                 | 当前运行目录的 `swanlog` 文件夹  |
| `SWANLAB_MODE`     | SwanLab 的解析模式，涉及操作员注册的回调。可选值：`local`、`online`（`cloud` 为 `online` 的别名）、`offline`、`disabled`。**注意：大小写敏感** | `online`                         |
| `SWANLAB_API_HOST` | SwanLab 云端环境的 API 地址                                                                                                                    | `https://api.swanlab.cn`         |
| `SWANLAB_WEB_HOST` | SwanLab 云端环境的 Web 地址。私有化部署时仅需设置此变量，无需设置 `SWANLAB_API_HOST`                                                           | `https://swanlab.cn`             |

## 实验配置

| 环境变量                     | 描述                                                                                                           |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `SWANLAB_PROJ_NAME`          | 项目名称，效果等价于 `swanlab.init(project="...")`                                                             |
| `SWANLAB_WORKSPACE`          | 工作空间名称，效果等价于 `swanlab.init(workspace="...")`                                                       |
| `SWANLAB_EXP_NAME`           | 实验名称，效果等价于 `swanlab.init(experiment_name="...")`                                                     |
| `SWANLAB_RUN_ID`             | 实验运行ID，效果等价于 `swanlab.init(id="...")`                                                                |
| `SWANLAB_RESUME`             | 是否断点续训，效果等价于 `swanlab.init(resume=...)`，可选值为 `must`、`allow`、`never`                         |
| `SWANLAB_DESCRIPTION`        | 实验描述，效果等价于 `swanlab.init(description="...")`                                                         |
| `SWANLAB_TAGS`               | 实验标签，效果等价于 `swanlab.init(tags=[...])`，如果你想要添加多个tags，写法为`SWANLAB_TAGS="tag1,tag2,tag3"` |
| `SWANLAB_GROUP`              | 实验分组，用于将实验分组以便管理和区分                                                                         |
| `SWANLAB_JOB_TYPE`           | 实验任务类型，用于标识当前实验的任务类型（如分类、回归等）                                                     |
| `SWANLAB_EXP_COLOR`          | 实验颜色，效果等价于 `swanlab.init(color="...")`，支持预设颜色名称、RGB字符串或十六进制颜色码                  |
| `SWANLAB_RUN_PARALLEL`       | 并行模式，效果等价于 `swanlab.init(parallel="...")`，可选值为 `shared`                                         |
| `SWANLAB_RUN_DIR`            | 自定义运行文件夹名称，设置后将跳过目录冲突重试，默认自动生成                                                   |
| `SWANLAB_RUN_DIR_MAX_LENGTH` | 自动生成的运行文件夹名称的最大长度，默认 `255`                                                                 |

## 登录认证

| 环境变量          | 描述                                                                                                                                                                                                            |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SWANLAB_API_KEY` | 云端 API Key。登录时会首先查找此环境变量，如果不存在，判断用户是否已登录，未登录则进入登录流程。<br>- 如果 `login` 接口传入字符串，此环境变量无效<br>- 如果用户已登录，此环境变量的优先级高于本地存储的登录信息 |

## 行为控制

| 环境变量              | 描述                                                                         |
| --------------------- | ---------------------------------------------------------------------------- |
| `SWANLAB_DISABLE_GIT` | 是否禁用Git，可选值为 `True`、`False`，当设置为 `True` 时，将不会记录Git信息 |

## 其他

| 环境变量                  | 描述                                                                                                                   |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `SWANLAB_WEBHOOK`         | Webhook 地址。<br> SwanLab 初始化完毕时，如果此环境变量存在，会调用此地址发送消息                                      |
| `SWANLAB_WEBHOOK_VALUE`   | Webhook回调结构体传递的value。<br> 当`SWANLAB_WEBHOOK`存在时，如果此环境变量存在，会作为Webhook回调结构体的value值发送 |
| `SWANLAB_WEBHOOK_TIMEOUT` | Webhook 请求超时时间（秒），默认为 `5`                                                                                 |

## 探针配置

通过 `probe` 参数（如 `swanlab.init(probe={"hardware": False})`）或以下环境变量控制 SwanLab 在启动时收集的环境信息类型。所有探针选项均接受 `true`/`false`、`1`/`0`、`yes`/`no`。

| 环境变量                         | 描述                                                          | 默认值     |
| -------------------------------- | ------------------------------------------------------------- | ---------- |
| `SWANLAB_PROBE_HARDWARE`         | 是否收集静态硬件信息（GPU 型号、CPU 核心数、总内存等）        | `true`     |
| `SWANLAB_PROBE_RUNTIME`          | 是否收集软件运行环境信息（操作系统、Python 版本、启动命令等） | `true`     |
| `SWANLAB_PROBE_REQUIREMENTS`     | 是否收集 Python 依赖项列表（类似 `pip freeze`）               | `true`     |
| `SWANLAB_PROBE_CONDA`            | 是否收集 Conda 环境配置，默认关闭以避免额外的启动开销         | `false`    |
| `SWANLAB_PROBE_GIT`              | 是否收集 Git 仓库信息（当前分支、最新 commit、远程 URL）      | `true`     |
| `SWANLAB_PROBE_SWANLAB`          | 是否收集 SwanLab 自身信息（版本、运行目录等）                 | `true`     |
| `SWANLAB_PROBE_MONITOR`          | 是否启用周期性的硬件监控（CPU 使用率、GPU 利用率、内存等）    | `true`     |
| `SWANLAB_PROBE_MONITOR_INTERVAL` | 周期性硬件监控的采集间隔（秒）                                | `10`       |
| `SWANLAB_PROBE_MONITOR_DISK_DIR` | 磁盘 I/O 监控的基准目录，用于计算磁盘使用率                   | 系统根目录 |

## 本地看板配置

| 环境变量                 | 描述                                | 默认值      |
| ------------------------ | ----------------------------------- | ----------- |
| `SWANLAB_DASHBOARD_HOST` | 本地 SwanLab 看板服务绑定的主机地址 | `127.0.0.1` |
| `SWANLAB_DASHBOARD_PORT` | 本地 SwanLab 看板服务的端口号       | `5092`      |

## 日志配置

| 环境变量            | 描述                                                                                                                        | 默认值 |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------ |
| `SWANLAB_LOG_LEVEL` | SwanLab 日志的输出级别，控制日志详细程度。可选值：`debug`（最详细）、`info`（默认）、`warning`、`error`、`critical`（最少） | `info` |

## 核心行为

| 环境变量                               | 描述                                                                                                   | 默认值  |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------- |
| `SWANLAB_SKIP_SWANBOARD_VERSION_CHECK` | 是否跳过 swanboard 版本兼容性检查，设置为 `1` 可跳过                                                   | 不跳过  |
| `SWANLAB_FS_TIMEOUT`                   | 文件系统操作超时时间（秒），适用于 NAS 等异步延迟较高的存储环境                                        | `5.0`   |
| `SWANLAB_CORE_SECTION_RULE`            | 指标 key 的分段规则，指定用第几个 `/` 分割 section 和 metric 名：`0`=第一个，`1`=第二个，`-1`=最后一个 | `0`     |
| `SWANLAB_CORE_RECORD_BATCH`            | 单次 HTTP 请求上传的记录条数上限                                                                       | `10000` |
| `SWANLAB_TERMINAL_PROXY_TYPE`          | 终端日志代理策略：`all`（全部）、`stdout`、`stderr`、`none`（不收集）                                  | `all`   |
| `SWANLAB_TERMINAL_MAX_LENGTH`          | 单行终端日志的最大字符长度                                                                             | `1024`  |

## 高级配置

| 环境变量              | 描述                                                                        | 默认值         |
| --------------------- | --------------------------------------------------------------------------- | -------------- |
| `SWANLAB_SECRETS_DIR` | K8s / Docker 容器中 Secret 配置文件所在目录，用于注入敏感信息（如 API Key） | 无             |
| `SWANLAB_CONFIG_DIR`  | 全局配置文件目录路径，SwanLab 会从此目录读取 `*.yaml` / `*.yml` 配置文件    | `/etc/swanlab` |
