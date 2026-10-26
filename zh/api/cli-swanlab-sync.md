# swanlab sync

```bash
swanlab sync [options] [logdir]
```

| 选项                | 描述                                                                                                             |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `-k`, `--api-key`   | 用于身份验证的API密钥。如果未指定，将使用环境中的默认API密钥。如果指定，将使用此API密钥登录但不会保存密钥。      |
| `-h`, `--host`      | 同步日志的主机地址。如果未指定，将使用默认主机(`https://swanlab.cn`)。                                           |
| `-w`, `--workspace` | 同步日志的工作空间。如果未指定，将使用默认工作空间。                                                             |
| `-p`, `--project`   | 同步日志的项目。如果未指定，将使用默认项目。                                                                     |
| `-i`, `--id`        | 同步日志的实验ID。仅当路径为单个目录时可用。                                                                     |
| `--start-step`      | 仅同步 `step` 大于等于该值的指标/媒体记录（含端点）。                                                            |
| `--end-step`        | 仅同步 `step` 小于等于该值的指标/媒体记录（含端点）。                                                            |
| `--start-time`      | 仅同步该时间及之后的记录。支持 ISO 8601（如 `2026-10-20 10:00:00`）或秒/毫秒级 Unix 时间戳，默认按本机时区解析。 |
| `--end-time`        | 仅同步该时间及之前的记录，格式同 `--start-time`。                                                                |

## 介绍

将本地日志，同步上传到SwanLab云端/私有化部署端。

## 版本对照

> 版本对照仅适用于`swanlab sync`命令

| swanlab库版本 | 特性                                         | 支持的日志文件                 |
| ------------- | -------------------------------------------- | ------------------------------ |
| >=0.6.8       | 支持同步训练异常终端的日志文件；支持`id`参数 | 由`>=0.6.8`版本的swanlab库产生 |
| <0.6.8        | -                                            | 由`<0.6.8`版本的swanlab库产生  |

## 命令行示例

找到你需要上传到云端的日志文件目录（默认是`swanlog`下的以`run-`开头的目录），然后执行命令：

```bash
swanlab sync ./swanlog/run-xxx
```

::: info
默认同步到的项目的是日志文件中记录的`project`，即跑该实验时设置的`project`。  
如果想要同步到其他项目，可以使用`-p`选项指定项目。
:::

看到下面的打印信息，则表示同步成功：

![swanlab sync](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/zh/api/cli-swanlab-sync/console.png)

完成sync操作后，项目上会多出一个新的实验。

## 批量上传

```bash
swanlab sync ./swanlog/run-*
```

## 在原实验上同步

如果你不希望创建新实验，而是在原本的实验上同步，可以使用`--id`参数：

```bash
swanlab sync ./swanlog/run-xxx --id <实验ID>
```

- 同步过程中，目标实验状态会短暂显示为运行中（`RUNNING`），同步完成后自动恢复为实验原本的结束状态。
- 若原实验仍在训练中，同步完成后会继续保持 `RUNNING` 状态，不会中断或影响正在进行的训练。
- 默认情况下，同步会从云端已有的最新进度往后增量续传；如需补传中间缺失的数据段，请配合范围参数使用，详见[范围同步](#范围同步)。

实验ID获取方式见：[恢复实验/断点续训](../guide_cloud/experiment_track/resume-experiment.md)

## 指定范围同步

默认情况下，`swanlab sync --id` 会从云端已记录的最新进度往后**增量续传**。如果训练中途因断网等原因缺失了**中间某一段**数据，但后续数据已正常上报，默认的增量续传不会自动回填中间的空缺。

此时可以使用范围参数，定向补传指定区间内的记录：

| 参数                          | 作用范围                 | 说明                                         |
| ----------------------------- | ------------------------ | -------------------------------------------- |
| `--start-step` / `--end-step` | 标量指标、媒体           | 仅同步 `step` 落在指定区间内的记录（含端点） |
| `--start-time` / `--end-time` | 标量指标、媒体、终端日志 | 按数据记录产生的时间筛选（含端点）           |

- 同时传入 `step` 与时间范围时，两者取交集。
- 仅传入 `step` 范围时，**不会同步终端日志**（终端日志仅支持按时间筛选）；如需补传终端日志，请配合使用时间范围。
- 训练仍在运行时，也可以在另一个终端对同一实验执行带 `--id` 的范围同步，同步完成后不会中断或影响正在进行的训练。
- 时间参数支持标准日期时间格式（如 `2026-10-20 10:00:00`）以及秒级/毫秒级 Unix 时间戳；未指定时区偏移时，默认按执行命令的本机时区解析。

```bash
# 仅补传 step 100~200 的指标与媒体
swanlab sync ./swanlog/run-xxx --id <实验ID> --start-step 100 --end-step 200

# 仅补传 10:00~12:00 之间的指标、媒体和终端日志
swanlab sync ./swanlog/run-xxx --id <实验ID> \
    --start-time "2026-10-20 10:00:00" --end-time "2026-10-20 12:00:00"

# step 与时间范围取交集
swanlab sync ./swanlog/run-xxx --id <实验ID> \
    --start-step 100 --end-step 200 --start-time "2026-10-20 10:00:00"
```

::: warning
⚠️ 注意：尽量避免重复同步线上已有区间，范围同步会直接上传指定区间内的所有记录，不会自动跳过云端已存在的数据。
:::

因此：

- 若只需续传实验末尾未上传的数据，直接执行默认的**全量同步**（不传范围参数）即可，不会产生重复数据；
- 使用**范围同步**补传中间缺失的数据段时，建议将范围尽量收窄到实际缺失的区间内。

## Python代码示例

```python
import swanlab

swanlab.login(api_key="你的API Key")

# 全量同步：同步到日志文件中记录的 project
swanlab.sync("./swanlog/run-xxx")

# 指定项目/工作空间
swanlab.sync(
    "./swanlog/run-xxx",
    settings=swanlab.Settings(
        project=swanlab.Settings.Project(workspace="swanlab", name="sync_test")
    ),
)

# 范围同步：补到已有实验的 step 100~200
swanlab.sync(
    "./swanlog/run-xxx",
    settings=swanlab.Settings(run=swanlab.Settings.Run(id="<实验ID>")),
    start_step=100,
    end_step=200,
)
```
