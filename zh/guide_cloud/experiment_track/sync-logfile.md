# 上传离线实验数据

> 将本地的实验数据，同步到SwanLab云端/私有化部署端

**使用场景：**

1. **训练过程中断网：** 模型训一半服务器断网了1小时，SwanLab的云端记录断了
2. **无网训练环境：** 你不幸地发现公司/学校的训练服务器完全不联网，也不让安装Docker
3. **多端同步：** 你使用了私有化部署，但也希望把实验记录同步到云端
4. **复制实验：** 这个实验对你创建的多个项目都有普适性（比如baseline、benchmark实验），希望能复制到多个项目下
5. **云端记录缺失：** 训练正常结束，但云端只记录了其中一部分（例如中途断网或上报失败），需要按时间段或 `step` 区间补齐中间的数据

当你遇到上述场景时，`swanlab sync`命令可以帮你解决，[API文档](../../api/cli-swanlab-sync.md)。

::: warning 版本兼容性注意

`swanlab sync` **不向后兼容**旧版 SwanLab SDK 生成的日志文件。

自 **SwanLab SDK `v0.8.0`** 起，本地日志格式发生了较大变动。由 `v0.8.0` 及以上版本生成的日志文件，**无法被旧版 `swanlab sync` 同步**；反过来，旧版 swanlog 文件也**无法被新版 SDK 同步**。

**同步前请确认：**

1. 你使用的 **`swanlab sync` CLI 版本**与**生成日志文件的 SDK 版本**一致或更旧。
2. 如果日志文件由低于 `v0.8.0` 的 SDK 生成，请使用**相同版本**的 SwanLab 运行 `swanlab sync`，不要升级。
3. 如果同步过程中遇到解析错误或数据缺失，说明日志格式不兼容——请安装生成日志时的原始 SDK 版本后重试。

:::

## 本地实验数据目录

SwanLab会将**实验记录文件**默认存放到项目目录的`swanlog`目录下，你也可以通过`logdir`参数自定义保存路径，[文档](../../api/py-init.md#设置日志文件保存位置)。

每当你创建1个实验时，SwanLab会自动在`swanlog`目录下创建1个`run-[实验ID]`目录，将数据持续记录到该目录下。

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/zh/guide_cloud/experiment_track/sync-logfile/run_dir.png)

::: warning Tips
这个run开头的目录非常重要，是`swanlab sync`的基础。
:::

## 上传实验数据

找到你需要上传到云端的数据文件目录，然后执行命令：

```bash
swanlab sync ./swanlog/run-xxx
```

:::info
默认同步到的项目的是日志文件中记录的`project`，即跑该实验时设置的`project`。
如果想要同步到其他项目，可以使用`-p`选项指定项目。
:::

看到下面的打印信息，则表示上传成功：

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/zh/guide_cloud/experiment_track/sync-logfile/cli.png)

## 在原实验上同步

如果你不希望创建1个新实验，而是在原本的实验上同步，可以使用`--id`参数：

```bash
swanlab sync ./swanlog/run-xxx --id <实验ID>
```

- 默认情况下，向已有实验同步会从云端已记录的最新进度往后增量续传。如果缺失的是中间某一段数据，请参考使用范围参数定向补传。

实验ID获取方式见：[恢复实验/断点续训](./resume-experiment.md)

## 批量上传

可以使用通配符批量上传：

```bash
swanlab sync ./swanlog/run-*
```

## 上传到指定项目/团队空间

```bash
swanlab sync ./swanlog/run-xxx -p <project_name> -w <workspace_name>
```

:::warning 复制实验场景
如果你希望一个实验可以出现在多个项目下，那么可以使用上述命令，将实验上传到多个项目下。
:::

## 上传到私有化部署

```bash
swanlab sync ./swanlog/run-xxx --host <私有化部署地址>
```

## 指定范围同步

默认的全量同步仅会续传云端最新进度之后的数据，如果仅缺失的训练中间某一段数据（例如中途数据库宕机，但后续恢复上报），可以使用范围参数定向补传该区间的记录：

```bash
# 按时间补传：补齐 10:00~12:00 的指标、媒体和终端日志
swanlab sync ./swanlog/run-xxx --id <实验ID> \
    --start-time "2026-10-20 10:00:00" --end-time "2026-10-20 12:00:00"

# 按 step 补传：补齐 step 100~200 的指标和媒体
swanlab sync ./swanlog/run-xxx --id <实验ID> --start-step 100 --end-step 200
```

| 参数                          | 作用范围                       |
| ----------------------------- | ------------------------------ |
| `--start-step` / `--end-step` | 指标、媒体（含端点）           |
| `--start-time` / `--end-time` | 指标、媒体、终端日志（含端点） |

同时指定 `step` 与时间范围时取交集；仅指定 `step` 范围时不会同步终端日志，如需补传终端日志请使用时间范围。

::: warning 避免重复同步已有区间
范围同步会直接上传指定区间内的所有记录（不自动跳过云端已存在的数据）。如果同步范围与云端已有数据重叠：

- **指标**：图表曲线按 `step` 对齐展示，不会叠加显示，但可能会影响实验的汇总统计值；
- **媒体**（图像、音频等）：同一 `step` 下重复上传的媒体记录会并列展示多条。

因此，建议将范围尽量收窄到实际缺失的区间内。若只需续传实验末尾的数据，直接执行**默认的全量同步（不传范围参数）**即可，不会产生重复记录。

> 训练仍在运行时，也可以在另一个终端对同一实验执行带 `--id` 的范围同步，同步完成后不会中断或影响原训练继续写入。
