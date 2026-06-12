# swanlab api

```bash
swanlab api [OPTIONS] COMMAND ARGS [ARGS]
```

`swanlab api` 是 [开放接口（OpenAPI）](./py-api.md) 的 CLI 实现，允许通过命令行直接操作云端 **工作空间 / 项目 / 实验 / 用户 / 私有化部署** 资源。

> 认证方式与 Python API 一致：显式传入 `--api-key` / `--host`，或使用 `swanlab login` 本地登录态。详见 [认证说明](./py-api.md)。

## 子命令总览

### workspace — 工作空间

| 子命令 | 描述 |
|--------|------|
| `swanlab api workspace info <username>` | 获取工作空间信息 |

### project — 项目

| 子命令 | 描述 |
|--------|------|
| `swanlab api project info <path>` | 获取项目信息 |
| `swanlab api project list` | 列出工作空间下的项目 |
| `swanlab api project create` | 创建项目 |

### run — 实验

| 子命令 | 描述 |
|--------|------|
| `swanlab api run info <path>` | 获取实验信息 |
| `swanlab api run list <project_path>` | 列出项目下的实验 |
| `swanlab api run filter <project_path>` | 按条件过滤实验 |
| `swanlab api run metrics` | 获取实验标量指标 |
| `swanlab api run summary` | 获取实验指标汇总 |
| `swanlab api run column` | 获取实验单个指标列 |
| `swanlab api run columns` | 获取实验指标列列表 |
| `swanlab api run medias` | 获取实验媒体指标 |
| `swanlab api run logs` | 获取实验控制台日志 |
| `swanlab api run export-logs` | 导出实验日志文件 |

### user — 用户

| 子命令 | 描述 |
|--------|------|
| `swanlab api user info` | 获取当前用户信息 |

### self-hosted — 私有化部署管理

> 仅超级管理员可用。

| 子命令 | 描述 |
|--------|------|
| `swanlab api self-hosted info` | 获取实例信息 |
| `swanlab api self-hosted create-user` | 创建用户 |
| `swanlab api self-hosted list-users` | 列出所有用户 |
| `swanlab api self-hosted list-projects` | 列出所有项目 |
| `swanlab api self-hosted summary` | 系统用量汇总 |
| `swanlab api self-hosted list-workspaces` | 列出所有工作空间 |

## 通用选项

所有 `swanlab api` 子命令均支持以下选项：

| 选项 | 描述 |
|------|------|
| `-h`, `--host` | SwanLab 服务主机地址 |
| `-k`, `--api-key` | API 密钥，优先于本地登录态 |
| `--save` | 将输出保存为 JSON 文件，传值指定文件名，不传则自动生成 |


## 子命令详解

### workspace info

获取指定工作空间的详细信息。

```bash
swanlab api workspace info <username> [OPTIONS]
```

| 参数/选项 | 类型 | 描述 |
|-----------|------|------|
| `username` | 位置参数 | 工作空间用户名（唯一 ID） |
| `--save` | 选项 | 保存输出为 JSON 文件，传值指定文件名，不传则自动生成 |

```bash
# 查看工作空间信息
swanlab api workspace info my-team

# 保存到文件
swanlab api workspace info my-team --save workspace.json
```


### project info

获取单个项目的详细信息。

```bash
swanlab api project info <path> [OPTIONS]
```

| 参数/选项 | 类型 | 描述 |
|-----------|------|------|
| `path` | 位置参数 | 项目路径，格式为 `username/project-name` |
| `--save` | 选项 | 保存输出为 JSON 文件 |

```bash
swanlab api project info my-team/image-classification
```


### project list

列出指定工作空间下的所有项目。

```bash
swanlab api project list [OPTIONS]
```

| 参数/选项 | 类型 | 默认值 | 描述 |
|-----------|------|--------|------|
| `--workspace` | `str` | 当前登录用户 | 工作空间用户名 |
| `--page_num` / `-n` | `int` | `1` | 页码 |
| `--page_size` / `-s` | `str` | `"20"` | 每页数量，可选值：`"10"`, `"20"`, `"50"`, `"100"` |
| `--all` | 布尔标志 | `False` | 自动翻页，获取全部项目 |
| `--save` | 选项 | — | 保存输出为 JSON 文件 |

```bash
# 列出当前工作空间的项目
swanlab api project list

# 列出指定工作空间的项目，每页 50 条
swanlab api project list --workspace my-team --page_size 50

# 获取全部项目
swanlab api project list --workspace my-team --all
```


### project create

在指定工作空间下创建新项目。

```bash
swanlab api project create [OPTIONS]
```

| 参数/选项 | 类型 | 默认值 | 描述 |
|-----------|------|--------|------|
| `-n` / `--name` | `str` | 必填 | 项目名（1–100 字符，仅允许 `0-9a-zA-Z-_.+`） |
| `-v` / `--visibility` | `str` | `"PRIVATE"` | 可见性：`PUBLIC` 或 `PRIVATE` |
| `-d` / `--description` | `str` | `None` | 项目描述 |
| `-w` / `--workspace` | `str` | 当前登录用户 | 工作空间用户名 |
| `--save` | 选项 | — | 保存输出为 JSON 文件 |

```bash
# 创建一个私有项目
swanlab api project create -n my-project -v PRIVATE

# 在指定工作空间创建一个公开项目
swanlab api project create -n my-project -v PUBLIC -w my-team -d "图像分类实验"
```


### run info

获取单个实验的详细信息。

```bash
swanlab api run info <path> [OPTIONS]
```

| 参数/选项 | 类型 | 描述 |
|-----------|------|------|
| `path` | 位置参数 | 实验路径，格式为 `username/project-name/experiment-id` |
| `--save` | 选项 | 保存输出为 JSON 文件 |

```bash
swanlab api run info my-team/image-classification/abc123
```


### run list

列出指定项目下的所有实验。

```bash
swanlab api run list <project_path> [OPTIONS]
```

| 参数/选项 | 类型 | 默认值 | 描述 |
|-----------|------|--------|------|
| `project_path` | 位置参数 | 必填 | 项目路径，格式为 `username/project-name` |
| `--page_num` / `-n` | `int` | `1` | 页码 |
| `--page_size` / `-s` | `str` | `"20"` | 每页数量 |
| `--all` | 布尔标志 | `False` | 自动翻页，获取全部实验 |
| `--save` | 选项 | — | 保存输出为 JSON 文件 |

```bash
# 列出项目下的实验
swanlab api run list my-team/image-classification

# 获取全部实验
swanlab api run list my-team/image-classification --all
```


### run filter

按过滤条件查询实验。使用 JSON 数组指定一个或多个过滤规则。

```bash
swanlab api run filter <project_path> --filter_query <json> [OPTIONS]
```

| 参数/选项 | 类型 | 描述 |
|-----------|------|------|
| `project_path` | 位置参数 | 必填，项目路径，格式为 `username/project-name` |
| `--filter_query` / `-f` | `str` | 必填，过滤条件（JSON 数组或 JSON 文件路径） |
| `--save` | 选项 | 保存输出为 JSON 文件 |

每个过滤规则为一个 JSON 对象：

```json
{
  "key": "<字段名>",
  "type": "STABLE | CONFIG | SCALAR",
  "op": "EQ | NEQ | GT | LT | GTE | LTE | CONTAIN | NOT_CONTAIN",
  "value": ["<值>"]
}
```

`type` 取值说明：

| type | key 取值 | value 说明 |
|------|----------|-----------|
| `STABLE` | `state`, `name`, `description`, `show`, `pin`, `labels`, `createdAt`, `updatedAt`, `finishedAt` 等 | 对应字段的值 |
| `CONFIG` | 配置参数名（如 `param_2`，**不带** `config.` 前缀） | 配置参数的值 |
| `SCALAR` | 标量指标名（如 `loss`） | 指标的值 |

```bash
# 查询状态为 FINISHED 的实验
swanlab api run filter my-team/image-classification \
  -f '[{"key": "state", "type": "STABLE", "op": "EQ", "value": ["FINISHED"]}]'

# 从文件读取过滤条件
swanlab api run filter my-team/image-classification -f ./filter.json
```


### run metrics

获取实验的标量指标数据，返回 JSON 格式。

```bash
swanlab api run metrics <path> --keys <keys> [OPTIONS]
```

| 参数/选项 | 类型 | 默认值 | 描述 |
|-----------|------|--------|------|
| `path` | 位置参数 | 必填 | 实验路径 |
| `--keys` | `str` | 必填 | 逗号分隔的指标名，如 `"loss,acc"` |
| `--sample` / `-s` | `int` | `1500` | 采样数量，超过自动截断 |
| `--ignore-timestamp` | 布尔标志 | `False` | 去掉指标数据中的时间戳字段 |
| `--all` | 布尔标志 | `False` | 获取全量数据（CSV 导出） |
| `--range-type` | `str` | `None` | 范围查询类型：`step` 或 `timestamp` |
| `--range-start` | `int` | `None` | 范围起始值（含），步数或毫秒时间戳 |
| `--range-end` | `int` | `None` | 范围结束值（含），步数或毫秒时间戳 |
| `--range-head` | `int` | `None` | 返回前 N 条数据 |
| `--range-tail` | `int` | `None` | 返回后 N 条数据 |
| `--range-last` | `int` | `None` | 最近 N 毫秒的数据（与 `--range-start`/`--range-end` 互斥） |
| `--save` | 选项 | — | 保存输出为 JSON 文件 |

**注意事项：**

- `--range-head` 和 `--range-tail` 互斥。
- `--range-last` 与 `--range-start`/`--range-end` 互斥。
- `--range-head`/`--range-tail` 可与 `--range-start`/`--range-end` 或 `--range-last` 组合（先范围过滤，再截取）。
- `--range-start` 和 `--range-end` 配合 `--range-type` 使用（`step` 或 `timestamp`），时间戳单位为毫秒。

```bash
# 获取 loss 指标（默认采样 1500 条）
swanlab api run metrics my-team/image-classification/abc123 --keys loss

# 获取多个指标，采样 500 条
swanlab api run metrics my-team/image-classification/abc123 --keys loss,acc -s 500

# 按步数范围查询
swanlab api run metrics my-team/image-classification/abc123 \
  --keys loss --range-type step --range-start 100 --range-end 500

# 最近 5 分钟的数据
swanlab api run metrics my-team/image-classification/abc123 \
  --keys loss --range-last 300000

# 获取最后 200 条数据
swanlab api run metrics my-team/image-classification/abc123 \
  --keys loss --range-tail 200

# 步数范围 + 取前 50 个点
swanlab api run metrics my-team/image-classification/abc123 \
  --keys loss --range-type step --range-start 0 --range-end 500 --range-head 50
```


### run summary

获取实验的标量指标汇总（如最终值、最小值、最大值等）。

```bash
swanlab api run summary <path> [OPTIONS]
```

| 参数/选项 | 类型 | 描述 |
|-----------|------|------|
| `path` | 位置参数 | 实验路径 |
| `--keys` | `str` | 逗号分隔的指标名，不传则查询全部 |
| `--save` | 选项 | 保存输出为 JSON 文件 |

```bash
# 获取全部指标汇总
swanlab api run summary my-team/image-classification/abc123

# 获取指定指标的汇总
swanlab api run summary my-team/image-classification/abc123 --keys loss,acc
```


### run column

获取实验的单个指标列。

```bash
swanlab api run column <path> --key <key> [OPTIONS]
```

| 参数/选项 | 类型 | 默认值 | 描述 |
|-----------|------|--------|------|
| `path` | 位置参数 | 必填 | 实验路径 |
| `--key` | `str` | 必填 | 指标列名 |
| `--class` | `str` | `"CUSTOM"` | 列分类：`CUSTOM` 或 `SYSTEM` |
| `--type` | `str` | `None` | 列数据类型：`FLOAT`, `STRING`, `IMAGE` 等 |
| `--save` | 选项 | — | 保存输出为 JSON 文件 |

```bash
swanlab api run column my-team/image-classification/abc123 --key loss
```


### run columns

列出实验的所有指标列。

```bash
swanlab api run columns <path> [OPTIONS]
```

| 参数/选项 | 类型 | 默认值 | 描述 |
|-----------|------|--------|------|
| `path` | 位置参数 | 必填 | 实验路径 |
| `--page_num` / `-n` | `int` | `1` | 页码 |
| `--page_size` / `-s` | `str` | `"20"` | 每页数量 |
| `--search` | `str` | `None` | 模糊搜索关键词（匹配列 name） |
| `--class` | `str` | `"CUSTOM"` | 列分类筛选 |
| `--type` | `str` | `None` | 列数据类型筛选 |
| `--all` | 布尔标志 | `False` | 自动翻页，获取全部列 |
| `--save` | 选项 | — | 保存输出为 JSON 文件 |

```bash
# 列出所有指标列
swanlab api run columns my-team/image-classification/abc123

# 模糊搜索
swanlab api run columns my-team/image-classification/abc123 --search loss

# 仅列出 FLOAT 类型的系统列
swanlab api run columns my-team/image-classification/abc123 \
  --class SYSTEM --type FLOAT --all
```


### run medias

获取实验的媒体指标（图片、音频等），返回预签名 URL。

```bash
swanlab api run medias <path> --keys <keys> [OPTIONS]
```

| 参数/选项 | 类型 | 默认值 | 描述 |
|-----------|------|--------|------|
| `path` | 位置参数 | 必填 | 实验路径 |
| `--keys` | `str` | 必填 | 逗号分隔的媒体指标名，如 `"image,audio"` |
| `--step` / `-s` | `int` | `0` | 步数 |
| `--all` | 布尔标志 | `False` | 获取所有步数 |
| `--save` | 选项 | — | 保存输出为 JSON 文件 |

```bash
# 获取 step=0 的图片指标
swanlab api run medias my-team/image-classification/abc123 --keys generated_image

# 获取所有步数的音频指标
swanlab api run medias my-team/image-classification/abc123 --keys audio --all
```


### run logs

获取实验的控制台日志。

```bash
swanlab api run logs <path> [OPTIONS]
```

| 参数/选项 | 类型 | 默认值 | 描述 |
|-----------|------|--------|------|
| `path` | 位置参数 | 必填 | 实验路径 |
| `--offset` / `-o` | `int` | `0` | 日志分片索引（shard index） |
| `--level` / `-l` | `str` | `"INFO"` | 日志级别：`DEBUG`, `INFO`, `WARN`, `ERROR` |
| `--ignore-timestamp` | 布尔标志 | `False` | 去掉时间戳字段 |
| `--save` | 选项 | — | 保存输出为 JSON 文件 |

```bash
# 获取 INFO 级别日志
swanlab api run logs my-team/image-classification/abc123

# 获取 WARN 及以上级别的日志
swanlab api run logs my-team/image-classification/abc123 --level WARN

# 获取指定分片的日志
swanlab api run logs my-team/image-classification/abc123 --offset 1
```


### run export-logs

将实验日志导出为可下载的 `.log` 文件。

```bash
swanlab api run export-logs <path> [OPTIONS]
```

| 参数/选项 | 类型 | 默认值 | 描述 |
|-----------|------|--------|------|
| `path` | 位置参数 | 必填 | 实验路径 |
| `--start` | `int` | `0` | 起始行索引（0-based） |
| `--rows` / `-r` | `int` | `500000` | 导出行数，最大 500000 |
| `--save` | 选项 | — | 保存输出为 JSON 文件（含下载 URL） |

```bash
# 导出前 10000 行日志
swanlab api run export-logs my-team/image-classification/abc123 --rows 10000
```


### user info

获取当前登录用户的信息。

```bash
swanlab api user info [OPTIONS]
```

| 选项 | 描述 |
|------|------|
| `--save` | 保存输出为 JSON 文件 |

```bash
swanlab api user info
```


### self-hosted info

获取私有化部署实例的信息。

```bash
swanlab api self-hosted info [OPTIONS]
```

| 选项 | 描述 |
|------|------|
| `--save` | 保存输出为 JSON 文件 |

```bash
swanlab api self-hosted info
```


### self-hosted create-user

在私有化部署中创建一个新用户（仅超级管理员）。

```bash
swanlab api self-hosted create-user [OPTIONS]
```

| 选项 | 类型 | 描述 |
|------|------|------|
| `-u` / `--username` | `str` | 必填，新用户名 |
| `-p` / `--password` | `str` | 必填，新用户密码 |
| `--save` | 选项 | 保存输出为 JSON 文件 |

```bash
swanlab api self-hosted create-user -u testuser -p test123456
```


### self-hosted list-users

列出私有化部署中的所有用户（仅超级管理员）。

```bash
swanlab api self-hosted list-users [OPTIONS]
```

| 选项 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--page_num` / `-n` | `int` | `1` | 页码 |
| `--page_size` / `-s` | `int` | `20` | 每页数量 |
| `--all` | 布尔标志 | `False` | 自动翻页，获取全部用户 |
| `--save` | 选项 | — | 保存输出为 JSON 文件 |

```bash
# 列出所有用户
swanlab api self-hosted list-users --all
```


### self-hosted list-projects

列出私有化部署中的所有项目（仅超级管理员）。

```bash
swanlab api self-hosted list-projects [OPTIONS]
```

| 选项 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--page_num` / `-n` | `int` | `1` | 页码 |
| `--page_size` / `-s` | `int` | `20` | 每页数量 |
| `--all` | 布尔标志 | `False` | 自动翻页，获取全部项目 |
| `--search` | `str` | `None` | 搜索关键词 |
| `--creator` | `str` | `None` | 按创建者用户名筛选 |
| `--workspace` | `str` | `None` | 按工作空间用户名筛选 |
| `--save` | 选项 | — | 保存输出为 JSON 文件 |

```bash
# 列出所有项目
swanlab api self-hosted list-projects --all

# 搜索特定项目
swanlab api self-hosted list-projects --search image --all
```


### self-hosted summary

获取私有化部署的系统用量汇总（仅超级管理员）。

```bash
swanlab api self-hosted summary [OPTIONS]
```

| 选项 | 描述 |
|------|------|
| `--save` | 保存输出为 JSON 文件 |

```bash
swanlab api self-hosted summary
```


### self-hosted list-workspaces

列出私有化部署中的所有工作空间（仅超级管理员）。

```bash
swanlab api self-hosted list-workspaces [OPTIONS]
```

| 选项 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--page_num` / `-n` | `int` | `1` | 页码 |
| `--page_size` / `-s` | `int` | `20` | 每页数量 |
| `--all` | 布尔标志 | `False` | 自动翻页，获取全部工作空间 |
| `--search` | `str` | `None` | 搜索关键词 |
| `--save` | 选项 | — | 保存输出为 JSON 文件 |

```bash
# 列出所有工作空间
swanlab api self-hosted list-workspaces --all
```
