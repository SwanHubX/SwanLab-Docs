# swanlab.Api

::: warning 版本提示
此文档适用于 swanlab >= `0.8.0`。
:::

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/zh/api/py-openapi/logo.jpg)

- 新版开放接口使用 **OOP 风格**，所有操作通过 `Api` 入口获取实体对象，实体支持懒加载（访问属性时才发起请求），并统一通过 `.json()` 序列化为 `dict`。
- 所有操作均可通过 [`swanlab api`](/api/cli-swanlab-api.md) **以 CLI 命令行方式调用**，适用于脚本、CI/CD 或无需编写 Python 代码的场景。


> 认证优先级：显式传入的 `api_key` / `host` > `swanlab.login()` 登录态 > 环境变量 `SWANLAB_API_KEY` / `SWANLAB_API_HOST`

## 🚀 快速开始

```python
import swanlab

api = swanlab.Api()                              # 自动读取 .netrc 凭证
api = swanlab.Api(api_key="your-api-key", host="your-host")  # 显式传入
```

私有化部署示例：

```python
api = swanlab.Api(api_key="your-api-key", host="https://your-server.com")
```

## 🧱 路径规范

- **工作空间**：`username`
- **项目**：`username/project-name`
- **实验**：`username/project-name/experiment-id`

## 🔨 核心概念

新版开放接口围绕三个核心概念组织：**工作空间（Workspace）** → **项目（Project）** → **实验（Experiment/Run）**，形成清晰的层级关系：

```
Workspace（工作空间）
 └── Project（项目）
      └── Experiment/Run（实验）
           ├── metrics（标量/媒体指标）
           ├── columns（指标列）
           └── logs（日志）
```

| 概念 | 说明 |
|------|------|
| **Workspace** | 项目的集合，对应一个团队或用户。分为个人空间（`PERSON`）和组织空间（`TEAM`）。 |
| **Project** | 实验的集合，对应一个研发任务，包含名称、描述、标签、可见性等元信息。 |
| **Experiment** | 单次训练/推理任务，包含指标、配置、日志、环境信息等。 |
| **Column** | 实验下的指标列，如 `loss`、`acc`，支持 FLOAT / STRING / IMAGE 等多种数据类型。 |
| **Metric** | 实验下某一指标列对应的指标值。 |



### 数据类型：metrics / medias / logs

实验下的数据分为三类，通过不同的方法获取：

| 类型 | 方法 | 说明 | 典型场景 |
|------|------|------|----------|
| **标量指标** | `run.metrics(keys=[...])` | 数值型指标（如 loss、acc），支持采样、范围查询，返回结构化数据 | 训练曲线分析、趋势对比 |
| **媒体指标** | `run.medias(keys=[...])` | 图片、音频、视频等非结构化媒体数据，返回预签名 URL | 可视化检查、结果预览 |
| **日志** | `run.logs()` / `run.export_logs()` | 实验运行时的文本日志，支持按级别筛选 | 调试、排错、审计 |

```python
import swanlab

api = swanlab.Api()
run = api.run(path="my-team/my-project/abc123")

# 标量指标：loss、acc 等数值
scalars = run.metrics(keys=["loss", "acc"], sample=500)
print(scalars["metrics"])  # [{step: 1, value: 0.9, timestamp: ...}, ...]

# 媒体指标：图片、音频等
media = run.medias(keys=["generated_image"], step=10)
print(media["metrics"])  # [{index: 10, items: [{url: "https://...", ...}]}, ...]

# 日志：文本输出
logs = run.logs(offset=0, level="INFO")
print(logs["logs"])  # [{message: "...", level: "INFO", timestamp: ...}, ...]
```

**注意事项：**

- `metrics()` 的 `sample` 参数默认 1500，超过会自动截断；设置 `all=True` 获取全量数据。
- `range_query` 仅对 **SCALAR** 标量数值类型有效，支持按步数或时间戳范围过滤。
- `medias()` 返回的媒体数据通过预签名 URL 提供，需在有效期内下载。
- `export_logs()` 可导出大量日志为 `.log` 文件，适合持久化保存。

## Api

`Api` 是所有操作的入口，构造时即完成认证，返回独立的 `Client` 实例（与 SDK 运行时互不干扰）。

| 方法 | 描述 |
|------|------|
| `api.workspace(username)` | 获取单个工作空间 |
| `api.workspaces(username)` | 获取工作空间列表（迭代器） |
| `api.project(path)` | 获取单个项目 |
| `api.projects(path, ...)` | 获取项目列表（迭代器） |
| `api.create_project(username, name, ...)` | 创建项目 |
| `api.run(path)` | 获取单个实验 |
| `api.runs(path, filters=...)` | 获取实验列表（POST 过滤模式） |
| `api.runs_get(path, ...)` | 获取实验列表（GET 分页模式） |
| `api.column(path, key)` | 获取单个指标列 |
| `api.columns(path, ...)` | 获取实验的指标列列表 |
| `api.user()` | 获取当前用户信息 |
| `api.self_hosted()` | 私有化部署管理入口 |



## 工作空间(Workspace)

**Workspace 属性：**

| 属性 | 类型 | 描述 |
|------|------|------|
| `name` | `str` | 空间名称 |
| `username` | `str` | 空间用户名（唯一ID） |
| `workspace_type` | `str` | `PERSON` 或 `TEAM` |
| `role` | `str` | 当前用户角色，`OWNER` 或 `MEMBER` |
| `profile` | `dict` | 介绍信息 |
| `comment` | `str` | 空间简介 |

**Workspace 方法：**

| 方法 | 描述 |
|------|------|
| `projects(sort, search, detail, page, size, all)` | 获取空间下的项目列表（迭代器） |
| `create_project(name, visibility, description)` | 创建项目，返回 `Project` 或 `None` |
| `json()` | 序列化为 `dict` |

**workspace 参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `username` | `str` | 当前登录用户 | 工作空间用户名 |

:::code-group

```python [获取单个工作空间]
import swanlab

api = swanlab.Api()

ws = api.workspace(username="my-team")

data = ws.json()
print(data["name"], data["username"], data["workspace_type"])
```

```python [遍历工作空间列表]
import swanlab

api = swanlab.Api()

for ws in api.workspaces("my-team"):
    print(ws.name)
```

:::



## 项目(Project)

**Project 属性：**

| 属性 | 类型 | 描述 |
|------|------|------|
| `name` | `str` | 项目名 |
| `path` | `str` | 项目路径 `username/project-name` |
| `description` | `str` | 项目描述 |
| `labels` | `list` | 项目标签列表 |
| `created_at` | `str` | 创建时间（ISO 8601 UTC） |
| `updated_at` | `str` | 更新时间 |
| `url` | `str` | 项目 Web 页面 URL |
| `visibility` | `str` | `PUBLIC` 或 `PRIVATE` |
| `count` | `dict` | 统计信息（实验数、协作者数等） |

**Project 方法：**

| 方法 | 描述 |
|------|------|
| `runs(filters)` | 获取实验列表（POST 过滤模式） |
| `runs_get(page, size, all)` | 获取实验列表（GET 分页模式） |
| `delete_runs(run_ids, commit)` | 批量删除实验 |
| `delete(commit)` | 删除项目 |
| `json()` | 序列化为 `dict` |

**projects 参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `path` | `str` | — | 工作空间用户名，如 `"my-team"` |
| `sort` | `str` | — | 排序字段：`created_at`、`updated_at`、`name` |
| `search` | `str` | — | 搜索关键词（模糊匹配项目名） |
| `detail` | `bool` | `True` | 是否返回详细信息 |
| `page` | `int` | `1` | 起始页码 |
| `size` | `int` | `20` | 每页条数 |
| `all` | `bool` | `False` | 是否自动翻页获取全部 |

:::code-group

```python [获取项目]
import swanlab

api = swanlab.Api()

project = api.project(path="my-team/my-project")

print(project.name)
print(project.description)
print(project.labels)
print(project.visibility)  # PUBLIC 或 PRIVATE
print(project.created_at)
print(project.url)
print(project.count)       # 实验数、协作者数等
```

```python [获取项目列表]
import swanlab

api = swanlab.Api()

for p in api.projects(path="my-team", sort="updated_at", search="image"):
    print(p.name, p.path)
```

```python [获取实验列表（过滤模式）]
import swanlab

api = swanlab.Api()

project = api.project(path="my-team/my-project")

for run in project.runs():
    print(run.name, run.state, run.created_at)
```

```python [创建项目]
import swanlab

api = swanlab.Api()

project = api.create_project(
    username="my-team",
    name="new-project",
    visibility="PRIVATE",
    description="项目描述",
)
```

```python [删除项目]
import swanlab

api = swanlab.Api()

project = api.project(path="my-team/my-project")
project.delete(commit=True)  # commit=False 仅打印待删除信息
```

```python [批量删除实验]
import swanlab

api = swanlab.Api()

project = api.project(path="my-team/my-project")
project.delete_runs(["run_id_1", "run_id_2"], commit=True)
```

:::


## 实验(Run/Experiment)

**Experiment 属性：**

| 属性 | 类型 | 描述 |
|------|------|------|
| `name` | `str` | 实验名 |
| `description` | `str` | 实验描述 |
| `state` | `str` | 实验状态：`RUNNING`、`FINISHED`、`CRASHED`、`ABORTED`、`OFFLINE` |
| `labels` | `list` | 实验标签列表 |
| `group` | `str` | 实验分组名 |
| `job_type` | `str` | 任务类型 |
| `created_at` | `str` | 创建时间 |
| `finished_at` | `str` | 结束时间，未结束为 `None` |
| `url` | `str` | 实验 Web 页面 URL |
| `show` | `bool` | 图表对比视图是否显示 |
| `profile` | `dict` | 实验配置、环境信息、依赖等 |
| `user` | `dict` | 创建者信息 |

**Experiment 方法：**

| 方法 | 描述 |
|------|------|
| `metrics(keys, sample, all, range_query)` | 获取标量指标数据，返回 `dict` |
| `summary(keys)` | 获取标量指标概要统计，返回 `dict` |
| `logs(offset, level)` | 获取日志数据，返回 `dict` |
| `export_logs(start, rows)` | 导出日志为 `.log` 文件，返回 `ApiResponseType` |
| `medias(keys, step, all)` | 获取媒体指标数据，返回 `dict` |
| `columns(page, size, search, column_type, column_class, all)` | 获取指标列列表（迭代器） |
| `column(key, column_class, column_type)` | 获取单个指标列 |
| `delete(commit)` | 删除实验 |
| `json()` | 序列化为 `dict` |

### 获取实验列表 — 过滤模式

通过条件过滤获取项目下的实验列表。

```python
import swanlab

api = swanlab.Api()

for run in api.runs(path="my-team/my-project"):
    print(run.name, run.state)
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `path` | `str` | 项目路径 `username/project` |
| `filters` | `list[dict]` | 过滤规则列表，每项 `{key, type, op, value}` |

**过滤规则（filter）字段：**

| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `key` | `str` | ✓ | 字段名 |
| `type` | `str` | ✓ | 字段类型：`STABLE`、`CONFIG`、`SCALAR` |
| `op` | `str` | ✓ | 操作符：`EQ`、`NEQ`、`GTE`、`LTE`、`IN`、`NOT IN`、`CONTAIN` |
| `value` | `list` | ✓ | 比较值 |

**type 字段说明：**

| type 值 | key 取值范围 | value 说明 |
|---------|-------------|-----------|
| `STABLE` | 实验固有字段：`state`、`name`、`description`、`show`、`pin`、`baseline`、`colors`、`cluster`、`job`、`createdAt`、`updatedAt`、`finishedAt`、`pinnedAt`、`labels` | 对应字段的值 |
| `CONFIG` | 实验配置的参数名，如 `param_2`（不带 `config.` 前缀） | 配置参数的值 |
| `SCALAR` | 标量指标名 | 指标的值 |

```python
# 过滤示例
for run in api.runs(path="my-team/my-project", filters=[
    {"key": "state", "type": "STABLE", "op": "EQ", "value": ["FINISHED"]},
    {"key": "name", "type": "STABLE", "op": "CONTAIN", "value": ["v2"]},
]):
    print(run.name)
```

### 获取实验列表 — 分页模式

通过标准分页获取项目下的实验列表，返回精简信息。

```python
import swanlab

api = swanlab.Api()

for run in api.runs_get(path="my-team/my-project", page=1, size=20, all=True):
    print(run.name, run.state)
```

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `path` | `str` | — | 项目路径 `username/project` |
| `page` | `int` | `1` | 起始页码 |
| `size` | `int` | `20` | 每页条数 |
| `all` | `bool` | `False` | 是否自动翻页获取全部 |

### 获取单个实验

```python
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

data = run.json()
print(data["name"], data["state"], data["created_at"])
```

### metrics — 标量指标

获取数值型指标数据（如 loss、acc），支持采样控制、范围查询，返回结构化数据。

```python
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

# 获取指标数据，返回 dict
result = run.metrics(keys=["loss", "acc"])
print(result["metrics"])  # 指标列表

# 指定采样数（默认 1500，最大 1500）
result = run.metrics(keys=["loss"], sample=500)

# 全量数据（不受采样限制）
result = run.metrics(keys=["loss"], all=True)

# 范围查询
result = run.metrics(
    keys=["loss"],
    range_query={"type": "step", "start": 100, "end": 500},
)

# 按时间戳范围查询
result = run.metrics(
    keys=["loss"],
    range_query={"type": "timestamp", "start": 1715769600000, "end": 1715773200000},
)
```

**参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `keys` | `list[str]` | — | 指标 key 列表，如 `["loss", "acc"]` |
| `sample` | `int` | `1500` | 采样数量（SCALAR 最大 1500） |
| `all` | `bool` | `False` | 获取全量数据（不受采样限制） |
| `range_query` | `dict` 或 `RangeQuery` | `None` | 范围查询：`{"type": "step", "start": 100, "end": 500}` |
| `x_axis` | `str` | `"step"` | X 轴维度：`step`（步数） |
| `ignore_timestamp` | `bool` | `False` | 是否去除时间戳字段 |

> `range_query` 仅对 SCALAR 类型有效。支持两种传入方式：`dict` 或 `RangeQuery` 对象。

**RangeQuery 用法：**

```python
from swanlab.api.typings.common import RangeQuery

# 按步数范围
rq = RangeQuery(type="step", start=100, end=500)

# 取最后 50 条
rq = RangeQuery(tail=50)

# 按时间戳范围（毫秒）
rq = RangeQuery(type="timestamp", start=1715769600000, end=1715773200000)

# 或直接使用 dict
result = run.metrics(keys=["loss"], range_query={"type": "step", "start": 100})
```

> `head` 和 `tail` 互斥。时间戳不足 13 位会自动补齐到毫秒级。

### summary — 概要统计

获取标量指标的统计摘要（min / max / avg / median / latest），**每个指标以 latest 值为准**。

```python
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

summary = run.summary(keys=["loss", "acc"])
# 返回每个 key 的统计值
print(summary)
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `keys` | `list[str]` | 需要查询的标量 key 列表，为 `None` 表示查询全量 keys |

### medias — 媒体指标

获取图片、音频、视频等非结构化媒体数据，返回预签名 URL。

```python
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

# 获取指定 step 的媒体数据
result = run.medias(keys=["generated_image"], step=10)
print(result)

# 获取全部媒体数据
result = run.medias(keys=["generated_image"], all=True)
print(result)
```

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `keys` | `list[str]` | — | 媒体指标 key 列表 |
| `step` | `int` | `0` | 指定 step，不传则返回最新 |
| `all` | `bool` | `False` | 获取全部历史媒体数据 |

> 返回的媒体数据通过预签名 URL 提供，需在有效期内下载。

### logs — 日志

获取实验运行时的文本日志，支持按级别筛选；也可导出为 `.log` 文件。

```python
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

# 获取日志
logs = run.logs(offset=0, level="INFO")
print(logs)

# 导出日志为 .log 文件（返回预签名下载链接）
result = run.export_logs(start=0, rows=500)
if result.ok:
    print(result.data["url"])
```

**logs 参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `offset` | `int` | `0` | 分页偏移量 |
| `level` | `str` | `"INFO"` | 日志级别：`DEBUG`、`INFO`、`WARN`、`ERROR` |
| `ignore_timestamp` | `bool` | `False` | 是否去除时间戳字段 |

**export_logs 参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `start` | `int` | `0` | 导出起始行号（0-based） |
| `rows` | `int` | `500000` | 导出行数，最大 500000 |

### columns — 指标列

获取实验下的指标列列表，或通过 key 获取单个列。

```python
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

# 获取所有指标列
for col in run.columns(page=1, size=20, all=True):
    print(col.name)

# 获取单个列
col = run.column(key="loss", column_type="FLOAT")
data = col.json()
print(data["name"], data["column_type"], data["created_at"])
```

**columns 参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `page` | `int` | `1` | 起始页码 |
| `size` | `int` | `20` | 每页条数 |
| `search` | `str` | `None` | 搜索关键词（匹配列名） |
| `column_class` | `str` | `None` | 列分类：`CUSTOM` 或 `SYSTEM` |
| `column_type` | `str` | `None` | 列数据类型：`FLOAT`、`STRING`、`IMAGE` 等 |
| `all` | `bool` | `False` | 是否自动翻页获取全部 |

**column 参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `path` | `str` | — | 实验路径 `username/project/run_id` |
| `key` | `str` | — | 列的 key，如 `"loss"` |
| `column_class` | `str` | `"CUSTOM"` | 列分类：`CUSTOM` 或 `SYSTEM` |
| `column_type` | `str` | `None` | 列数据类型：`FLOAT`、`STRING`、`IMAGE` 等 |

### delete — 删除实验

```python
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")
run.delete(commit=True)  # commit=False 仅打印待删除信息，不实际删除
```



## 指标列(Column)

**column 参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `path` | `str` | — | 实验路径 `username/project/run_id` |
| `key` | `str` | — | 列的 key，如 `"loss"` |
| `column_class` | `str` | `"CUSTOM"` | 列分类：`CUSTOM` 或 `SYSTEM` |
| `column_type` | `str` | `None` | 列数据类型：`FLOAT`、`STRING`、`IMAGE`、`VIDEO` 等 |

**columns 参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `path` | `str` | — | 实验路径 `username/project/run_id` |
| `page` | `int` | `1` | 起始页码 |
| `size` | `int` | `20` | 每页条数 |
| `search` | `str` | `None` | 搜索关键词（匹配列名） |
| `column_class` | `str` | `None` | 列分类：`CUSTOM` 或 `SYSTEM` |
| `column_type` | `str` | `None` | 列数据类型 |
| `all` | `bool` | `False` | 是否自动翻页获取全部 |

**Column 属性：**

| 属性 | 类型 | 描述 |
|------|------|------|
| `name` | `str` | 列显示名称 |
| `key` | `str` | 列 key |
| `column_class` | `str` | 列分类：`CUSTOM` 或 `SYSTEM` |
| `column_type` | `str` | 数据类型：`FLOAT`、`STRING`、`IMAGE`、`VIDEO`、`OBJECT3D` 等 |
| `created_at` | `int` | 创建时间戳 |
| `error` | `dict` | 错误信息（如有） |

**Column 方法：**

| 方法 | 描述 |
|------|------|
| `metric(sample, metric_type)` | 获取该列的指标数据，返回 `dict` |
| `export_csv()` | 导出 SCALAR 列为 CSV，返回 `ApiResponseType` |
| `json()` | 序列化为 `dict` |

:::code-group

```python [获取单个列]
import swanlab

api = swanlab.Api()

col = api.column(
    path="my-team/my-project/abc123",
    key="loss",
    column_type="FLOAT",
)

data = col.json()
print(data["name"], data["column_type"], data["created_at"])
```

```python [遍历列列表]
import swanlab

api = swanlab.Api()

for col in api.columns(
    path="my-team/my-project/abc123",
    page=1,
    size=20,
    all=True,
    column_type="FLOAT",
):
    print(col.name)
```

```python [导出 CSV]
import swanlab

api = swanlab.Api()

col = api.column(path="my-team/my-project/abc123", key="loss")
result = col.export_csv()
if result.ok:
    print(result.data["url"])  # CSV 下载链接
```

:::


## 用户信息(User)

```python
import swanlab

api = swanlab.Api()
user = api.user()

data = user.json()
print(data["username"], data["email"])
```

**User 属性：**

| 属性 | 类型 | 描述 |
|------|------|------|
| `username` | `str` | 用户名 |
| `name` | `str` | 显示名称 |
| `bio` | `str` | 个人简介 |
| `institution` | `str` | 机构 |
| `school` | `str` | 学校 |
| `email` | `str` | 邮箱 |
| `location` | `str` | 所在地 |
| `url` | `str` | 个人主页 |



## 私有化管理(self_hosted)

> 此操作仅适用于私有化部署，且需要超级管理员权限。

**SelfHosted 属性：**

| 属性 | 类型 | 描述 |
|------|------|------|
| `enabled` | `bool` | 是否启用私有化模式 |
| `expired` | `bool` | 许可证是否过期 |
| `root` | `bool` | 当前用户是否为超级管理员 |
| `plan` | `str` | 许可证类型：`free` 或 `commercial` |
| `seats` | `int` | 许可证席位数 |

**SelfHosted 方法：**

| 方法 | 描述 |
|------|------|
| `create_user(username, password)` | 创建用户（root 限定），返回 `ApiResponseType` |
| `get_users(page, size, all)` | 分页获取用户列表（root 限定），返回迭代器 |
| `get_projects(page, size, search, sort, state, creator, group, all)` | 分页获取所有项目（root 限定），返回迭代器 |
| `get_groups(page, size, search, type, sort, all)` | 分页获取所有空间（root 限定），返回迭代器 |
| `get_usage_summary()` | 获取系统汇总信息（root 限定），返回 `ApiResponseType` |

:::code-group

```python [实例信息]
import swanlab

api = swanlab.Api()
sh = api.self_hosted()

data = sh.json()
print(data["enabled"], data["plan"], data["seats"])
```

```python [用户管理]
import swanlab

api = swanlab.Api()
sh = api.self_hosted()

sh.create_user(username="newuser", password="pass123")

for user in sh.get_users(page=1, size=20, all=True):
    print(user)
```

```python [项目/空间管理]
import swanlab

api = swanlab.Api()
sh = api.self_hosted()

for proj in sh.get_projects(page=1, size=20, all=True, search="image"):
    print(proj)

for group in sh.get_groups(page=1, size=20, all=True):
    print(group)
```

```python [系统概况]
import swanlab

api = swanlab.Api()
sh = api.self_hosted()

result = sh.get_usage_summary()
print(result.data if result.ok else result.errmsg)
```

:::


## 统一类型说明

**ApiResponseType** — 统一响应封装，所有 API 调用保证不抛异常：

```python
result = run.export_logs(start=0, rows=500)

if result.ok:
    print(result.data)   # 正常数据
else:
    print(result.errmsg) # 错误信息
```
