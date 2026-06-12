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

新版OpenAPI围绕三个核心概念组织：**工作空间（Workspace）** → **项目（Project）** → **实验（Experiment/Run）**，形成清晰的层级关系：

```
Workspace（工作空间）
 └── Project（项目）
      └── Experiment/Run（实验）
           ├── columns（指标列，即指标的`key`/名称）
           ├── metrics（标量指标）
           ├── medias（媒体指标）
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

# 媒体指标：图片、音频等
media = run.medias(keys=["generated_image"], step=10)

# 日志：文本输出
logs = run.logs(offset=0, level="INFO")
```

**注意事项：**
- `metrics()` 默认采样返回，采样值 `sample` 参数默认上限为 1500，超过会自动截断；可通过设置 `all=True` 返回全量数据，或 `range_query`  按照 `step` 或 `timestamp` 返回精确指标区间；
- `medias()` 返回的媒体数据通过预签名 URL 提供，需在有效期内下载；
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
| `workspace_type` | `str` | 工作空间类型，`PERSON` 或 `TEAM` |
| `role` | `str` | 当前用户RBAC角色，`OWNER` 或 `MEMBER` |
| `profile` | `dict` | 介绍信息 |
| `comment` | `str` | 空间简介 |


:::code-group

```python [获取单个工作空间]
import swanlab

api = swanlab.Api()

# username: 指定工作空间用户名，为 None 时使用当前登录用户
ws = api.workspace(username="my-team")

data = ws.json()
print(data["name"], data["username"], data["workspace_type"])
```

```python [遍历工作空间列表]
import swanlab

api = swanlab.Api()

# username: 指定用户名，为 None 时使用当前登录用户
for ws in api.workspaces("my-team"):
    print(ws.name)
```

```python [获取工作空间下的项目列表]
import swanlab

api = swanlab.Api()


ws = api.workspace(username="my-team")

# sort 可选参数:
# - create: 表示按照创建时间排序
# - name: 表示按照名称排序
# None: 为空时默认按照 「最近更新」排序

# search: 模糊搜索关键词
projects = ws.projects(sort="create", search="v1")
print(projects.json())

```

```python [创建项目]
import swanlab

api = swanlab.Api()


ws = api.workspace(username="my-team")

project = ws.create_project(name="my_project", visibility="PUBLIC")
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

:::code-group

```python [获取项目]
import swanlab

api = swanlab.Api()

# path: 格式为 'username/project-name'
project = api.project(path="my-team/my-project")

print(project.json())
```

```python [获取项目列表]
import swanlab

api = swanlab.Api()

"""
- path: 工作空间名称 'username'
- sort: 排序方式，支持 `create` 或 `name`, 分别表示创建时间或名称, 默认按照更新时间排序
- search: 模糊搜索关键词
- detail: 是否返回详细信息, bool 类型
- page: 起始页码，默认 1
- size: 每页数量，默认 100
- all: 是否获取全部数据，默认 False
"""
for p in api.projects(path="my-team", sort="name", search="image"):
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
project.delete(commit=False)  # commit=False 仅打印待删除项目的信息，不实际执行删除操作
```

```python [批量删除实验]
import swanlab

api = swanlab.Api()

project = api.project(path="my-team/my-project")
project.delete_runs(["run_id_1", "run_id_2"], commit=True) # commit=True 确认删除，操作前务必确认
```

:::


## 实验(Run/Experiment)

**Run 属性：**

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


### 获取单个实验

```python
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

data = run.json()
print(data["name"], data["state"], data["created_at"])
```

### 获取实验列表 — 过滤模式

通过条件过滤获取项目下的实验列表。

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
import swanlab

api = swanlab.Api()

for run in api.runs(path="my-team/my-project"):
    print(run.name, run.state)
```

```python
# 过滤示例
for run in api.runs(path="my-team/my-project", filters=[
    {"key": "state", "type": "STABLE", "op": "EQ", "value": ["FINISHED"]},
    {"key": "name", "type": "STABLE", "op": "CONTAIN", "value": ["v2"]},
]):
    print(run.name)
```

### 获取实验列表 — 分页模式
通过标准分获取项目下的实验列表，返回精简信息。**不支持过滤**。

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `path` | `str` | — | 项目路径 `username/project` |
| `page` | `int` | `1` | 起始页码 |
| `size` | `int` | `100` | 每页条数 |
| `all` | `bool` | `False` | 是否自动翻页获取全部 |


```python
import swanlab

api = swanlab.Api()

for run in api.runs_get(path="my-team/my-project", page=1, size=100, all=True):
    print(run.name, run.state)
```



### metrics — 标量指标

获取标量指标数据（如 loss、acc），支持采样控制、范围查询，返回结构化数据。

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `keys` | `list[str]` | — | 指标 key 名称列表，如 `["loss", "acc"]` |
| `sample` | `int` | `1500` | 采样数量（SCALAR 最大 1500），`all` 或 `range_query` 时忽略 |
| `all` | `bool` | `False` | 获取全量数据（不受采样限制） |
| `range_query` | `dict` 或 `RangeQuery` | `None` | 范围查询，仅对 SCALAR 类型有效 |
| `ignore_timestamp` | `bool` | `False` | 是否去除时间戳字段 |




**RangeQuery 字段：**

| 字段 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `type` | `str` | `"step"` | 过滤轴：`"step"` 或 `"timestamp"` |
| `start` | `int` | `None` | 下界（含），`None` 表示从头开始截断，type 为 `timestamp` 时要求**输入为 UNIX 时间戳** |
| `end` | `int` | `None` | 上界（含），`None` 表示截断到最后一个step，type 为 `timestamp` 时要求**输入为 UNIX 时间戳** |
| `last` | `int` | `None` | 最近 N 毫秒（与 `start`/`end` 互斥） |
| `head` | `int` | `None` | 取前 N 个数据点（与 `tail` 互斥） ，后采样|
| `tail` | `int` | `None` | 取后 N 个数据点（与 `head` 互斥） ，后采样|

**互斥规则：**
- `last` 与 `start`/`end` 互斥
- `head` 与 `tail` 互斥，具有最低优先级
- `head`/`tail` 可与 `start`/`end` 或 `last` 组合（先范围过滤，再截取）


```python
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

# 获取指标数据，返回 dict
result = run.metrics(keys=["loss", "acc"])

# 指定采样数（默认 1500，最大 1500）
result = run.metrics(keys=["loss"], sample=500)

# 全量数据（不受采样限制）
result = run.metrics(keys=["loss"], all=True)

# 按步数范围查询
result = run.metrics(
    keys=["loss"],
    range_query={"type": "step", "start": 100, "end": 500},
)

# 按时间戳范围查询（毫秒，不足 13 位自动补齐）
result = run.metrics(
    keys=["loss"],
    range_query={"type": "timestamp", "start": 1715769600000, "end": 1715773200000},
)

# 最近 5 分钟的数据
result = run.metrics(keys=["loss"], range_query={"last": 300_000})

# 步数范围 + 取前 50 个点
result = run.metrics(
    keys=["loss"],
    range_query={"start": 0, "end": 500, "head": 50},
)

# 取最后 30 个数据点
result = run.metrics(keys=["loss"], range_query={"tail": 30})
```



```python
from swanlab.api.typings.common import RangeQuery

# 按步数范围
rq = RangeQuery(type="step", start=100, end=500)

# 取最后 50 条
rq = RangeQuery(tail=50)

# 按时间戳范围（毫秒，不足 13 位自动补齐）
rq = RangeQuery(type="timestamp", start=1715769600000, end=1715773200000)

# 最近 5 分钟
rq = RangeQuery(last=300_000)

# 或直接使用 dict
result = run.metrics(keys=["loss"], range_query={"type": "step", "start": 100})
```

```python
import swanlab

api = swanlab.Api()
run = api.run(path="my-team/my-project/abc123")

# 默认采样（最多 1500 点）
result = run.metrics(keys=["loss", "acc"])

# 指定采样数
result = run.metrics(keys=["loss"], sample=500)

# 全量数据
result = run.metrics(keys=["loss"], all=True)

# 按步数范围
result = run.metrics(keys=["loss"], range_query={"start": 100, "end": 500})

# 最近 5 分钟
result = run.metrics(keys=["loss"], range_query={"last": 300_000})

# 取最后 30 个数据点
result = run.metrics(keys=["loss"], range_query={"tail": 30})

# 按时间戳范围（毫秒）
result = run.metrics(keys=["loss"], range_query={
    "type": "timestamp", "start": 1715769600000, "end": 1715773200000,
})

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

获取图片、音频、视频、echarts 等非结构化媒体数据，存储在对象存储中，响应仅返回预签名 URL。

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `keys` | `list[str]` | — | 媒体指标 key 列表 |
| `step` | `int` | `0` | 指定 step，不传则返回最新 |
| `all` | `bool` | `False` | 获取全部历史媒体数据 |

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

> 返回的媒体数据通过预签名 URL 提供，需在有效期内下载。

### logs — 日志

获取实验运行时的文本日志，支持按级别筛选；也可导出为 `.log` 文件。

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `offset` | `int` | `0` | 分页偏移量 |
| `level` | `str` | `"INFO"` | 日志级别：`DEBUG`、`INFO`、`WARN`、`ERROR` |
| `ignore_timestamp` | `bool` | `False` | 是否去除时间戳字段 |

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `start` | `int` | `0` | 导出起始行号（0-based） |
| `rows` | `int` | `500000` | 导出行数，最大 500000 |

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




### columns — 指标列

获取实验下的指标列列表，或通过 key 获取单个列。

**columns 参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `page` | `int` | `1` | 起始页码 |
| `size` | `int` | `100` | 每页条数 |
| `search` | `str` | `None` | 模糊搜索关键词（大小写不敏感，匹配列 **name** 字段） |
| `column_class` | `str` | `None` | 列分类：`CUSTOM` 或 `SYSTEM` |
| `column_type` | `str` | `None` | 列数据类型：`FLOAT`、`STRING`、`IMAGE` 等 |
| `all` | `bool` | `False` | 是否自动翻页获取全部 |

**column 参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `path` | `str` | — | 实验路径 `username/project/run_id` |
| `key` | `str` | — | 搜索关键词（模糊匹配列名，返回第一个匹配项） |
| `column_class` | `str` | `"CUSTOM"` | 列分类：`CUSTOM` 或 `SYSTEM` |
| `column_type` | `str` | `"FLOAT"` | 列数据类型：`FLOAT`、`STRING`、`IMAGE` 等 |

```python
import swanlab

api = swanlab.Api()
run = api.run(path="my-team/my-project/abc123")

# 获取所有指标列
for col in run.columns(all=True):
    print(col.name)

# 模糊搜索（匹配列 name）
for col in run.columns(search="loss"):
    print(col.name)

# 获取单个列（模糊匹配，返回第一个）
col = run.column(key="loss", column_type="FLOAT")
```


### delete — 删除实验
```python
import swanlab

api = swanlab.Api()
run = api.run(path="my-team/my-project/abc123")

run.delete(commit=False) # commit=False 时不实际执行删除
```


## 指标列(Column)
表示通过 `swanlab.log()` 上报的指标名称。

**column 入参：**
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `path` | `str` | — | 实验路径 `username/project/run_id` |
| `key` | `str` | — | 搜索关键词（模糊匹配列名，返回第一个匹配项） |
| `column_class` | `str` | `"CUSTOM"` | 列分类：`CUSTOM` 或 `SYSTEM` |
| `column_type` | `str` | `"FLOAT"` | 列数据类型：`FLOAT`、`STRING`、`IMAGE`、`VIDEO` 等 |

**columns 参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `path` | `str` | — | 实验路径 `username/project/run_id` |
| `page` | `int` | `1` | 起始页码 |
| `size` | `int` | `100` | 每页条数 |
| `search` | `str` | `None` | 模糊搜索关键词（大小写不敏感，匹配列 **name** 字段） |
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


:::code-group

```python [获取单个列]
import swanlab

api = swanlab.Api()

col = api.column(
    path="my-team/my-project/abc123",
    key="loss",
    column_type="FLOAT",
)
```

```python [遍历列列表]
import swanlab

api = swanlab.Api()

for col in api.columns(
    path="my-team/my-project/abc123",
    page=1,
    size=100,
    all=True,
    column_type="FLOAT",
):
    print(col.name)
```

```python [模糊搜索列]
import swanlab

api = swanlab.Api()

# search 参数对列名做大小写不敏感的 contains 匹配
for col in api.columns(
    path="my-team/my-project/abc123",
    search="loss",
):
    print(col.name)
```


```python [遍历列表]
import swanlab

api = swanlab.Api()

for col in api.columns(
    path="my-team/my-project/abc123",
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

:::code-group
```python[获取用户信息]
import swanlab

api = swanlab.Api()
user = api.user() # 无入参，以通过 Api 实例化的用户信息为准

data = user.json()
```
:::


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

for user in sh.get_users(page=1, size=100, all=True):
    print(user)
```

```python [项目/空间管理]
import swanlab

api = swanlab.Api()
sh = api.self_hosted()

for proj in sh.get_projects(page=1, size=100, all=True, search="image"):
    print(proj)

for group in sh.get_groups(page=1, size=100, all=True):
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

**ApiResponseType** — OpenAPI 的统一响应封装，所有返回 `ApiResponseType` 的 API 调用保证不抛异常，实际返回数据在 `data` 字段中。

| 属性 | 类型 | 描述 |
|------|------|------|
| `ok` | `bool` | 请求是否成功 |
| `errmsg` | `str` | 错误信息，成功时为空字符串 |
| `data` | `Any` | 响应数据，失败时为 `None` |

| 方法 | 描述 |
|------|------|
| `json()` | 序列化为 `dict`，自动将实体 `data` 调用 `.json()` 转换 |
