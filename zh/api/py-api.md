# swanlab.Api

::: warning 版本提示
此文档适用于swanlab>=`0.7.8`版本。
:::

基于 SwanLab 云端功能, 在 SDK 端提供访问 **开放 API（OpenAPI）** 的能力, 允许用户通过编程方式在本地环境中操作云端 **实验/项目/空间/指标** 资源。

![](./py-openapi/logo.jpg)

通过开放 API 的形式, 用户可以在本地编程环境中:

- 获取实验数据、个人信息、工作空间信息、项目列表等
- 进行实验的自动管理（如查询、组织、元数据编辑等）
- 更方便地与其他工具集成（如 CI/CD、实验调度等）

利用好此特性可极大提升 SDK 的灵活性和可扩展性, 方便构建高级用法或扩展体系


## 核心术语
- **空间（workspace）**：项目的集合，对应一个研发团队（如“SwanLab”），分为个人空间（PERSON）和组织空间（TEAM）；
- **项目（project）**：实验的集合，对应一个研发任务（如“图像分类”）；
- **实验（run）**：单次训练/推理任务，包含指标、配置、日志等数据；
- **实验ID（experiment_id）**：实验的唯一标识符，用于精准定位单个实验，可以在WebUI上实验的"环境"选项卡找到，也可以在URL中找到，格式为`https://swanlab.cn/@username/project_name/runs/experiment_id/···`。

## 介绍

> 前置条件：需要在编程环境下登录过SwanLab账号。

要使用 SwanLab 的开放 API, 只需实例化一个 `Api` 对象。

```python
import swanlab

api = swanlab.Api() # 使用本地（swanlab login）登录信息
```

如果你需要获取指定用户的数据：

```python
import swanlab

other_api = swanlab.Api(api_key='other_api_key') # 使用一个账户的api_key
```

具体来说, **OpenApi**的认证逻辑如下：

1. 如果显式提供了`api_key`参数, 则优先使用该`api_key`进行身份认证, 可以在[这里](https://swanlab.cn/space/~/settings)查看自己的 API 密钥；
2. 否则，使用本地的认证信息。

::: warning 私有化部署使用OpenAPI

对于私有化部署的SwanLab，可在`Api`中传入`host`参数。

```python
import swanlab

api = swanlab.Api(api_key='your-api-key', host='your-host')
```
:::

## workspace

:::code-group

```python [遍历所有空间]
import swanlab

api = swanlab.Api()

workspaces = api.workspaces()

for workspace in workspaces:
    print(workspace.username)
    print(workspace.name)
    print(workspace.profile)
```

```python [指定1个空间]
import swanlab

api = swanlab.Api()

workspace = api.workspace(username='username')

print(workspace.username)
print(workspace.name)
print(workspace.profile)
```

```python [获取空间下的项目]
import swanlab

api = swanlab.Api()

workspace = api.workspace(username='username')
projects = workspace.projects()

for project in projects:
    print(project.name)
    print(project.url)
```

:::


`workspaces`的参数：

| 参数 | 类型 | 描述 |
| --- | --- | --- |
| `username` | `str` | 空间用户名，即唯一ID，默认为当前登录用户 |

---

`workspace`的参数：

| 参数 | 类型 | 描述 |
| --- | --- | --- |
| `username` | `str` | 空间用户名，即唯一ID，默认为当前登录用户 |

`workspace`的属性与方法：

| 属性 | 类型 | 描述 |
| --- | --- | --- |
| `username` | `str` | 空间用户名，即唯一ID |
| `name` | `str` | 空间名称 |
| `role` | `str` | 当前登录用户在该空间中的角色，`OWNER` 或 `MEMBER` |
| `profile` | `dict` | 空间的介绍信息，包含简介、url、机构、邮箱 |
| `workspace_type` | `str` | 空间类型，分为个人空间和组织空间，`PERSON` 或 `TEAM` |
| `projects()` | - | 获取空间下的项目对象 |
| `json()` | - | 获取全部的空间信息，返回`dict` |


## project

:::code-group

```python [遍历所有项目]
import swanlab

api = swanlab.Api()

projects = api.projects(path='username')

for project in projects:
    print(project.name)
    print(project.created_at)
    print(project.url)
    print(project.visibility)
```

```python [指定1个项目]
import swanlab

api = swanlab.Api()

project = api.project(path='username/project_name')

print(project.name)
print(project.created_at)
print(project.url)
print(project.visibility)
```

```python [获取项目下的实验]
import swanlab

api = swanlab.Api()

project = api.project(path='username/project_name')
runs = project.runs()

for run in runs:
    print(run.name)
    print(run.url)
```
:::

`projects`的参数：

| 参数 | 类型 | 描述 |
|---|---|---|
| `path` | `str` | 空间路径（用户名），格式为`username`，用于筛选指定空间下的所有项目 |
| `sort` | `str` | 排序方式，可选：`created_at`（创建时间）、`updated_at`（更新时间） |
| `search` | `str` | 搜索关键词，模糊匹配项目名 |
| `detail` | `bool` | 是否返回项目详细信息（如描述、标签），默认为`True` |

---

`project`的参数：

| 参数 | 类型 | 描述 |
| --- | --- | --- |
| `path` | `str` | 项目路径，格式为`username/project_name` |


`project`的属性与方法：

| 属性 | 类型 | 描述 |
| --- | --- | --- | 
| `name` | `str` | 项目名 |
| `path` | `str` | 项目路径，格式为`username/project_name` |
| `description` | `str` | 项目描述 |
| `labels` | `list` | 项目标签，格式为`[label1, label2, ...]` |
| `created_at` | `str` | 项目创建时间，格式为ISO 8601 标准的 UTC 时间格式，如`2025-12-09T17:57:38.224Z` |
| `updated_at` | `str` | 项目更新时间，格式同`created_at` |
| `url` | `str` | 项目URL |
| `visibility` | `str` | 项目可见性，`PUBLIC` 或 `PRIVATE` |
| `count` | `dict` | 项目统计信息，包含实验个数、协作者数量等 |
| `runs()` | - | 获取项目下的实验对象 |
| `json()` | - | 获取全部的项目信息，返回`dict` |

## run

:::code-group

```python [遍历项目下所有实验]
import swanlab

api = swanlab.Api()

runs = api.runs(path='username/project_name')

for run in runs:
    print(run.name)
    print(run.id)
    print(run.created_at)
    print(run.url)
```

```python [指定1个实验]
import swanlab

api = swanlab.Api()

run = api.run(path='username/project_name/experiment_id')

print(run.name)
print(run.id)
print(run.created_at)
print(run.url)
```

```python [获取实验config]
import swanlab

api = swanlab.Api()

run = api.run(path='username/project_name/experiment_id')

print(run.profile.config)
```

```python [获取实验的环境数据]
import swanlab

api = swanlab.Api()

run = api.run(path='username/project_name/experiment_id')

# 获取实验的Python版本、硬件信息等数据
print(run.profile.metadata)
# 获取实验的Python环境信息
print(run.profile.requirements)
```
:::

`runs`的参数：

| 参数 | 类型 | 描述 |
| --- | --- | --- |
| `path` | `str` | 项目路径，格式为`username/project_name` |
| `filters` | `dict` | 筛选条件，比如`{'state': 'FINISHED', 'config.batch_size': '64'}` |

---

`run`的参数：

| 参数 | 类型 | 描述 |
| --- | --- | --- |
| `path` | `str` | 实验路径，格式为`username/project_name/experiment_id` |


`run`对象的属性如下：

| 属性 | 类型 | 描述 |
| --- | --- | --- | 
| `name` | `str` | 实验名 |
| `path` | `str` | 实验路径，格式为`username/project_name/experiment_id` |
| `description` | `str` | 实验描述 |
| `id` | `str` | 实验ID |
| `state` | `str` | 实验状态，`FINISHED`、`RUNNING`、`CRASHED`、`ABORTED`|
| `group` | `list` | 实验组，格式为`['A', 'B', 'C']` |
| `labels` | `list` | 实验标签，格式为`[label1, label2, ...]` |
| `created_at` | `str` | 实验创建时间，格式为ISO 8601 标准的 UTC 时间格式，如`2025-12-09T17:57:38.224Z` |
| `finished_at` | `str` | 实验结束时间，格式同`created_at`；如果实验未结束，则该字段为`None` |
| `url` | `str` | 实验URL |
| `job_type` | `str` | 任务类型（job_type），格式为`job_name` |
| `profile` | `dict` | 实验配置信息，包含`conda`、`config`、`metadata`、`requirements`属性 |
| `show` | `bool` | 实验在图表对比视图的显示状态，`True` 或 `False` |
| `user` | `dict` | 实验用户，格式为`{'is_self': True, 'username': 'username'}` |
| `metrics()` | - | 获取实验的指标数据，返回`pd.DataFrame` |
| `json()` | - | 获取全部的实验信息，返回`dict` |

## runs filters

通过条件筛选参数`filters`，可以在指定项目下筛选出符合条件的实验。

```python
import swanlab

api = swanlab.Api()

runs = api.runs(
    path='username/project_name', 
    filters={
        'state': 'FINISHED',
        'config.batch_size': '64',
        },
    )

```

`filters`支持的筛选条件如下：

- `state`：实验状态，可选：`FINISHED`、`RUNNING`、`CRASHED`、`ABORTED`
- `config.<配置名>`：配置名，需要`config.`前缀；会筛选出与配置值相等的实验，支持多层嵌套，如`config.data.run_id`

![config](./py-api/filter_config.png)

## metrics

获取实验的指标数据，返回值的类型为`pd.DataFrame`。

:::code-group
```python [获取1个指标数据]
import swanlab

api = swanlab.Api()

run = api.run(path='username/project_name/experiment_id')
metrics = run.metrics(keys=['loss'])

print(metrics)
```

```python [获取多个指标数据]
import swanlab

api = swanlab.Api()

run = api.run(path='username/project_name/experiment_id')
metrics = run.metrics(keys=['loss', 'acc'])

print(metrics)
```

```python [指定x轴]
import swanlab

api = swanlab.Api()

run = api.run(path='username/project_name/experiment_id')
metrics = run.metrics(keys=['loss'], x_axis='acc')

print(metrics)
```

```python [指定采样数量]
import swanlab

api = swanlab.Api()

run = api.run(path='username/project_name/experiment_id')
metrics = run.metrics(keys=['loss'], sample=100)

print(metrics)
```
:::

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `keys` | `list[str]` | `None` | 要获取的指标名称列表，如 `['loss', 'acc']`；不传则返回空DataFrame |
| `x_axis` | `str` | `step` | X轴维度，可选：`step`（步数）、指标名（如 `acc`） |
| `sample` | `int` | `None` | 采样数量，限制返回的行数；不传则返回全部数据 |

## user

> 此操作仅限于私有化部署的超级管理员使用

:::code-group
```python [列出所有用户]
import swanlab

# 确保你已经登录到私有化部署的SwanLab，且身份为超级管理员
api = swanlab.Api()

users = api.users()

for user in users:
    print(user.username)
    print(user.is_self)
    print(user.teams)
```

```python [创建用户]
import swanlab

# 确保你已经登录到私有化部署的SwanLab，且身份为超级管理员
api = swanlab.Api()

user = api.user()

# 创建1个用户名为testuser，密码为test123456的用户
user.create(username='testuser', password='test123456')

"""如需批量创建
for i in range(3):
    user.create(username=f'testuser{i}', password='test123456')
"""
```

```python [打印/生成/删除 API Key]
import swanlab

# 确保你已经登录到私有化部署的SwanLab，且身份为超级管理员
api = swanlab.Api()

user = api.user()

# 下面的操作目前仅支持对当前登录的账号进行
# 打印API Key
print(user.api_keys)

# 生成API Key
new_api_key = user.generate_api_key()

# 删除API Key
user.delete_api_key(api_key=new_api_key)
```

:::

`user`的参数：

| 参数 | 类型 | 描述 |
| --- | --- | --- |
| `username` | `str` | 用户名，默认为当前登录用户 |

`user`对象的属性如下：

| 属性 | 类型 | 描述 |
| --- | --- | --- |
| `username` | `str` | 用户名 |
| `is_self` | `bool` | 是否为当前登录用户 |
| `teams` | `list` | 用户所属团队 |
| `api_keys` | `list` | 用户API密钥列表，仅支持获取当前登录用户的API密钥 |