# swanlab.Api

::: warning Version Note
This documentation applies to swanlab >= `0.8.0`.
:::

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/en/api/py-openapi/logo.jpg)

The new OpenAPI uses an **OOP style** — all operations start from the `Api` entry point to retrieve entity objects. Entities support lazy loading (requests are sent when properties are accessed) and uniformly serialize to `dict` via `.json()`.

> Authentication priority: explicitly passed `api_key` / `host` > `swanlab.login()` session > environment variables `SWANLAB_API_KEY` / `SWANLAB_API_HOST`

## Quick Start

```python
import swanlab

api = swanlab.Api()                                # reads .netrc credentials automatically
api = swanlab.Api(api_key="your-api-key", host="your-host")  # explicit credentials
```

Self-hosted deployment:

```python
api = swanlab.Api(api_key="your-api-key", host="https://your-server.com")
```

## Path Format

- **Workspace**: `username`
- **Project**: `username/project-name`
- **Experiment**: `username/project-name/experiment-id`

## Core Concepts

The new OpenAPI is organized around three core concepts: **Workspace** → **Project** → **Experiment**, forming a clear hierarchy:

```
Workspace
 └── Project
      └── Experiment
           ├── metrics (scalar / media)
           ├── columns (metric columns)
           └── logs
```

| Concept | Description |
|---------|-------------|
| **Workspace** | A collection of projects, corresponding to a team or user. Divided into personal (`PERSON`) and organization (`TEAM`). |
| **Project** | A collection of experiments, corresponding to a research task, containing metadata like name, description, labels, and visibility. |
| **Experiment** | A single training/inference run, containing metrics, config, logs, environment info, etc. |
| **Column** | A metric column under an experiment, such as `loss` or `acc`, supporting FLOAT / STRING / IMAGE and more. |

### Lazy Loading

All entity objects use **lazy loading**: construction only records the path, and HTTP requests are sent when properties (e.g. `.name`, `.profile`) are accessed. This means you can safely construct objects without worrying about unnecessary network overhead.

```python
# No request sent at construction time
run = api.run(path="my-team/my-project/abc123")

# Request sent when .name is accessed
print(run.name)
```

### Serialization

Every entity provides a `.json()` method that recursively serializes all properties to a `dict`, ready for JSON output or data passing:

```python
project = api.project(path="my-team/my-project")
data = project.json()  # → dict
```

### Iterators

List operations (such as `workspaces`, `projects`, `runs`, `columns`) return **iterators** with auto-pagination support (`all=True`):

```python
# Auto-paginate to get all projects
for p in api.projects(path="my-team", all=True):
    print(p.name)
```

### Data Types: metrics / medias / logs

Data under an experiment falls into three categories, accessed through different methods:

| Type | Method | Description | Typical Use Case |
|------|--------|-------------|-----------------|
| **Scalar metrics** | `run.metrics(keys=[...])` | Numeric metrics (e.g. loss, acc), supports sampling and range queries, returns structured data | Training curve analysis, trend comparison |
| **Media metrics** | `run.medias(keys=[...])` | Images, audio, video and other unstructured media data, returns presigned URLs | Visual inspection, result preview |
| **Logs** | `run.logs()` / `run.export_logs()` | Text logs from experiment runtime, supports level filtering | Debugging, troubleshooting, auditing |

```python
import swanlab

api = swanlab.Api()
run = api.run(path="my-team/my-project/abc123")

# Scalar metrics: loss, acc, etc.
scalars = run.metrics(keys=["loss", "acc"], sample=500)
print(scalars["metrics"])  # [{step: 1, value: 0.9, timestamp: ...}, ...]

# Media metrics: images, audio, etc.
media = run.medias(keys=["generated_image"], step=10)
print(media["metrics"])  # [{index: 10, items: [{url: "https://...", ...}]}, ...]

# Logs: text output
logs = run.logs(offset=0, level="INFO")
print(logs["logs"])  # [{message: "...", level: "INFO", timestamp: ...}, ...]
```

**Notes:**

- `metrics()` `sample` parameter defaults to 1500 and is auto-capped; set `all=True` for full data.
- `range_query` only works for SCALAR type, supports filtering by step or timestamp range.
- `medias()` returns media data via presigned URLs — download within the validity period.
- `export_logs()` can export large volumes of logs to `.log` files, suitable for persistent storage.

## Api

`Api` is the entry point for all operations. Authentication is completed at construction time, creating an independent `Client` instance (separate from the SDK runtime).

| Method | Description |
|--------|-------------|
| `api.workspace(username)` | Get a single workspace |
| `api.workspaces(username)` | List workspaces (iterator) |
| `api.project(path)` | Get a single project |
| `api.projects(path, ...)` | Paginated project list with search/sort |
| `api.create_project(username, name, ...)` | Create a project |
| `api.run(path)` | Get a single experiment |
| `api.runs(path, filters=...)` | Get experiment list (POST filter mode) |
| `api.runs_get(path, ...)` | Get experiment list (GET pagination mode) |
| `api.column(path, key)` | Get a single metric column |
| `api.columns(path, ...)` | Get metric columns for an experiment |
| `api.user()` | Get current user info |
| `api.self_hosted()` | Self-hosted management entry point |

---

## workspace

**Workspace properties:**

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Workspace name |
| `username` | `str` | Workspace username (unique ID) |
| `workspace_type` | `str` | `PERSON` or `TEAM` |
| `role` | `str` | Current user role: `OWNER` or `MEMBER` |
| `profile` | `dict` | Profile info |
| `comment` | `str` | Workspace description |

**Workspace methods:**

| Method | Description |
|--------|-------------|
| `projects(sort, search, detail, page, size, all)` | Get project list (iterator) |
| `create_project(name, visibility, description)` | Create project, returns `Project` or `None` |
| `json()` | Serialize to `dict` |

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `username` | `str` | current user | Workspace username |

:::code-group

```python [Get single workspace]
import swanlab

api = swanlab.Api()

ws = api.workspace(username="my-team")

data = ws.json()
print(data["name"], data["username"], data["workspace_type"])
```

```python [Iterate workspace list]
import swanlab

api = swanlab.Api()

for ws in api.workspaces("my-team"):
    print(ws.name)
```

:::

---

## project

**Project properties:**

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Project name |
| `path` | `str` | Project path `username/project-name` |
| `description` | `str` | Project description |
| `labels` | `list` | Project labels |
| `created_at` | `str` | Creation time (ISO 8601 UTC) |
| `updated_at` | `str` | Update time |
| `url` | `str` | Project web page URL |
| `visibility` | `str` | `PUBLIC` or `PRIVATE` |
| `count` | `dict` | Stats (experiment count, collaborator count, etc.) |

**Project methods:**

| Method | Description |
|--------|-------------|
| `runs(filters)` | Get experiment list (POST filter mode) |
| `runs_get(page, size, all)` | Get experiment list (GET pagination mode) |
| `delete_runs(run_ids, commit)` | Batch delete experiments |
| `delete(commit)` | Delete project |
| `json()` | Serialize to `dict` |

**projects parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Workspace username, e.g. `"my-team"` |
| `sort` | `str` | — | Sort field: `created_at`, `updated_at`, `name` |
| `search` | `str` | — | Search keyword (fuzzy match project name) |
| `detail` | `bool` | `True` | Whether to return detailed info |
| `page` | `int` | `1` | Start page |
| `size` | `int` | `20` | Items per page |
| `all` | `bool` | `False` | Auto-paginate to get all results |

:::code-group

```python [Get project]
import swanlab

api = swanlab.Api()

project = api.project(path="my-team/my-project")

print(project.name)
print(project.description)
print(project.labels)
print(project.visibility)  # PUBLIC or PRIVATE
print(project.created_at)
print(project.url)
print(project.count)       # experiment count, collaborator count, etc.
```

```python [Get project list]
import swanlab

api = swanlab.Api()

for p in api.projects(path="my-team", sort="updated_at", search="image"):
    print(p.name, p.path)
```

```python [Get experiment list (filter mode)]
import swanlab

api = swanlab.Api()

project = api.project(path="my-team/my-project")

for run in project.runs():
    print(run.name, run.state, run.created_at)
```

```python [Create project]
import swanlab

api = swanlab.Api()

project = api.create_project(
    username="my-team",
    name="new-project",
    visibility="PRIVATE",
    description="Project description",
)
```

```python [Delete project]
import swanlab

api = swanlab.Api()

project = api.project(path="my-team/my-project")
project.delete(commit=True)  # commit=False only prints pending deletion info
```

```python [Batch delete experiments]
import swanlab

api = swanlab.Api()

project = api.project(path="my-team/my-project")
project.delete_runs(["run_id_1", "run_id_2"], commit=True)
```

:::

---

## run

**Experiment properties:**

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Experiment name |
| `description` | `str` | Experiment description |
| `state` | `str` | Experiment state: `RUNNING`, `FINISHED`, `CRASHED`, `ABORTED`, `OFFLINE` |
| `labels` | `list` | Experiment labels |
| `group` | `str` | Experiment group name |
| `job_type` | `str` | Job type |
| `created_at` | `str` | Creation time |
| `finished_at` | `str` | End time, `None` if not finished |
| `url` | `str` | Experiment web page URL |
| `show` | `bool` | Whether shown in comparison view |
| `profile` | `dict` | Experiment config, environment, dependencies, etc. |
| `user` | `dict` | Creator info |

**Experiment methods:**

| Method | Description |
|--------|-------------|
| `metrics(keys, sample, all, range_query)` | Get scalar metric data, returns `dict` |
| `summary(keys)` | Get scalar metric summary stats, returns `dict` |
| `logs(offset, level)` | Get log data, returns `dict` |
| `export_logs(start, rows)` | Export logs as `.log` file, returns `ApiResponseType` |
| `medias(keys, step, all)` | Get media metric data, returns `dict` |
| `columns(page, size, search, column_type, column_class, all)` | Get metric column list (iterator) |
| `column(key, column_class, column_type)` | Get a single metric column |
| `delete(commit)` | Delete experiment |
| `json()` | Serialize to `dict` |

### Get single experiment

```python
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

data = run.json()
print(data["name"], data["state"], data["created_at"])
```

### Get experiment list — Filter mode

Fetch experiment list under a project via conditional filtering.

```python
import swanlab

api = swanlab.Api()

for run in api.runs(path="my-team/my-project"):
    print(run.name, run.state)
```

```python
# With filters
for run in api.runs(path="my-team/my-project", filters=[
    {"key": "state", "type": "STABLE", "op": "EQ", "value": ["FINISHED"]},
]):
    print(run.name)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Project path `username/project` |
| `filters` | `list[dict]` | Filter rules, each `{key, type, op, value}` |

**Filter rule fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `key` | `str` | ✓ | Field name |
| `type` | `str` | ✓ | Field type: `STABLE`, `CONFIG`, `SCALAR` |
| `op` | `str` | ✓ | Operator: `EQ`, `NEQ`, `GTE`, `LTE`, `IN`, `NOT IN`, `CONTAIN` |
| `value` | `list` | ✓ | Comparison value |

**`type` field values:**

| type | `key` values | `value` |
|------|-------------|---------|
| `STABLE` | Experiment built-in fields: `state`, `name`, `description`, `show`, `pin`, `baseline`, `colors`, `cluster`, `job`, `createdAt`, `updatedAt`, `finishedAt`, `pinnedAt`, `labels` | Corresponding field value |
| `CONFIG` | Config parameter name, e.g. `param_2` (no `config.` prefix) | Config parameter value |
| `SCALAR` | Scalar metric name | Metric value |

### Get experiment list — Pagination mode

Fetch experiment list via standard pagination, returns minimal info.

```python
import swanlab

api = swanlab.Api()

for run in api.runs_get(path="my-team/my-project", page=1, size=20, all=True):
    print(run.name, run.state)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Project path `username/project` |
| `page` | `int` | `1` | Start page |
| `size` | `int` | `20` | Items per page |
| `all` | `bool` | `False` | Auto-paginate to get all results |

---

### metrics — Scalar metrics

Fetch numeric metrics (e.g. loss, acc), supports sampling control and range queries, returns structured data.

```python
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

# Get metric data, returns dict
result = run.metrics(keys=["loss", "acc"])
print(result["metrics"])  # metric list

# Specify sample count (default 1500, max 1500)
result = run.metrics(keys=["loss"], sample=500)

# Full data (no sampling limit)
result = run.metrics(keys=["loss"], all=True)

# Range query
result = run.metrics(
    keys=["loss"],
    range_query={"type": "step", "start": 100, "end": 500},
)

# Timestamp range query
result = run.metrics(
    keys=["loss"],
    range_query={"type": "timestamp", "start": 1715769600000, "end": 1715773200000},
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keys` | `list[str]` | — | Metric key list, e.g. `["loss", "acc"]` |
| `sample` | `int` | `1500` | Sample count (SCALAR max 1500) |
| `all` | `bool` | `False` | Get full data (no sampling limit) |
| `range_query` | `dict` or `RangeQuery` | `None` | Range query: `{"type": "step", "start": 100, "end": 500}` |
| `x_axis` | `str` | `"step"` | X-axis dimension: `step` (step count) |
| `ignore_timestamp` | `bool` | `False` | Whether to remove timestamp fields |

> `range_query` only works for SCALAR type. Supports two formats: `dict` or `RangeQuery` object.

**RangeQuery usage:**

```python
from swanlab.api.typings.common import RangeQuery

# By step range
rq = RangeQuery(type="step", start=100, end=500)

# Last 50 items
rq = RangeQuery(tail=50)

# By timestamp range (milliseconds)
rq = RangeQuery(type="timestamp", start=1715769600000, end=1715773200000)

# Or use dict directly
result = run.metrics(keys=["loss"], range_query={"type": "step", "start": 100})
```

> `head` and `tail` are mutually exclusive. Timestamps with fewer than 13 digits are auto-padded to millisecond precision.

### summary — Summary stats

Get statistical summary of scalar metrics (min / max / avg / median / latest).

```python
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

summary = run.summary(keys=["loss", "acc"])
# Returns stats for each key
print(summary)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `keys` | `list[str]` | Scalar keys to query, `None` means all keys |

### medias — Media metrics

Fetch images, audio, video and other unstructured media data, returns presigned URLs.

```python
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

# Get media data at a specific step
result = run.medias(keys=["generated_image"], step=10)
print(result)

# Get all media data
result = run.medias(keys=["generated_image"], all=True)
print(result)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keys` | `list[str]` | — | Media metric key list |
| `step` | `int` | `0` | Specific step, omit to get latest |
| `all` | `bool` | `False` | Get all historical media data |

> Returned media data is provided via presigned URLs — download within the validity period.

### logs — Logs

Fetch text logs from experiment runtime, supports level filtering; can also export as `.log` file.

```python
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

# Get logs
logs = run.logs(offset=0, level="INFO")
print(logs)

# Export logs as .log file (returns presigned download URL)
result = run.export_logs(start=0, rows=500000)
if result.ok:
    print(result.data["url"])
```

**logs parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `offset` | `int` | `0` | Pagination offset |
| `level` | `str` | `"INFO"` | Log level: `DEBUG`, `INFO`, `WARN`, `ERROR` |
| `ignore_timestamp` | `bool` | `False` | Whether to remove timestamp fields |

**export_logs parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `int` | `0` | Export start line (0-based) |
| `rows` | `int` | `500000` | Number of rows to export, max 500000 |

### columns — Metric columns

Get the list of metric columns under an experiment, or fetch a single column by key.

```python
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

# Get all metric columns
for col in run.columns(page=1, size=20, all=True):
    print(col.name, col.column_type)

# Get a single column
col = run.column(key="loss", column_type="FLOAT")
print(col.name, col.column_type, col.created_at)
print(col.metric())  # get metric data for this column
```

**columns parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | `int` | `1` | Start page |
| `size` | `int` | `20` | Items per page |
| `search` | `str` | `None` | Search keyword (matches column name) |
| `column_class` | `str` | `None` | Column class: `CUSTOM` or `SYSTEM` |
| `column_type` | `str` | `None` | Column data type: `FLOAT`, `STRING`, `IMAGE`, etc. |
| `all` | `bool` | `False` | Auto-paginate to get all results |

**column parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Experiment path `username/project/run_id` |
| `key` | `str` | — | Column key, e.g. `"loss"` |
| `column_class` | `str` | `"CUSTOM"` | Column class: `CUSTOM` or `SYSTEM` |
| `column_type` | `str` | `None` | Column data type: `FLOAT`, `STRING`, `IMAGE`, etc. |

### delete — Delete experiment

```python
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")
run.delete(commit=True)  # commit=False only prints pending deletion info
```

---

## column

**column parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Experiment path `username/project/run_id` |
| `key` | `str` | — | Column key, e.g. `"loss"` |
| `column_class` | `str` | `"CUSTOM"` | Column class: `CUSTOM` or `SYSTEM` |
| `column_type` | `str` | `None` | Column data type: `FLOAT`, `STRING`, `IMAGE`, `VIDEO`, etc. |

**columns parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Experiment path `username/project/run_id` |
| `page` | `int` | `1` | Start page |
| `size` | `int` | `20` | Items per page |
| `search` | `str` | `None` | Search keyword (matches column name) |
| `column_class` | `str` | `None` | Column class: `CUSTOM` or `SYSTEM` |
| `column_type` | `str` | `None` | Column data type |
| `all` | `bool` | `False` | Auto-paginate to get all results |

**Column properties:**

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Column display name |
| `key` | `str` | Column key |
| `column_class` | `str` | Column class: `CUSTOM` or `SYSTEM` |
| `column_type` | `str` | Data type: `FLOAT`, `STRING`, `IMAGE`, `VIDEO`, `OBJECT3D`, etc. |
| `created_at` | `int` | Creation timestamp |
| `error` | `dict` | Error info (if any) |

**Column methods:**

| Method | Description |
|--------|-------------|
| `metric(sample, metric_type)` | Get metric data for this column, returns `dict` |
| `export_csv()` | Export SCALAR column as CSV, returns `ApiResponseType` |
| `json()` | Serialize to `dict` |

:::code-group

```python [Get single column]
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

```python [Iterate column list]
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

```python [Export CSV]
import swanlab

api = swanlab.Api()

col = api.column(path="my-team/my-project/abc123", key="loss")
result = col.export_csv()
if result.ok:
    print(result.data["url"])  # CSV download URL
```

:::

---

## user

```python
import swanlab

api = swanlab.Api()
user = api.user()

data = user.json()
print(data["username"], data["email"])
```

**User properties:**

| Property | Type | Description |
|----------|------|-------------|
| `username` | `str` | Username |
| `name` | `str` | Display name |
| `bio` | `str` | Bio |
| `institution` | `str` | Institution |
| `school` | `str` | School |
| `email` | `str` | Email |
| `location` | `str` | Location |
| `url` | `str` | Personal website |

---

## self_hosted

> Applies only to self-hosted deployments and requires super admin privileges.

**SelfHosted properties:**

| Property | Type | Description |
|----------|------|-------------|
| `enabled` | `bool` | Whether self-hosted mode is enabled |
| `expired` | `bool` | Whether license is expired |
| `root` | `bool` | Whether current user is super admin |
| `plan` | `str` | License type: `free` or `commercial` |
| `seats` | `int` | License seat count |

**SelfHosted methods:**

| Method | Description |
|--------|-------------|
| `create_user(username, password)` | Create user (root only), returns `ApiResponseType` |
| `get_users(page, size, all)` | Paginate user list (root only), returns iterator |
| `get_projects(page, size, search, sort, state, creator, group, all)` | Paginate all projects (root only), returns iterator |
| `get_groups(page, size, search, type, sort, all)` | Paginate all workspaces (root only), returns iterator |
| `get_usage_summary()` | Get system summary (root only), returns `ApiResponseType` |

:::code-group

```python [Instance info]
import swanlab

api = swanlab.Api()
sh = api.self_hosted()

data = sh.json()
print(data["enabled"], data["plan"], data["seats"])
```

```python [User management]
import swanlab

api = swanlab.Api()
sh = api.self_hosted()

sh.create_user(username="newuser", password="pass123")

for user in sh.get_users(page=1, size=20, all=True):
    print(user)
```

```python [Project/Workspace management]
import swanlab

api = swanlab.Api()
sh = api.self_hosted()

for proj in sh.get_projects(page=1, size=20, all=True, search="image"):
    print(proj)

for group in sh.get_groups(page=1, size=20, all=True):
    print(group)
```

```python [System summary]
import swanlab

api = swanlab.Api()
sh = api.self_hosted()

result = sh.get_usage_summary()
print(result.data if result.ok else result.errmsg)
```

:::

---

## Type Reference

**ApiResponseType** — Unified response wrapper, all API calls guarantee no exceptions:

```python
result = api.export_logs(start=0, rows=500000)

if result.ok:
    print(result.data)   # normal data
else:
    print(result.errmsg) # error message
```
