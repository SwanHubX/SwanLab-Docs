# swanlab.Api

::: warning Version Note
This documentation applies to swanlab >= `0.8.0`.
:::

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/en/api/py-openapi/logo.jpg)

- The new OpenAPI uses an **OOP style** — all operations start from the `Api` entry point to retrieve entity objects. Entities support lazy loading (requests are sent when properties are accessed) and uniformly serialize to `dict` via `.json()`.
- All operations are also available via [`swanlab api`](./cli-swanlab-api.md) **as CLI commands**, useful for scripts, CI/CD, or any scenario where writing Python code is not needed.

> Authentication priority: explicitly passed `api_key` / `host` > `swanlab.login()` session > environment variables `SWANLAB_API_KEY` / `SWANLAB_API_HOST`

## 🚀 Quick Start

```python
import swanlab

api = swanlab.Api()                              # reads .netrc credentials automatically
api = swanlab.Api(api_key="your-api-key", host="your-host")  # explicit credentials
```

Self-hosted deployment:

```python
api = swanlab.Api(api_key="your-api-key", host="https://your-server.com")
```

## 🧱 Path Format

- **Workspace**: `username`
- **Project**: `username/project-name`
- **Experiment**: `username/project-name/experiment-id`

## 🔨 Core Concepts

The new OpenAPI is organized around three core concepts: **Workspace** → **Project** → **Experiment**, forming a clear hierarchy:

```
Workspace
 └── Project
      └── Experiment/Run
           ├── columns (metric column keys/names)
           ├── metrics (scalar metrics)
           ├── medias (media metrics)
           └── logs
```

| Concept        | Description                                                                                                                        |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Workspace**  | A collection of projects, corresponding to a team or user. Divided into personal (`PERSON`) and organization (`TEAM`).             |
| **Project**    | A collection of experiments, corresponding to a research task, containing metadata like name, description, labels, and visibility. |
| **Experiment** | A single training/inference run, containing metrics, config, logs, environment info, etc.                                          |
| **Column**     | A metric column under an experiment, such as `loss` or `acc`, supporting FLOAT / STRING / IMAGE and more.                          |
| **Metric**     | The metric value corresponding to a specific column in an experiment.                                                              |

### Data Types: metrics / medias / logs

Data under an experiment falls into three categories, accessed through different methods:

| Type               | Method                             | Description                                                                                    | Typical Use Case                          |
| ------------------ | ---------------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------- |
| **Scalar metrics** | `run.metrics(keys=[...])`          | Numeric metrics (e.g. loss, acc), supports sampling and range queries, returns structured data | Training curve analysis, trend comparison |
| **Media metrics**  | `run.medias(keys=[...])`           | Images, audio, video and other unstructured media data, returns presigned URLs                 | Visual inspection, result preview         |
| **Logs**           | `run.logs()` / `run.export_logs()` | Text logs from experiment runtime, supports level filtering                                    | Debugging, troubleshooting, auditing      |

```python
import swanlab

api = swanlab.Api()
run = api.run(path="my-team/my-project/abc123")

# Scalar metrics: loss, acc, etc.
scalars = run.metrics(keys=["loss", "acc"], sample=500)

# Media metrics: images, audio, etc.
media = run.medias(keys=["generated_image"], step=10)

# Logs: text output
logs = run.logs(offset=0, level="INFO")
```

**Notes:**

- `metrics()` returns sampled data by default, with a `sample` parameter default cap of 1500 (auto-truncated if exceeded). Set `all=True` for full data, or use `range_query` to query exact metric ranges by `step` or `timestamp`.
- `medias()` returns media data via presigned URLs — download within the validity period.
- `export_logs()` can export large volumes of logs to `.log` files, suitable for persistent storage.

## Api

`Api` is the entry point for all operations. Authentication is completed at construction time, creating an independent `Client` instance (separate from the SDK runtime).

| Method                                    | Description                               |
| ----------------------------------------- | ----------------------------------------- |
| `api.workspace(username)`                 | Get a single workspace                    |
| `api.workspaces(username)`                | List workspaces (iterator)                |
| `api.project(path)`                       | Get a single project                      |
| `api.projects(path, ...)`                 | Get project list (iterator)               |
| `api.create_project(username, name, ...)` | Create a project                          |
| `api.run(path)`                           | Get a single experiment                   |
| `api.runs(path, filters=...)`             | Get experiment list (POST filter mode)    |
| `api.runs_get(path, ...)`                 | Get experiment list (GET pagination mode) |
| `api.column(path, key)`                   | Get a single metric column                |
| `api.columns(path, ...)`                  | Get metric columns for an experiment      |
| `api.user()`                              | Get current user info                     |
| `api.self_hosted()`                       | Self-hosted management entry point        |

## Workspace

**Workspace properties:**

| Property         | Type   | Description                                 |
| ---------------- | ------ | ------------------------------------------- |
| `name`           | `str`  | Workspace name                              |
| `username`       | `str`  | Workspace username (unique ID)              |
| `workspace_type` | `str`  | `PERSON` or `TEAM`                          |
| `role`           | `str`  | Current user RBAC role: `OWNER` or `MEMBER` |
| `profile`        | `dict` | Profile info                                |
| `comment`        | `str`  | Workspace description                       |

:::code-group

```python [Get single workspace]
import swanlab

api = swanlab.Api()

# username: specify workspace username, uses current logged-in user when None
ws = api.workspace(username="my-team")

data = ws.json()
print(data["name"], data["username"], data["workspace_type"])
```

```python [Iterate workspace list]
import swanlab

api = swanlab.Api()

# username: specify username, uses current logged-in user when None
for ws in api.workspaces("my-team"):
    print(ws.name)
```

```python [Get project list under workspace]
import swanlab

api = swanlab.Api()


ws = api.workspace(username="my-team")

# sort optional parameters:
# - create: sort by creation time
# - name: sort by name
# None: defaults to "recently updated" sorting

# search: fuzzy search keyword
projects = ws.projects(sort="create", search="v1")
print(projects.json())

```

```python [Create project]
import swanlab

api = swanlab.Api()


ws = api.workspace(username="my-team")

project = ws.create_project(name="my_project", visibility="PUBLIC")
```

:::

## Project

### Project properties

| Property      | Type   | Description                                        |
| ------------- | ------ | -------------------------------------------------- |
| `name`        | `str`  | Project name                                       |
| `path`        | `str`  | Project path `username/project-name`               |
| `description` | `str`  | Project description                                |
| `labels`      | `list` | Project labels                                     |
| `created_at`  | `str`  | Creation time (ISO 8601 UTC)                       |
| `updated_at`  | `str`  | Update time                                        |
| `url`         | `str`  | Project web page URL                               |
| `visibility`  | `str`  | `PUBLIC` or `PRIVATE`                              |
| `count`       | `dict` | Stats (experiment count, collaborator count, etc.) |

### Project method examples

:::code-group

```python [Get project]
import swanlab

api = swanlab.Api()

# path: format is 'username/project-name'
project = api.project(path="my-team/my-project")

print(project.json())
```

```python [Get project list]
import swanlab

api = swanlab.Api()

"""
- path: workspace name 'username'
- sort: sort order, supports `create` or `name` for creation time or name sorting, defaults to update time
- search: fuzzy search keyword
- detail: whether to return detailed info, bool type
- page: start page, default 1
- size: items per page, default 100
- all: whether to get all data, default False
"""
for p in api.projects(path="my-team", sort="name", search="image"):
    print(p.name, p.path)
```

```python [Get experiment list (filter mode)]
import swanlab

api = swanlab.Api()

project = api.project(path="my-team/my-project")

"""
- filters: list of filter rules, each item is a dict with {key, type, op, value}
"""
# Example: return finished experiments under the project
runs = project.runs(filters=[{"key": "state", "type": "STABLE", "op": "EQ", "value": ["FINISHED"]}])
print(runs.json())
```

```python [Get experiment list (pagination mode)]
import swanlab

api = swanlab.Api()

project = api.project(path="my-team/my-project")

# Max page size is 100
runs = project.runs_get(page=1, size=100)
print(runs.json())
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
project.delete(commit=False)  # commit=False only prints pending deletion info, does not actually delete
```

```python [Batch delete experiments]
import swanlab

api = swanlab.Api()

project = api.project(path="my-team/my-project")
project.delete_runs(["run_id_1", "run_id_2"], commit=True)  # commit=True confirms deletion, verify before proceeding
```

:::

## Experiment (Run)

### Run properties

| Property      | Type   | Description                                                              |
| ------------- | ------ | ------------------------------------------------------------------------ |
| `name`        | `str`  | Experiment name                                                          |
| `description` | `str`  | Experiment description                                                   |
| `state`       | `str`  | Experiment state: `RUNNING`, `FINISHED`, `CRASHED`, `ABORTED`, `OFFLINE` |
| `labels`      | `list` | Experiment labels                                                        |
| `group`       | `str`  | Experiment group name                                                    |
| `job_type`    | `str`  | Job type                                                                 |
| `created_at`  | `str`  | Creation time                                                            |
| `finished_at` | `str`  | End time, `None` if not finished                                         |
| `url`         | `str`  | Experiment web page URL                                                  |
| `show`        | `bool` | Whether shown in comparison view                                         |
| `profile`     | `dict` | Experiment config, environment, dependencies, etc.                       |
| `user`        | `dict` | Creator info                                                             |

### Run method examples

#### 1. Get single experiment

Get a single experiment's info, input must match the format `username/project_name/run_id`
:::code-group

```python [Get single experiment]
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

data = run.json()
```

:::

#### 2. Get experiment list (filter mode)

Fetch experiment list under a project via conditional filtering.

| Parameter | Type         | Description                                                                             |
| --------- | ------------ | --------------------------------------------------------------------------------------- |
| `path`    | `str`        | Project path `username/project`                                                         |
| `filters` | `list[dict]` | Filter rules, each item is a `dict` containing exactly 4 keys: `{key, type, op, value}` |

**Filter rule fields:**

| Field   | Type   | Required | Description                                                                                                                                |
| ------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `key`   | `str`  | ✓        | Field name                                                                                                                                 |
| `type`  | `str`  | ✓        | Field type: `STABLE`, `CONFIG`, `SCALAR`                                                                                                   |
| `op`    | `str`  | ✓        | Operator: `EQ`(=), `NEQ`(≠), `GTE`(≥), `LTE`(≤), `IN`(element exists in set), `NOT IN`(element does not exist in set), `CONTAIN`(contains) |
| `value` | `list` | ✓        | Comparison value                                                                                                                           |

**`type` field values:**

| type     | `key` values                                                                                                                                                                    | `value`                   |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| `STABLE` | Experiment built-in fields: `state`, `name`, `description`, `show`, `pin`, `baseline`, `colors`, `cluster`, `job`, `createdAt`, `updatedAt`, `finishedAt`, `pinnedAt`, `labels` | Corresponding field value |
| `CONFIG` | Config parameter name, e.g. `param_2` (no `config.` prefix)                                                                                                                     | Config parameter value    |
| `SCALAR` | Scalar metric name                                                                                                                                                              | Metric value              |

:::code-group

```python [Experiment filter example]
# Filter example
# Experiments under my-team/my-project that are [finished] AND name contains `v2`
for run in api.runs(path="my-team/my-project", filters=[
    {"key": "state", "type": "STABLE", "op": "EQ", "value": ["FINISHED"]},
    {"key": "name", "type": "STABLE", "op": "CONTAIN", "value": ["v2"]},
]):
    print(run.name)
```

:::

#### 3. Get experiment list (pagination mode)

Fetch experiment list via standard pagination, returns minimal info. **Does not support filtering.**

| Parameter | Type   | Default | Description                      |
| --------- | ------ | ------- | -------------------------------- |
| `path`    | `str`  | —       | Project path `username/project`  |
| `page`    | `int`  | `1`     | Start page                       |
| `size`    | `int`  | `100`   | Items per page                   |
| `all`     | `bool` | `False` | Auto-paginate to get all results |

:::code-group

```python [Experiment pagination example]
import swanlab

api = swanlab.Api()

for run in api.runs_get(path="my-team/my-project", page=1, size=100, all=True):
    print(run.name, run.state)
```

:::

#### 4. metrics

Fetch scalar metric data (e.g. loss, acc), supports sampling control and range queries, returns structured data.

| Parameter          | Type                   | Default | Description                                                                |
| ------------------ | ---------------------- | ------- | -------------------------------------------------------------------------- |
| `keys`             | `list[str]`            | —       | Metric key list, e.g. `["loss", "acc"]`                                    |
| `sample`           | `int`                  | `1500`  | Sample count (SCALAR max 1500), ignored when `all` or `range_query` is set |
| `all`              | `bool`                 | `False` | Get full data (no sampling limit)                                          |
| `range_query`      | `dict` or `RangeQuery` | `None`  | Range query, only valid for SCALAR type                                    |
| `ignore_timestamp` | `bool`                 | `False` | Whether to remove timestamp fields                                         |

**RangeQuery fields:**

| Field   | Type  | Default  | Description                                                                                                             |
| ------- | ----- | -------- | ----------------------------------------------------------------------------------------------------------------------- |
| `type`  | `str` | `"step"` | Filter axis: `"step"` or `"timestamp"`                                                                                  |
| `start` | `int` | `None`   | Lower bound (inclusive), `None` means from the beginning. When type is `timestamp`, **input must be a UNIX timestamp**  |
| `end`   | `int` | `None`   | Upper bound (inclusive), `None` means up to the last step. When type is `timestamp`, **input must be a UNIX timestamp** |
| `last`  | `int` | `None`   | Last N milliseconds (mutually exclusive with `start`/`end`)                                                             |
| `head`  | `int` | `None`   | Take first N data points (mutually exclusive with `tail`), post-sampled                                                 |
| `tail`  | `int` | `None`   | Take last N data points (mutually exclusive with `head`), post-sampled                                                  |

**Mutual exclusion rules:**

- `last` is mutually exclusive with `start`/`end`
- `head` and `tail` are mutually exclusive, with the lowest priority
- `head`/`tail` can be combined with `start`/`end` or `last` (range filter first, then truncate)

:::code-group

```python [RangeQuery usage example]
from swanlab.api.typings.common import RangeQuery

# By step range
rq = RangeQuery(type="step", start=100, end=500)

# Take last 50 items
rq = RangeQuery(tail=50)

# By timestamp range (milliseconds, auto-padded if fewer than 13 digits)
rq = RangeQuery(type="timestamp", start=1715769600000, end=1715773200000)

# Last 5 minutes
rq = RangeQuery(last=300_000)

# Or use dict directly — metrics with step in [100, 500]
result = run.metrics(keys=["loss"], range_query={"type": "step", "start": 100, "end": 500})
```

```python [Query examples]
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

# Get metric data, returns dict
result = run.metrics(keys=["loss", "acc"])

# Specify sample count (default 1500, max 1500)
result = run.metrics(keys=["loss"], sample=500)

# Full data (no sampling limit)
result = run.metrics(keys=["loss"], all=True)

# Range query by step
result = run.metrics(
    keys=["loss"],
    range_query={"type": "step", "start": 100, "end": 500},
)

# Range query by timestamp (milliseconds, auto-padded if fewer than 13 digits)
result = run.metrics(
    keys=["loss"],
    range_query={"type": "timestamp", "start": 1715769600000, "end": 1715773200000},
)

# Data from the last 5 minutes
result = run.metrics(keys=["loss"], range_query={"last": 300_000})

# Step range + take first 50 points
result = run.metrics(
    keys=["loss"],
    range_query={"start": 0, "end": 500, "head": 50},
)

# Take last 30 data points
result = run.metrics(keys=["loss"], range_query={"tail": 30})
```

:::

#### 5. summary

Get statistical summary of scalar metrics (min / max / avg / median / latest), **each metric uses the `latest` value as the authoritative value**.
| Parameter | Type | Description |
|-----------|------|-------------|
| `keys` | `list[str]` | Scalar keys to query, `None` means all keys |

:::code-group

```python [summary usage example]
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

summary = run.summary(keys=["loss", "acc"])
# Returns stats for each key
print(summary)
```

:::

#### 6. medias

Fetch images, audio, video, echarts and other unstructured media data stored in object storage, response returns only presigned URLs.

| Parameter | Type        | Default | Description                       |
| --------- | ----------- | ------- | --------------------------------- |
| `keys`    | `list[str]` | —       | Media metric key list             |
| `step`    | `int`       | `0`     | Specific step, omit to get latest |
| `all`     | `bool`      | `False` | Get all historical media data     |

:::code-group

```python [medias usage example]
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

:::

:::warning
Returned media data is provided via presigned URLs — download before the expiration time.
:::

#### 7. logs

Fetch text logs from experiment runtime, supports level filtering; can also export as `.log` file.

| Parameter          | Type   | Default  | Description                                 |
| ------------------ | ------ | -------- | ------------------------------------------- |
| `offset`           | `int`  | `0`      | Pagination offset                           |
| `level`            | `str`  | `"INFO"` | Log level: `DEBUG`, `INFO`, `WARN`, `ERROR` |
| `ignore_timestamp` | `bool` | `False`  | Whether to remove timestamp fields          |

| Parameter | Type  | Default  | Description                          |
| --------- | ----- | -------- | ------------------------------------ |
| `start`   | `int` | `0`      | Export start line (0-based)          |
| `rows`    | `int` | `500000` | Number of rows to export, max 500000 |

:::code-group

```python [Log query example]
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

# Get logs
logs = run.logs(offset=0, level="INFO")
print(logs)
```

:::

#### 8. export_logs

Export logs as .log file (returns presigned download link)

:::code-group

```python [Log export example]
import swanlab

api = swanlab.Api()

run = api.run(path="my-team/my-project/abc123")

result = run.export_logs(start=0, rows=500)
if result.ok:
    print(result.data["url"])
```

:::

#### 9. columns

Get metric column names under an experiment, or get a single column by key.

**columns parameters:**

| Parameter      | Type   | Default | Description                                                            |
| -------------- | ------ | ------- | ---------------------------------------------------------------------- |
| `page`         | `int`  | `1`     | Start page                                                             |
| `size`         | `int`  | `100`   | Items per page                                                         |
| `search`       | `str`  | `None`  | Fuzzy search keyword (case-insensitive, matches column **name** field) |
| `column_class` | `str`  | `None`  | Column class: `CUSTOM` or `SYSTEM`                                     |
| `column_type`  | `str`  | `None`  | Column data type: `FLOAT`, `STRING`, `IMAGE`, etc.                     |
| `all`          | `bool` | `False` | Auto-paginate to get all results                                       |

**column parameters:**

| Parameter      | Type  | Default    | Description                                                     |
| -------------- | ----- | ---------- | --------------------------------------------------------------- |
| `path`         | `str` | —          | Experiment path `username/project/run_id`                       |
| `key`          | `str` | —          | Search keyword (fuzzy matches column name, returns first match) |
| `column_class` | `str` | `"CUSTOM"` | Column class: `CUSTOM` or `SYSTEM`                              |
| `column_type`  | `str` | `"FLOAT"`  | Column data type: `FLOAT`, `STRING`, `IMAGE`, etc.              |

```python
import swanlab

api = swanlab.Api()
run = api.run(path="my-team/my-project/abc123")

# Get all metric columns
for col in run.columns(all=True):
    print(col.name)

# Fuzzy search (matches column name)
for col in run.columns(search="loss"):
    print(col.name)

# Get a single column (fuzzy match, returns first)
col = run.column(key="loss", column_type="FLOAT")
```

#### 10. delete

Delete experiment, controls actual deletion behavior via `commit`.

```python
import swanlab

api = swanlab.Api()
run = api.run(path="my-team/my-project/abc123")

run.delete(commit=False)  # commit=False does not actually delete
```

## Column

Represents metric names reported via `swanlab.log()`.

### Column properties

| Property       | Type  | Description                                                      |
| -------------- | ----- | ---------------------------------------------------------------- |
| `name`         | `str` | Column display name                                              |
| `key`          | `str` | Column key                                                       |
| `column_class` | `str` | Column class: `CUSTOM` or `SYSTEM`                               |
| `column_type`  | `str` | Data type: `FLOAT`, `STRING`, `IMAGE`, `VIDEO`, `OBJECT3D`, etc. |
| `created_at`   | `int` | Creation timestamp                                               |

### Column method examples

**column parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Experiment path `username/project/run_id` |
| `key` | `str` | — | Search keyword (fuzzy matches column name, returns first match) |
| `column_class` | `str` | `"CUSTOM"` | Column class: `CUSTOM` or `SYSTEM` |
| `column_type` | `str` | `None` | Column data type: `FLOAT`, `STRING`, `IMAGE`, `VIDEO`, etc. |

**columns parameters:**

| Parameter      | Type   | Default | Description                                                            |
| -------------- | ------ | ------- | ---------------------------------------------------------------------- |
| `path`         | `str`  | —       | Experiment path `username/project/run_id`                              |
| `page`         | `int`  | `1`     | Start page                                                             |
| `size`         | `int`  | `100`   | Items per page                                                         |
| `search`       | `str`  | `None`  | Fuzzy search keyword (case-insensitive, matches column **name** field) |
| `column_class` | `str`  | `None`  | Column class: `CUSTOM` or `SYSTEM`                                     |
| `column_type`  | `str`  | `None`  | Column data type                                                       |
| `all`          | `bool` | `False` | Auto-paginate to get all results                                       |

**Column.metric() parameters:**

| Parameter          | Type   | Default    | Description                                                                      |
| ------------------ | ------ | ---------- | -------------------------------------------------------------------------------- |
| `sample`           | `int`  | `1500`     | Sample count (SCALAR max 1500)                                                   |
| `metric_type`      | `str`  | `"SCALAR"` | Metric type, auto-inferred from column type, usually no need to specify manually |
| `ignore_timestamp` | `bool` | `False`    | Whether to remove timestamp fields                                               |
| `media_step`       | `int`  | `None`     | Only effective for MEDIA type, specifies the step                                |

**Columns.total property:**

| Property | Type  | Description                                                             |
| -------- | ----- | ----------------------------------------------------------------------- |
| `total`  | `int` | Total number of matching columns (triggers one request on first access) |

:::code-group

```python [Get single metric column]
import swanlab

api = swanlab.Api()

col = api.column(
    path="my-team/my-project/abc123",
    key="loss",
    column_type="FLOAT",
)
```

```python [Paginate metric column list]
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

```python [Fuzzy search metric name]
import swanlab

api = swanlab.Api()

# search parameter does case-insensitive contains match on column name, paginated query
for col in api.columns(
    path="my-team/my-project/abc123",
    search="loss",
):
    print(col.name)
```

```python [Full iteration]
import swanlab

api = swanlab.Api()

for col in api.columns(
    path="my-team/my-project/abc123",
    all=True,
    column_type="FLOAT",
):
    print(col.name)
```

```python [Get column total count]
import swanlab

api = swanlab.Api()

cols = api.columns(path="my-team/my-project/abc123")
print(cols.total)  # returns the total number of matching columns
```

```python [Get single column metric data]
import swanlab

api = swanlab.Api()

col = api.column(path="my-team/my-project/abc123", key="loss")
data = col.metric(sample=500)
```

```python [Export column metrics as CSV]
import swanlab

api = swanlab.Api()

col = api.column(path="my-team/my-project/abc123", key="loss")
result = col.export_csv()
if result.ok:
    print(result.data["url"])  # CSV download link
```

:::

## User

### User properties

| Property      | Type  | Description      |
| ------------- | ----- | ---------------- |
| `username`    | `str` | Username         |
| `name`        | `str` | Display name     |
| `bio`         | `str` | Bio              |
| `institution` | `str` | Institution      |
| `school`      | `str` | School           |
| `email`       | `str` | Email            |
| `location`    | `str` | Location         |
| `url`         | `str` | Personal website |

### User method examples

:::code-group

```python [Get user info]
import swanlab

api = swanlab.Api()
user = api.user()  # no parameters, uses the user info from the Api instance

data = user.json()
```

:::

## Self-hosted Management (self_hosted)

> Applies only to self-hosted deployments and requires super admin privileges.

### SelfHosted properties

| Property  | Type   | Description                          |
| --------- | ------ | ------------------------------------ |
| `enabled` | `bool` | Whether self-hosted mode is enabled  |
| `expired` | `bool` | Whether license is expired           |
| `root`    | `bool` | Whether current user is super admin  |
| `plan`    | `str`  | License type: `free` or `commercial` |
| `seats`   | `int`  | License seat count                   |

### SelfHosted method examples

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

for user in sh.get_users(page=1, size=100, all=True):
    print(user)
```

```python [Project management]
import swanlab

api = swanlab.Api()
sh = api.self_hosted()

# Filter by state + creator
for proj in sh.get_projects(
    page=1, size=100, all=True,
    search="image", state="FINISHED", creator="admin",
):
    print(proj)

# Filter by organization workspace
for proj in sh.get_projects(group="my-team", sort="create"):
    print(proj)
```

```python [Workspace management]
import swanlab

api = swanlab.Api()
sh = api.self_hosted()

# Filter by type
for group in sh.get_groups(all=True, type="TEAM", sort="name"):
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

## Type Reference

**ApiResponseType** — Unified response wrapper for OpenAPI. All API calls returning `ApiResponseType` guarantee no exceptions. Actual return data is in the `data` field.

| Property | Type   | Description                            |
| -------- | ------ | -------------------------------------- |
| `ok`     | `bool` | Whether the request succeeded          |
| `errmsg` | `str`  | Error message, empty string on success |
| `data`   | `Any`  | Response data, `None` on failure       |

| Method   | Description                                                         |
| -------- | ------------------------------------------------------------------- |
| `json()` | Serialize to `dict`, automatically calls `.json()` on entity `data` |
