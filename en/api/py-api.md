# swanlab.Api

::: warning Version Note
This documentation applies to swanlab >= `0.7.8`.
:::

Based on SwanLab's cloud capabilities, it provides access to **OpenAPI** on the SDK side, allowing users to programmatically manipulate cloud **experiment/project/workspace/metric** resources in their local environment.

![](./py-openapi/logo.jpg)

Through the OpenAPI form, users can in their local programming environment:

- Retrieve experiment data, personal information, workspace details, project lists, etc.
- Perform automated experiment management (e.g., querying, organizing, metadata editing, etc.)
- More easily integrate with other tools (e.g., CI/CD, experiment scheduling, etc.)

Leveraging this feature can greatly enhance the flexibility and extensibility of the SDK, facilitating the construction of advanced usage patterns or extension systems.

## Core Terminology
- **Workspace**: A collection of projects, corresponding to a development team (e.g., "SwanLab"). It is divided into Personal Workspace (PERSON) and Organizational Workspace (TEAM).
- **Project**: A collection of experiments, corresponding to a development task (e.g., "Image Classification").
- **Run (Experiment)**: A single training/inference task, containing metrics, configurations, logs, and other data.
- **Experiment ID**: The unique identifier for an experiment, used to precisely locate a single experiment. It can be found in the "Environment" tab of the experiment on the WebUI or in the URL, with the format `https://swanlab.cn/@username/project_name/runs/experiment_id/...`.

## Introduction

> Prerequisite: You must have logged into a SwanLab account within your programming environment.

To use SwanLab's OpenAPI, simply instantiate an `Api` object.

```python
import swanlab

api = swanlab.Api() # Uses local login information (from `swanlab login`)
```

If you need to access data for a specific user:

```python
import swanlab

other_api = swanlab.Api(api_key='other_api_key') # Uses a specific user's api_key
```

Specifically, the authentication logic for **OpenAPI** is as follows:

1. If the `api_key` parameter is explicitly provided, that `api_key` is prioritized for authentication. You can view your API key [here](https://swanlab.cn/space/~/settings).
2. Otherwise, local authentication information is used.

::: warning Using OpenAPI with Private Deployment

For privately deployed SwanLab, you can pass the `host` parameter to `Api`.

```python
import swanlab

api = swanlab.Api(api_key='your-api-key', host='your-host')
```
:::

## workspace

:::code-group

```python [List All Workspaces]
import swanlab

api = swanlab.Api()

workspaces = api.workspaces()

for workspace in workspaces:
    print(workspace.username)
    print(workspace.name)
    print(workspace.profile)
```

```python [Specify One Workspace]
import swanlab

api = swanlab.Api()

workspace = api.workspace(username='username')

print(workspace.username)
print(workspace.name)
print(workspace.profile)
```

```python [Get Projects Within a Workspace]
import swanlab

api = swanlab.Api()

workspace = api.workspace(username='username')
projects = workspace.projects()

for project in projects:
    print(project.name)
    print(project.url)
```

:::

Parameters for `workspaces`:

| Parameter | Type | Description |
| --- | --- | --- |
| `username` | `str` | Workspace username, i.e., the unique ID. Defaults to the currently logged-in user. |

---

Parameters for `workspace`:

| Parameter | Type | Description |
| --- | --- | --- |
| `username` | `str` | Workspace username, i.e., the unique ID. Defaults to the currently logged-in user. |

Attributes and Methods of a `workspace` object:

| Attribute/Method | Type | Description |
| --- | --- | --- |
| `username` | `str` | Workspace username, i.e., the unique ID |
| `name` | `str` | Workspace name |
| `role` | `str` | The role of the current logged-in user in this workspace: `OWNER` or `MEMBER` |
| `profile` | `dict` | Workspace introduction information, includes bio, url, institution, email |
| `workspace_type` | `str` | Workspace type, either Personal or Organizational: `PERSON` or `TEAM` |
| `projects()` | - | Gets project objects under this workspace |
| `json()` | - | Gets all workspace information, returns a `dict` |

## project

:::code-group

```python [List All Projects]
import swanlab

api = swanlab.Api()

projects = api.projects(path='username')

for project in projects:
    print(project.name)
    print(project.created_at)
    print(project.url)
    print(project.visibility)
```

```python [Specify One Project]
import swanlab

api = swanlab.Api()

project = api.project(path='username/project_name')

print(project.name)
print(project.created_at)
print(project.url)
print(project.visibility)
```

```python [Get Runs (Experiments) Within a Project]
import swanlab

api = swanlab.Api()

project = api.project(path='username/project_name')
runs = project.runs()

for run in runs:
    print(run.name)
    print(run.url)
```
:::

Parameters for `projects`:

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | Workspace path (username), format: `username`. Used to filter all projects under a specified workspace. |
| `sort` | `str` | Sorting method, options: `created_at` (creation time), `updated_at` (update time) |
| `search` | `str` | Search keyword, fuzzy matches project names |
| `detail` | `bool` | Whether to return detailed project information (e.g., description, tags). Default is `True`. |

---

Parameters for `project`:

| Parameter | Type | Description |
| --- | --- | --- |
| `path` | `str` | Project path, format: `username/project_name` |

Attributes and Methods of a `project` object:

| Attribute/Method | Type | Description |
| --- | --- | --- |
| `name` | `str` | Project name |
| `path` | `str` | Project path, format: `username/project_name` |
| `description` | `str` | Project description |
| `labels` | `list` | Project tags, format: `[label1, label2, ...]` |
| `created_at` | `str` | Project creation time, format: ISO 8601 UTC, e.g., `2025-12-09T17:57:38.224Z` |
| `updated_at` | `str` | Project update time, format same as `created_at` |
| `url` | `str` | Project URL |
| `visibility` | `str` | Project visibility: `PUBLIC` or `PRIVATE` |
| `count` | `dict` | Project statistics, includes number of runs, collaborators, etc. |
| `runs()` | - | Gets run objects under this project |
| `json()` | - | Gets all project information, returns a `dict` |

## run

:::code-group

```python [List All Runs in a Project]
import swanlab

api = swanlab.Api()

runs = api.runs(path='username/project_name')

for run in runs:
    print(run.name)
    print(run.id)
    print(run.created_at)
    print(run.url)
```

```python [Specify One Run]
import swanlab

api = swanlab.Api()

run = api.run(path='username/project_name/experiment_id')

print(run.name)
print(run.id)
print(run.created_at)
print(run.url)
```

```python [Get Run Config]
import swanlab

api = swanlab.Api()

run = api.run(path='username/project_name/experiment_id')

print(run.profile.config)
```

```python [Get Run Environment Data]
import swanlab

api = swanlab.Api()

run = api.run(path='username/project_name/experiment_id')

# Get the run's Python version, hardware info, etc.
print(run.profile.metadata)
# Get the run's Python environment information
print(run.profile.requirements)
```
:::

Parameters for `runs`:

| Parameter | Type | Description |
| --- | --- | --- |
| `path` | `str` | Project path, format: `username/project_name` |
| `filters` | `dict` | Filter conditions, e.g., `{'state': 'FINISHED', 'config.batch_size': '64'}` |

---

Parameters for `run`:

| Parameter | Type | Description |
| --- | --- | --- |
| `path` | `str` | Run path, format: `username/project_name/experiment_id` |

Attributes of a `run` object:

| Attribute/Method | Type | Description |
| --- | --- | --- |
| `name` | `str` | Run name |
| `path` | `str` | Run path, format: `username/project_name/experiment_id` |
| `description` | `str` | Run description |
| `id` | `str` | Run ID (Experiment ID) |
| `state` | `str` | Run state: `FINISHED`, `RUNNING`, `CRASHED`, `ABORTED` |
| `group` | `list` | Run group, format: `['A', 'B', 'C']` |
| `labels` | `list` | Run tags, format: `[label1, label2, ...]` |
| `created_at` | `str` | Run creation time, format: ISO 8601 UTC, e.g., `2025-12-09T17:57:38.224Z` |
| `finished_at` | `str` | Run finish time, format same as `created_at`; if the run has not finished, this field is `None` |
| `url` | `str` | Run URL |
| `job_type` | `str` | Job type, format: `job_name` |
| `profile` | `dict` | Run configuration information, includes `conda`, `config`, `metadata`, `requirements` attributes |
| `show` | `bool` | Run visibility in the chart comparison view: `True` or `False` |
| `user` | `dict` | Run user, format: `{'is_self': True, 'username': 'username'}` |
| `metrics()` | - | Gets the run's metric data, returns a `pd.DataFrame` |
| `json()` | - | Gets all run information, returns a `dict` |

## runs filters

Using the conditional filter parameter `filters`, you can filter runs that meet specific criteria within a project.

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

The `filters` parameter supports the following conditions:

- `state`: Run state, options: `FINISHED`, `RUNNING`, `CRASHED`, `ABORTED`
- `config.<config_name>`: Configuration name, requires `config.` prefix; filters runs where the config value equals the specified value. Supports nested keys, e.g., `config.data.run_id`

## metrics

Gets the metric data for a run. The return value type is `pd.DataFrame`.

:::code-group
```python [Get Data for One Metric]
import swanlab

api = swanlab.Api()

run = api.run(path='username/project_name/experiment_id')
metrics = run.metrics(keys=['loss'])

print(metrics)
```

```python [Get Data for Multiple Metrics]
import swanlab

api = swanlab.Api()

run = api.run(path='username/project_name/experiment_id')
metrics = run.metrics(keys=['loss', 'acc'])

print(metrics)
```

```python [Specify X-Axis]
import swanlab

api = swanlab.Api()

run = api.run(path='username/project_name/experiment_id')
metrics = run.metrics(keys=['loss'], x_axis='acc')

print(metrics)
```

```python [Specify Sample Count]
import swanlab

api = swanlab.Api()

run = api.run(path='username/project_name/experiment_id')
metrics = run.metrics(keys=['loss'], sample=100)

print(metrics)
```
:::

| Parameter | Type | Default | Description |
|---|---|---|---|
| `keys` | `list[str]` | `None` | List of metric names to retrieve, e.g., `['loss', 'acc']`. If not provided, returns an empty DataFrame. |
| `x_axis` | `str` | `step` | X-axis dimension, options: `step` (step number), or a metric name (e.g., `acc`) |
| `sample` | `int` | `None` | Sample count, limits the number of returned rows. If not provided, returns all data. |

## user

> This operation is limited to super administrators on private deployments.

:::code-group
```python [List All Users]
import swanlab

# Ensure you are logged into a privately deployed SwanLab as a super administrator
api = swanlab.Api()

users = api.users()

for user in users:
    print(user.username)
    print(user.is_self)
    print(user.teams)
```

```python [Create User]
import swanlab

# Ensure you are logged into a privately deployed SwanLab as a super administrator
api = swanlab.Api()

user = api.user()

# Create a user with username 'testuser' and password 'test123456'
user.create(username='testuser', password='test123456')

"""For batch creation
for i in range(3):
    user.create(username=f'testuser{i}', password='test123456')
"""
```

```python [Print/Generate/Delete API Key]
import swanlab

# Ensure you are logged into a privately deployed SwanLab as a super administrator
api = swanlab.Api()

user = api.user()

# The following operations currently only support the currently logged-in account.
# Print API Keys
print(user.api_keys)

# Generate a new API Key
new_api_key = user.generate_api_key()

# Delete an API Key
user.delete_api_key(api_key=new_api_key)
```

:::

Parameters for `user`:

| Parameter | Type | Description |
| --- | --- | --- |
| `username` | `str` | Username, defaults to the currently logged-in user. |

Attributes of a `user` object:

| Attribute | Type | Description |
| --- | --- | --- |
| `username` | `str` | Username |
| `is_self` | `bool` | Whether this user is the currently logged-in user |
| `teams` | `list` | Teams the user belongs to |
| `api_keys` | `list` | List of the user's API keys. Only supports getting API keys for the currently logged-in user. |