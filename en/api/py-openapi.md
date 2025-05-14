# swanlab.OpenApi

Based on SwanLab's cloud capabilities, the SDK provides access to **Open API** functionality, allowing users to programmatically operate and retrieve resources related to experiments, projects, and workspaces in the cloud environment from their local environment.

![](./py-openapi/logo.jpg)

Through Open API, users can:

* Retrieve personal information, workspace details, and project lists
* Automatically manage experiments (e.g., querying, organizing, editing metadata, etc.)
* More easily integrate with other tools (e.g., CI/CD, experiment scheduling, etc.)

Making good use of this feature greatly enhances the flexibility and extensibility of the SDK, making it convenient to build advanced workflows or extended systems.

## Introduction

To use SwanLab's Open API, simply instantiate an `OpenApi` object. Make sure you have previously logged in using `swanlab login` in your local environment, or provide an API key via the `api_key` parameter in code.

```python
from swanlab import OpenApi

my_api = OpenApi() # Uses existing login information
print(my_api.list_workspaces().data)

other_api = OpenApi(api_key='other_api_key') # Uses another account's API key
print(other_api.list_workspaces().data)
```

Specifically, the **OpenApi** authentication logic is as follows:

1. If the `api_key` parameter is explicitly provided, it will be used for authentication.
    - The API key can be found [here](https://swanlab.cn/space/~/settings).
2. Otherwise, the logic follows that of `swanlab.login()`

## OpenAPIs

Each API is implemented as a method of the `OpenApi` class, containing the following fields:

### Model Definitions

When using Open API, some cloud resources, such as experiments and projects, are too complex to be a Python based data structure.

Therefore, these resources are defined as objects in the SDK, supporting IDE auto-completion and type checking for easier manipulation.

For example, to retrieve the start time of an experiment object, you can use:

```python
api_response: ApiResponse = my_api.get_experiment(project="project1", exp_cuid="cuid1")
my_exp: Experiment = api_response.data
created_time: str = my_exp.createdAt
```

Or, to retrieve the name of the workspace to which a project object belongs, you can use:

```python
api_response: ApiResponse = my_api.list_projects()
my_project: Project = api_response.data[0]
workspace_name: str = my_project.group["name"]
```

对于一个模型, 其属性可通过以下三种方式访问:

- `my_exp.createdAt`
- `my_exp["createdAt"]`
- `my_exp.get("createdAt")`

模型可以通过字典风格访问, 但不是真正的字典, 可以通过`my_exp_dict: Dict = my_exp.model_dump()`获取此时模型对应的字典

As a Model, its attributes can be accessed in three ways:

- `my_exp.createdAt`
- `my_exp["createdAt"]`
- `my_exp.get("createdAt")`

> Note: The model can be accessed in a dictionary-like manner, but it is not a true dictionary. You can obtain the corresponding dictionary of the model using `my_exp_dict: Dict = my_exp.model_dump()`.

#### ApiResponse Model

Each Open API method returns a `swanlab.api.openapi.types.ApiResponse` object, which contains the following fields:

| Field | Type | Description |
| --- | --- | --- |
| `code` | `int` | HTTP status code |
| `errmsg` | `str` | Error message, non-empty if the status code is not `2XX` |
| `data` | `Any` | Specific data returned, as mentioned in the API descriptions below |

#### Experiment Model

The experiment object is of type `swanlab.api.openapi.types.Experiment`, containing the following fields:

| Field | Type | Description |
| --- | --- | --- |
| `cuid` | `str` | Unique identifier for the experiment |
| `name` | `str` | Name of the experiment |
| `description` | `str` | Description of the experiment |
| `state` | `str` | Status of the experiment, such as `FINISHED`, `RUNNING` |
| `show` | `bool` | Display status |
| `createdAt` | `str` | Time of experiment creation, formatted as `2024-11-23T12:28:04.286Z` |
| `finishedAt` | `str` | Time of experiment completion, formatted as `2024-11-23T12:28:04.286Z`, None if not finished |
| `user` | `Dict[str, str]` | Creator of the experiment, containing `username` and `name` |
| `profile` | `dict` | Detailed configuration information of the experiment, including user-defined configurations and Python runtime environment, etc. |

#### Project Model

The project object is of type `swanlab.api.openapi.types.Project`, containing the following fields:

| Field | Type | Description |
| --- | --- | --- |
| `cuid` | `str` | Unique identifier for the project |
| `name` | `str` | Name of the project |
| `description` | `str` | Description of the project |
| `visibility` | `str` | Visibility, such as `PUBLIC` or `PRIVATE` |
| `createdAt` | `str` | Time of project creation, formatted as `2024-11-23T12:28:04.286Z` |
| `updatedAt` | `str` | Time of project update, formatted as `2024-11-23T12:28:04.286Z` |
| `group` | `Dict[str, str]` | Workspace information, containing `type`, `username`, and `name` |
| `count` | `Dict[str, int]` | Project statistics, such as the number of experiments, number of collaborators, etc. |

Below is a list of all available APIs.

### Workspaces

#### `list_workspaces`

Retrieve the list of all workspaces (organizations) associated with the current user.

**Returns**

`data` `(List[Dict])`: A list of workspaces the user has joined. Each element is a dictionary containing basic workspace information:

| Field | Type | Description |
| --- | --- | --- |
| `name` | `str` | Name of the workspace |
| `username` | `str` | Unique identifier for the workspace (used in URLs) |
| `role` | `str` | Role of the user in this workspace, such as `OWNER` or `MEMBER` |

**Example**

::: code-group  

```python [Retrieve the list of workspaces]
my_api.list_workspaces().data
"""
[
    {
        "name": "workspace1",
        "username": "kites-test3",
        "role": "OWNER"
    },
    {
        "name": "hello-openapi",
        "username": "kites-test2",
        "role": "MEMBER"
    }
]
"""
```

```python [Retrieve the name of the first workspace]
my_api.list_workspaces().data[0]["name"]
"""
workspace1
"""
```

```python [Retrieve the response code]
my_api.list_workspaces().code
"""
200
"""
```

:::

### Experiments

#### `list_project_exps`

Retrieve the list of experiments in a specified project.

**Method Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `project` | `str` | Project name |
| `username` | `str` | Username of the workspace, defaults to the current user |

**Returns**

`data` `(List[Experiment])`: Returns a list of [Experiment](#experiment-model) objects.

**Example**

::: code-group

```python [Retrieve the list of experiments]
my_api.list_project_exps(project="project1").data
"""
[
    {
        "cuid": "cuid1",
        "name": "experiment1",
        "description": "This is a test experiment",
        "state": "FINISHED",
        "show": true,
        "createdAt": "2024-11-23T12:28:04.286Z",
        "finishedAt": null,
        "user": {
            "username": "kites-test3",
            "name": "Kites Test"
        },
        "profile": {
            "config": {
                "lr": 0.001,
                "epochs": 10
            }
        }
    },
    ...
]
"""
```

```python [Retrieve the name of the first experiment]
my_api.list_project_exps(project="project1").data.items[0].name
"""
"experiment1"
"""
```

:::

#### `get_experiment`

Retrieve the information of an experiment.

**Method Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `project` | `str` | Project name |
| `exp_cuid` | `str` | Unique identifier for the experiment |
| `user` | `str` | Username of the workspace, defaults to the current user |

**Returns**

`data` `(Experiment)`: Returns an [Experiment](#experiment-model) object containing detailed information about the experiment.

**Example**

::: code-group

```python [Retrieve the information of an experiment]
my_api.get_experiment(project="project1", exp_cuid="cuid1").data
"""
{
    "cuid": "cuid1",
    "name": "experiment1",
    "description": "This is a test experiment",
    "state": "FINISHED",
    "show": true,
    "createdAt": "2024-11-23T12:28:04.286Z",
    "finishedAt": null,
    "user": {
        "username": "kites-test3",
        "name": "Kites Test"
    },
    "profile": {
        "conda": "...",
        "requirements": "...",
        ...
    }
}
"""
```

```python [Retrieve the CUID of the experiment]
my_api.get_experiment(project="project1", exp_cuid="cuid1").data.cuid
"""
"cuid1"
"""
```

Retrieve the status of the experiment:

```python [Retrieve the status of the experiment]
my_api.get_experiment(project="project1", exp_cuid="cuid1").data.state
"""
FINISHED
"""
```

```python
my_api.get_experiment(project="project1", exp_cuid="cuid1").data.user["username"]
"""
"kites-test3"
"""
```

:::

<br>

#### `get_exp_summary`

获取一个实验的概要信息, 包含实验跟踪指标的最终值和最大最小值, 以及其对应的步数

**方法参数**

| 参数 | 类型 | 描述 |
| --- | --- | --- |
| `project` | `str` | 项目名 |
| `exp_cuid` | `str` | 实验CUID, 唯一标识符，可通过`list_project_exps`获取，也可以在URL如`https://swanlab.cn/usename/projectname/runs/{exp_cuid}/chart`中获取 |
| `username` | `str` | 工作空间名, 默认为用户个人空间 |

**返回值**

`data` `(Dict[str, Dict])`: 返回一个字典, 包含实验的概要信息

字典中的每个键是一个指标名称, 值是一个结构如下的字典:

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| `step` | `int` | 最后一个步数 |
| `value` | `float` | 最后一个步数的指标值 |
| `min` | `Dict[str, float]` | 最小值对应的步数和指标值 |
| `max` | `Dict[str, float]` | 最大值对应的步数和指标值 |


**示例**

::: code-group

```python [获取实验概要信息]
my_api.get_exp_summary(project="project1", exp_cuid="cuid1").data
"""
{
    "loss": {
        "step": 47,
        "value": 0.1907215012216071,
        "min": {
            "step": 33,
            "value": 0.1745886406861026
        },
        "max": {
            "step": 0,
            "value": 0.7108771095136294
        }
    },
    ...
}
"""
```


```python [获取指标的最大值]
my_api.get_exp_summary(project="project1", exp_cuid="cuid1").data["loss"]["max"]["value"]
"""
0.7108771095136294
"""
```

```python [获取指标最小值所在步]
my_api.get_exp_summary(project="project1", exp_cuid="cuid1").data["loss"]["min"]["step"]
"""
33
"""
```
:::

<br>

### Projects

#### `list_projects`

Retrieve the list of projects in a specified workspace.

**Method Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `username` | `str` | Username of the workspace, defaults to the current user |
| `detail` | `bool` | Whether to include project statistics, defaults to True |

**Returns**

`data` `(List[Project])`: Returns a list of [Project](#project-model) objects.

**Example**

::: code-group

```python [Retrieve the list of projects]
my_api.list_projects().data
"""
[
    {
        "cuid": "project1",
        "name": "Project 1",
        "description": "Description 1",
        "visibility": "PUBLIC",
        "createdAt": "2024-11-23T12:28:04.286Z",
        "updatedAt": null,
        "group": {
            "type": "PERSON",
            "username": "kites-test3",
            "name": "Kites Test"
        },
        "count": {
            "experiments": 4,
            "contributors": 1,
            "children": 0,
            "runningExps": 0
        }
    },
    ...
]
"""
```

:::
