# `swanlab.OpenApi`

Based on SwanLab's cloud capabilities, the SDK provides access to **Open API** functionality, allowing users to programmatically operate and retrieve metadata related to experiments, projects, and workspaces in the cloud environment from their local environment.

Through Open API, users can:

* Retrieve personal information, workspace details, and project lists
* Automatically manage experiments (e.g., querying, organizing, editing metadata, etc.)
* More easily integrate with other tools (e.g., CI/CD, experiment scheduling, etc.)

Making good use of this feature greatly enhances the flexibility and extensibility of the SDK, making it convenient to build advanced workflows or extended systems.

## Introduction

To use SwanLab's Open API, simply instantiate an `OpenApi` object. Make sure you have previously logged in using `swanlab login` in your local environment, or provide an API key via the `key` parameter in code.

```python
from swanlab import OpenApi

my_api = OpenApi() # Uses existing login information
print(my_api.list_workspaces())

other_api = OpenApi(key='other_api_key') # Uses another account's API key
print(other_api.list_workspaces())
```

Specifically, the **OpenApi** authentication logic is as follows:

1. If the `key` parameter is explicitly provided, it will be used for authentication.
2. Otherwise, the logic follows that of `swanlab.login()`

## OpenAPIs

Each API is implemented as a method of the `OpenApi` class. Below is a list of all available APIs.

### List Workspaces - `list_workspaces`

Retrieve the list of all workspaces (organizations) associated with the current user.

**Returns**

Each item in the returned list is a dictionary containing basic workspace information:

- `name`: `str`, Name of the workspace
- `username`: `str`, Unique identifier for the workspace (used in URLs)
- `role`: `str`, Role of the user in this workspace, such as `'OWNER'` or `'MEMBER'`

**Example**

```python
print(my_api.list_workspaces())

[
    {
        'name': 'workspace1',
        'username': 'kites-test3',
        'role': 'OWNER'
    },
    {
        'name': 'hello-openapi',
        'username': 'kites-test2',
        'role': 'MEMBER'
    }
]
```
