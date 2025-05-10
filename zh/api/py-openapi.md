# swanlab.OpenApi

基于 SwanLab 云端功能，在 SDK 端提供访问 **开放 API（OpenAPI）** 的能力，允许用户通过编程方式在本地环境中操作、获取与云端实验/项目/工作空间相关的元数据。

通过开放 API 的形式，用户可以在本地编程环境中：

- 获取个人信息、工作空间信息、项目列表等
- 进行实验的自动管理（如查询、组织、元数据编辑等）
- 更方便地与其他工具集成（如 CI/CD、实验调度等）

利用好此特性可极大提升 SDK 的灵活性和可扩展性，方便构建高级用法或扩展体系。

## 介绍

要使用 SwanLab 的开放 API，只需实例化一个 `OpenApi` 对象。需要确保之前在本地使用`swanlab login`登录过， 或者在代码中使用`key`参数传入 API 密钥。

```python
from swanlab import OpenApi

my_api = OpenApi() # 使用之前的登录信息
print(my_api.list_workspaces())

other_api = OpenApi(key='other_api_key') # 使用另一个账户的key
print(other_api.list_workspaces())
```

具体来说，**OpenApi**的认证逻辑如下：

1. 如果显式提供了`key`参数，则优先使用该`key`进行身份认证；
2. 否则，遵循与`swanlab.login()`相同的认证逻辑

## OpenAPIs

每个 API 都是`OpenApi`类的一个方法，下面是所有可用的 API 列表。

### 列出工作空间 `list_workspaces`

获取当前用户的所有工作空间（组织）列表。

**返回值**

每个元素是一个字典, 包含工作空间的基础信息:

- `name`: `str`, 工作空间名称
- `username`: `str`, 工作空间唯一标识(用于组织相关的 URL)
- `role`: `str`, 用户在该工作空间中的角色，如 'OWNER' 或 'MEMBER'

**示例**

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

### 查询一个实验状态 `get_exp_status`

获取实验状态

**参数**

- `project`: `str` (项目名)
- `exp_cuid`: `str` (实验id)
- `username`: `Optional[str]` (工作空间名, 默认为用户个人空间)

**返回值**

返回一个字典, 包含以下字段:

- `state`: `str`, 实验状态, 为 'FINISHED' 或 'RUNNING'
- `finishedAt` `str`, 实验完成时间（若有）, 格式如 '2024-11-23T12:28:04.286Z'

若请求失败, 将返回包含以下字段的字典:

- `code` `int`, HTTP 错误代码
- `message` `str`, 错误信息

**示例**

```python
print(my_api.api.get_exp_state(project="test_project", exp_cuid="test_exp_cuid"))

{
	"state": "FINISHED",
	"finishedAt": "2024-04-23T12:28:04.286Z",
}
请求失败时:
{
	"state": "RUNNING"
}
```