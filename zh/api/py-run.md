# run

run 指的是 `swanlab.init()` 返回的 `SwanLabRun` 对象，这里介绍run具有的一些方法。  
(逐步更新中...)

## public

public存储了SwanLabRun的一些公共信息，包括：
- `project_name`: 项目名称
- `version`: 版本
- `run_id`: 实验ID
- `swanlog_dir`: swanlog日志目录的路径
- `run_dir`: 运行目录的路径
- `cloud`: 云端信息
    - `project_name`: 项目名称（仅在cloud模式时有效）
    - `project_url`: 项目在云端的URL（仅在cloud模式时有效）
    - `experiment_name`: 实验名称（仅在cloud模式时有效）
    - `experiment_url`: 实验在云端的URL（仅在cloud模式时有效）

以字典形式获取public信息：

```python
import swanlab
run = swanlab.init()
print(run.public.json())
```

比如，你想要获取实验的URL，可以这样：

```python
print(run.public.cloud.experiment_url)
```

## get_url

获取实验的URL。

```python
run = swanlab.init()
print(run.get_url())
```

## get_project_url

获取项目的URL。

```python
run = swanlab.init()
print(run.get_project_url())