# 实验元数据

> 获取实验元数据需swanlab>=0.3.25

总有些时候，你想要在代码中获取实验的元数据，比如实验的项目名、ID、实验名、网址等。

获取方式：

```python
import swanlab

run = swanlab.init(
    project="test-project",
    experiment="test-exp",
)

# 打印出所有元数据
print(run.public.json())

# 打印出单个元数据
print(run.public.project_name)
print(run.public.cloud.experiment_url)
```

`swanlab.init`返回的类`run`会携带`public`属性，替换了之前的`settings`属性，他会返回：

- `project_name`：当前运行的项目名称
- `version`：当前运行的swanlab版本
- `run_id`：一个唯一id
- `swanlog_dir`：swanlab保存文件夹
- `run_dir`：本次实验的保存文件夹
- `cloud`：云端环境的相关信息
    - `available`：是否运行在云端模式，如果不是，下面的属性全部为None
    - `project_name`：本次运行的项目名称
    - `project_url`：本次运行在云端项目url
    - `experiment_name`：本次运行的实验名称
    - `experiment_url`：本次运行的云端实验url




