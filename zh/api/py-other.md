# 其他Python API

## get_run

获取当前运行的实验对象（`SwanLabRun`）。

```python
run = swanlab.init(...)

...

run = swanlab.get_run()
```

## get_url

获取实验的URL（cloud模式，否则为None）。

```python
print(swanlab.get_url())
```

## get_project_url

获取项目的URL（cloud模式，否则为None）。

```python
print(swanlab.get_project_url())
```

