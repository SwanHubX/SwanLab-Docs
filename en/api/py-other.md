# Other Python APIs

## get_run

Retrieves the currently running experiment object (`SwanLabRun`).

```python
run = swanlab.init(...)

...

run = swanlab.get_run()
```

## get_url

Retrieves the URL of the experiment (in cloud mode; otherwise, returns `None`).

```python
print(swanlab.get_url())
```

## get_project_url

Retrieves the URL of the project (in cloud mode; otherwise, returns `None`).

```python
print(swanlab.get_project_url())
```