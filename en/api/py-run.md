# run

The term "run" refers to the `SwanLabRun` object returned by `swanlab.init()`. This section introduces some of the methods available for the run.  
(Updating gradually...)

## public

The `public` attribute stores some public information about the `SwanLabRun`, including:
- `project_name`: The name of the project
- `version`: The version
- `run_id`: The experiment ID
- `swanlog_dir`: The path to the swanlog directory
- `run_dir`: The path to the run directory
- `cloud`: Cloud-related information
    - `project_name`: The project name (only valid in cloud mode)
    - `project_url`: The URL of the project in the cloud (only valid in cloud mode)
    - `experiment_name`: The experiment name (only valid in cloud mode)
    - `experiment_url`: The URL of the experiment in the cloud (only valid in cloud mode)

To retrieve the `public` information as a dictionary:

```python
import swanlab
run = swanlab.init()
print(run.public.json())
```

For example, if you want to get the experiment's URL, you can do this:

```python
print(run.public.cloud.experiment_url)
```

## get_url

Get the URL of the experiment.

```python
run = swanlab.init()
print(run.get_url())
```

## get_project_url

Get the URL of the project.

```python
run = swanlab.init()
print(run.get_project_url())
```