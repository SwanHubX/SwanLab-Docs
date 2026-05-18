# Experiment Metadata

> Requires swanlab>=0.3.25 to obtain experiment metadata.

There are times when you want to retrieve experiment metadata in your code, such as the project name, ID, experiment name, URL, etc.

Here's how to do it:

```python
import swanlab

run = swanlab.init(
    project="test-project",
    experiment="test-exp",
)

# Print all metadata
print(run.public.json())

# Print individual metadata
print(run.public.project_name)
print(run.public.cloud.experiment_url)
```

The `run` class returned by `swanlab.init` carries a `public` attribute, which replaces the previous `settings` attribute. It returns:

- `project_name`: The name of the current running project
- `version`: The current version of swanlab being used
- `run_id`: A unique ID
- `swanlog_dir`: The directory where swanlab saves files
- `run_dir`: The directory where the current experiment is saved
- `cloud`: Information related to the cloud environment
    - `available`: Whether it is running in cloud mode. If not, all the following attributes will be None.
    - `project_name`: The name of the current running project
    - `project_url`: The URL of the current running project in the cloud
    - `experiment_name`: The name of the current running experiment
    - `experiment_url`: The URL of the current running experiment in the cloud