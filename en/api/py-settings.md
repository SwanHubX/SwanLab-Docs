# swanlab.Settings

```python
Settings(
    model_config = ConfigDict(frozen=True),
    metadata_collect: StrictBool = True,
    collect_hardware: StrictBool = True,
    collect_runtime: StrictBool = True,
    requirements_collect: StrictBool = True,
    conda_collect: StrictBool = False,
    hardware_monitor: StrictBool = True,
    disk_io_dir: DirectoryPath = Field(...),
    upload_interval: PositiveInt = 1,
    max_log_length: int = Field(ge=500, le=4096, default=1024),
)
```

| Parameter              | Type          | Description                                                                                                                                                                                   |
|:-----------------------|:--------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `metadata_collect`     | StrictBool    | Whether to enable metadata collection. Default is `True`.                                                                                                                                     |
| `collect_hardware`     | StrictBool    | Whether to collect hardware information of the current system environment. Default is `True`.                                                                                                 |
| `collect_runtime`      | StrictBool    | Whether to collect runtime information. Default is `True`.                                                                                                                                    |
| `security_mask`        | StrictBool    | Whether to automatically mask privacy information, such as api_key, etc. When enabled, any detected privacy information will be replaced with encrypted characters (****). Default is `True`. |
| `requirements_collect` | StrictBool    | Whether to collect Python environment information (`pip list`). Default is `True`.                                                                                                            |
| `conda_collect`        | StrictBool    | Whether to collect Conda environment information. Default is `False`.                                                                                                                         |
| `hardware_monitor`     | StrictBool    | Whether to enable hardware monitoring. If `metadata_collect` is disabled, this setting is ineffective. Default is `True`.                                                                     |
| `disk_io_dir`          | DirectoryPath | The path for disk I/O monitoring. Default is the system root directory (`/` or `C:\`).                                                                                                        |
| `upload_interval`      | PositiveInt   | Log upload interval (in seconds). Default is `1`.                                                                                                                                             |
| `max_log_length`       | int           | Maximum characters per line for terminal log upload (range: 500-4096). Default is `1024`.                                                                                                     |

## Introduction

- The `swanlab.Settings` class is used to manage SwanLab's global feature switches and settings.

- When `import swanlab` is executed, a default global settings object is created. The settings and their default values are detailed in the table above.

- To adjust certain settings, you need to create a new `Settings` instance (e.g., `new_settings`), pass the configuration parameters you wish to modify during instantiation, and then update the global settings by running `swanlab.merge_settings(new_settings)`.

- Note that the `merge_settings()` method is only available before `swanlab.init()` is called. This means that once `swanlab.init()` is invoked during the use of `swanlab`, the global settings can no longer be modified.

## More Usage Examples

### Updating Global Settings

```python
import swanlab
from swanlab import Settings

# Create a new settings object
new_settings = Settings(
    metadata_collect=False,
    hardware_monitor=False,
    upload_interval=5
)

# Update global settings
swanlab.merge_settings(new_settings)

swanlab.init()
...
```

### Recording Conda Environment Information

```python
import swanlab
from swanlab import Settings

# Create a new settings object
new_settings = Settings(
    conda_collect=True  # Disabled by default
)

# Update global settings
swanlab.merge_settings(new_settings)

swanlab.init()
...
```
