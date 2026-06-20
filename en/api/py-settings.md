# swanlab.Settings

```python
Settings(
    model_config = SettingsConfigDict(frozen=True),
    interactive: bool = True,
    mode: Literal["disabled", "local", "online", "offline"] = "online",
    log_dir: DirectoryPath = Field(default_factory=log_dir_factory),
    api_key: Optional[str] = Field(default=None),
    api_host: str = Field(default="https://api.swanlab.cn"),
    web_host: str = Field(default="https://swanlab.cn"),
    project: ProjectSettings = Field(default_factory=ProjectSettings),
    experiment: ExperimentSettings = Field(default_factory=ExperimentSettings),
    run: RunSettings = Field(default_factory=RunSettings),
    terminal: TerminalSettings = Field(default_factory=TerminalSettings),
    integration: IntegrationSettings = Field(default_factory=IntegrationSettings),
    core: CoreSettings = Field(default_factory=CoreSettings),
    probe: ProbeSettings = Field(default_factory=ProbeSettings),
)
```

| Parameter     | Type          | Description                                                                                                                                                                                                     |
| :------------ | :------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `interactive` | StrictBool    | Whether to enable interactive mode. When disabled, all user input prompts and related interactions are disabled. Suitable for CI/CD environments or background batch jobs. Default is `True`.                   |
| `mode`        | Literal       | SwanLab run mode. Available values: `disabled` (disable SwanLab), `local` (run locally), `online` (sync to cloud, `cloud` is an alias), `offline` (run offline). **Note: Case-sensitive**. Default is `online`. |
| `log_dir`     | DirectoryPath | Directory for SwanLab log files. Default is the `swanlog` folder in the current working directory.                                                                                                              |
| `api_key`     | str           | API key for SwanLab services.                                                                                                                                                                                   |
| `api_host`    | str           | Base URL for SwanLab API services. Default is `https://api.swanlab.cn`.                                                                                                                                         |
| `web_host`    | str           | Base URL for SwanLab web services. This is only a display URL and does not affect SDK behavior. Default is `https://swanlab.cn`.                                                                                |

## Nested Configuration

### Project Configuration `ProjectSettings`

Passed via the `project` field, e.g. `Settings(project=ProjectSettings(workspace="my-ws"))`.

| Parameter   | Type       | Description                                                                                                  |
| :---------- | :--------- | :----------------------------------------------------------------------------------------------------------- |
| `name`      | str        | Project name.                                                                                                |
| `workspace` | str        | The workspace name this project belongs to.                                                                  |
| `public`    | StrictBool | Whether to make experiments public. Accepts `true`, `yes`, `1`; leave empty for private. Default is `False`. |

### Experiment Configuration `ExperimentSettings`

Passed via the `experiment` field, e.g. `Settings(experiment=ExperimentSettings(tags=["tag1", "tag2"]))`.

| Parameter     | Type      | Description                                                                                                             |
| :------------ | :-------- | :---------------------------------------------------------------------------------------------------------------------- |
| `name`        | str       | Experiment name.                                                                                                        |
| `color`       | str       | Experiment display color, supports preset color names, RGB strings, or hex color codes.                                 |
| `description` | str       | Experiment description.                                                                                                 |
| `tags`        | List[str] | List of experiment tags. Up to 50 tags. Also accepts comma-separated string (e.g., `"tag1,tag2"`) or JSON array string. |
| `group`       | str       | Experiment group name, used to categorize experiments for better management and differentiation.                        |
| `job_type`    | str       | Experiment task type, used to identify the current experiment's task type (e.g., classification, regression, etc.).     |

### Run Configuration `RunSettings`

Passed via the `run` field, e.g. `Settings(run=RunSettings(resume="allow"))`.

| Parameter            | Type    | Description                                                                                                                                                                                                                                                 |
| :------------------- | :------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`                 | str     | Experiment run ID. Leave empty for auto-generation.                                                                                                                                                                                                         |
| `resume`             | Literal | Resume training strategy. Available values: `must` (must resume, error if no history found), `allow` (allow resume), `never` (do not allow resume). Boolean values are also accepted (`True` maps to `allow`, `False` maps to `never`). Default is `never`. |
| `parallel`           | Literal | Parallel execution strategy. Available values: `none` (default), `shared` (shared run directory). When set to `shared`, `resume` is automatically forced to `allow`.                                                                                        |
| `history`            | Path    | History record path, used to resume from a previous run.                                                                                                                                                                                                    |
| `config`             | Path    | Configuration file path or dictionary, used to load experiment configuration.                                                                                                                                                                               |
| `dir`                | str     | Custom run directory name. When set, directory conflict retries are skipped and the directory is created directly.                                                                                                                                          |
| `dir_max_length`     | int     | Maximum length for the auto-generated run directory name. Range: 50-255, default is `255`.                                                                                                                                                                  |
| `dir_create_retries` | int     | Maximum number of retries for creating a unique run directory. When the generated directory name conflicts with an existing one, a new name will be generated after a short delay, up to this many times. Minimum value is 1, default is `10`.              |

### Terminal Log Configuration `TerminalSettings`

Passed via the `terminal` field, e.g. `Settings(terminal=TerminalSettings(proxy_type="stdout"))`.

| Parameter    | Type    | Description                                                                                                                                                                                    |
| :----------- | :------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `proxy_type` | Literal | Terminal log proxy strategy. Available values: `all` (proxy both stdout and stderr), `stdout` (proxy stdout only), `stderr` (proxy stderr only), `none` (do not proxy logs). Default is `all`. |
| `max_length` | int     | Maximum character length per line for terminal log collection. Range: 500-4096, default is `1024`.                                                                                             |

### Integration Configuration `IntegrationSettings`

Passed via the `integration` field, used to configure Webhook and local dashboard.

#### Webhook `WebhookSettings`

| Parameter | Type | Description                                         |
| :-------- | :--- | :-------------------------------------------------- |
| `url`     | str  | Webhook notification address.                       |
| `value`   | str  | The value passed to the Webhook callback structure. |
| `timeout` | int  | Webhook request timeout in seconds. Default is `5`. |

#### Local Dashboard `DashBoardSettings`

| Parameter | Type | Description                                                                  |
| :-------- | :--- | :--------------------------------------------------------------------------- |
| `host`    | str  | Host address for the local SwanLab dashboard server. Default is `127.0.0.1`. |
| `port`    | int  | Port number for the local SwanLab dashboard server. Default is `5092`.       |

### Core Behavior Configuration `CoreSettings`

Passed via the `core` field, used to control core upload and save behavior.

| Parameter         | Type  | Description                                                                                                                                                                                          |
| :---------------- | :---- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `section_rule`    | int   | Metric key segmentation rule, specifies which `/` to use to split section and metric name: `0` = first, `1` = second, `-1` = last. Out-of-range values wrap via `idx % slash_count`. Default is `0`. |
| `record_batch`    | int   | Maximum number of records per HTTP request. Range: 1-100000, default is `10000`.                                                                                                                     |
| `record_interval` | float | Batch upload interval for the upload thread (in seconds). Default is `5.0`. Can be reduced (e.g., to `0.5`) in high-throughput scenarios like Converters.                                            |
| `save_split`      | int   | File size threshold for multipart upload (in bytes). Default is `100 MiB` (104857600 bytes).                                                                                                         |
| `save_size`       | int   | Maximum save size per file (in bytes). Default is `50 GiB` (53687091200 bytes).                                                                                                                      |
| `save_part`       | int   | Multipart upload part size (in bytes). Default is `32 MiB` (33554432 bytes).                                                                                                                         |
| `save_batch`      | int   | Maximum number of files per save upload batch. Default is `100`.                                                                                                                                     |

### Probe Configuration `ProbeSettings`

Passed via the `probe` field, controls which environment information SwanLab collects at startup.

| Parameter          | Type          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| :----------------- | :------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hardware`         | StrictBool    | Whether to collect static hardware metadata (e.g., GPU model, CPU cores, total memory) and report it at startup. **Note**: If `monitor` is `True` while `hardware` is `False`, the monitoring module will still access hardware information to compute dynamic metrics (e.g., utilization percentages), but the static hardware snapshot itself will be discarded — it will neither be included in the telemetry payload nor saved to local persistent storage. Default is `True`. |
| `runtime`          | StrictBool    | Whether to collect software runtime environment information. When enabled, records details such as the operating system, Python version, hostname, current working directory, and the exact command used to launch the script. Default is `True`.                                                                                                                                                                                                                                  |
| `requirements`     | StrictBool    | Whether to collect Python dependency snapshots. When enabled, records installed pip packages and their exact versions (similar to `pip freeze`) to ensure environment reproducibility. Default is `True`.                                                                                                                                                                                                                                                                          |
| `conda`            | StrictBool    | Whether to collect Conda environment configuration. When enabled, records the active Conda environment details and exported dependencies. Default is `False` to avoid extra startup overhead.                                                                                                                                                                                                                                                                                      |
| `git`              | StrictBool    | Whether to collect Git repository metadata. When enabled, captures the current branch, latest commit hash, and remote URL, helping tightly link the experiment run to a specific version of your codebase. Default is `True`.                                                                                                                                                                                                                                                      |
| `swanlab`          | StrictBool    | Whether to collect SwanLab metadata. When enabled, records the SwanLab version, run directory, etc. Default is `True`.                                                                                                                                                                                                                                                                                                                                                             |
| `monitor`          | StrictBool    | Whether to enable periodic hardware monitoring (CPU usage, GPU utilization, memory, etc.). Default is `True`.                                                                                                                                                                                                                                                                                                                                                                      |
| `monitor_interval` | int           | Periodic hardware monitoring collection interval (in seconds). Minimum value is 5, default is `10`.                                                                                                                                                                                                                                                                                                                                                                                |
| `monitor_disk_dir` | DirectoryPath | Disk I/O monitoring reference directory, used to calculate disk usage. Defaults to the system root directory (`/` on Linux/macOS, `C:\` on Windows).                                                                                                                                                                                                                                                                                                                               |

## Introduction

- The `swanlab.Settings` class manages SwanLab's global feature switches and settings.

- When `import swanlab` is executed, a default global settings object is created. The settings and their default values are detailed in the table above.

- To adjust certain settings, you need to create a new `Settings` instance (e.g., `new_settings`), pass the configuration parameters you wish to modify during instantiation, and then update the global settings by running `swanlab.merge_settings(new_settings)`.

- Note that `merge_settings()` is only available before `swanlab.init()` is called. This means that once `swanlab.init()` is invoked, the global settings can no longer be modified.

## More Usage Examples

### Updating Global Settings

::: code-group

```python [Method 1]
import swanlab

# Create a new settings object
new_settings = swanlab.Settings(
    interactive=False,
    mode="offline",
    probe=swanlab.Settings.Probe(
        hardware=False,
        monitor=False,
        monitor_interval=5,
    ),
    core=swanlab.Settings.Core(
        record_batch=5000,
        record_interval=2.0,
    ),
)

swanlab.init(settings=new_settings)
...
```

```python [Method 2]
import swanlab

# Create a new settings object
new_settings = swanlab.Settings(
    interactive=False,
    mode="offline",
    probe=swanlab.Settings.Probe(
        hardware=False,
        monitor=False,
        monitor_interval=5,
    ),
    core=swanlab.Settings.Core(
        record_batch=5000,
        record_interval=2.0,
    ),
)

# Update global settings
swanlab.merge_settings(new_settings)

swanlab.init()
...
```

:::

### Disabling Interactive Mode

Suitable for CI/CD environments or background batch jobs:

```python
import swanlab
from swanlab import Settings

# Create a new settings object
new_settings = Settings(
    interactive=False,
)

# Update global settings
swanlab.merge_settings(new_settings)

swanlab.init()
...
```

### Configuring Probes

```python
import swanlab
from swanlab import Settings

# Create a new settings object
new_settings = Settings(
    probe=Settings.Probe(
        hardware=False,     # Do not collect static hardware information
        conda=True,         # Collect Conda environment configuration
        git=False,          # Do not collect Git information
        monitor=True,       # Enable hardware monitoring
        monitor_interval=5, # Monitoring interval: 5 seconds
        monitor_disk_dir="/data",  # Disk monitoring directory
    ),
)

# Update global settings
swanlab.merge_settings(new_settings)

swanlab.init()
...
```
