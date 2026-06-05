# Environment Variables

[⚙️Complete Environment Variables  -> Github](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/sdk/internal/settings/__init__.py)

## Debug Configuration

| Environment Variable | Description | Default Value |
| -------------------- | ---------------------------------------------------------------------------- | ------------- |
| `SWANLAB_DEBUG` | Whether to enable debug mode. When enabled, debug messages are printed in the terminal and mirrored to diagnostic log files. | `false` |

## Global Configuration

| Environment Variable | Description | Default Value |
| -------------------- | -------------------------------------------------------------------- | ------------------------------------------------- |
| `SWANLAB_ROOT` | Path where SwanLab global folder is saved | `.swanlab` folder in the user's home directory |
| `SWANLAB_PUBLIC` | Whether to make experiments public. Accepts `true`, `yes`, `1`; leave empty for private | Private |
| `SWANLAB_LOGDIR` | Path where SwanLab parsed log files are saved | `swanlog` folder in the current working directory |
| `SWANLAB_MODE` | SwanLab's parsing mode. Available modes: `local`, `online` (`cloud` is an alias), `offline`, `disabled`. **Case-sensitive** | `online` |
| `SWANLAB_API_HOST`   | API address for the SwanLab cloud environment | `https://api.swanlab.cn` |
| `SWANLAB_WEB_HOST`   | Web address for the SwanLab cloud environment. For private deployment, only this variable needs to be set, no need to set `SWANLAB_API_HOST` | `https://swanlab.cn` |

## Experiment Configuration

| Environment Variable   | Description                                                                                                                              |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `SWANLAB_PROJ_NAME`    | Project name, equivalent to `swanlab.init(project="...")`                                                                                |
| `SWANLAB_WORKSPACE`    | Workspace name, equivalent to `swanlab.init(workspace="...")`                                                                            |
| `SWANLAB_EXP_NAME`     | Experiment name, equivalent to `swanlab.init(experiment_name="...")`                                                                     |
| `SWANLAB_RUN_ID`       | Experiment run ID, equivalent to `swanlab.init(id="...")`                                                                                |
| `SWANLAB_RESUME`       | Whether to resume training, equivalent to `swanlab.init(resume=...)`, possible values: `must`, `allow`, `never`                          |
| `SWANLAB_DESCRIPTION`  | Experiment description, equivalent to `swanlab.init(description="...")`                                                                  |
| `SWANLAB_TAGS`         | Experiment tags, equivalent to `swanlab.init(tags=[...])`. If you want to add multiple tags, write it as `SWANLAB_TAGS="tag1,tag2,tag3"` |
| `SWANLAB_GROUP`        | Experiment group, used to categorize experiments into different groups for better management and differentiation                         |
| `SWANLAB_JOB_TYPE`     | Experiment task type, used to identify the current experiment's task type (e.g., classification, regression, etc.)                       |
| `SWANLAB_EXP_COLOR`    | Experiment color, equivalent to `swanlab.init(color="...")`. Supports preset color names, RGB strings, or hex color codes.               |
| `SWANLAB_RUN_PARALLEL` | Parallel mode, equivalent to `swanlab.init(parallel="...")`. Possible value: `shared`.                                                   |
| `SWANLAB_RUN_DIR`      | Custom run directory name. When set, directory conflict retries are skipped and the directory is created directly | Auto-generated |
| `SWANLAB_RUN_DIR_MAX_LENGTH` | Maximum length for the auto-generated run directory name | `255` |

## Login Authentication

| Environment Variable | Description                                                                                                                                                                                                                                                                                                                                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SWANLAB_API_KEY`    | Cloud API Key. During login, this environment variable is checked first. If it doesn't exist, the system checks if the user is already logged in. If not, the login process is initiated.<br>- If a string is passed to the `login` interface, this environment variable is ignored.<br>- If the user is already logged in, this environment variable takes precedence over locally stored login information. |

## Behavior Control

| Environment Variable  | Description                                                                                                         |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `SWANLAB_DISABLE_GIT` | Whether to disable Git, possible values: `True`, `False`. When set to `True`, Git information will not be recorded. |

## Others

| Environment Variable    | Description                                                                                                                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SWANLAB_WEBHOOK`       | Webhook address.<br> When SwanLab initialization is complete, if this environment variable exists, it will be called to send a message.                                                      |
| `SWANLAB_WEBHOOK_VALUE` | The value passed to the Webhook callback structure. <br> When `SWANLAB_WEBHOOK` exists, if this environment variable exists, it will be sent as the value of the Webhook callback structure. |
| `SWANLAB_WEBHOOK_TIMEOUT` | Webhook request timeout (seconds) | `5` |

## Probe Configuration

Control the types of environment information SwanLab collects at startup via the `probe` parameter (e.g., `swanlab.init(probe={"hardware": False})`) or the environment variables below. All probe options accept `true`/`false`, `1`/`0`, `yes`/`no`.

| Environment Variable | Description | Default Value |
| -------------------- | ---------------------------------------------------------------------------- | ------------- |
| `SWANLAB_PROBE_HARDWARE` | Whether to collect static hardware information (GPU model, CPU cores, total memory, etc.) | `true` |
| `SWANLAB_PROBE_RUNTIME` | Whether to collect software runtime information (OS, Python version, launch command, etc.) | `true` |
| `SWANLAB_PROBE_REQUIREMENTS` | Whether to collect Python dependency list (similar to `pip freeze`) | `true` |
| `SWANLAB_PROBE_CONDA` | Whether to collect Conda environment configuration, disabled by default to avoid extra startup overhead | `false` |
| `SWANLAB_PROBE_GIT` | Whether to collect Git repository information (current branch, latest commit, remote URL) | `true` |
| `SWANLAB_PROBE_SWANLAB` | Whether to collect SwanLab metadata (version, run directory, etc.) | `true` |
| `SWANLAB_PROBE_MONITOR` | Whether to enable periodic hardware monitoring (CPU usage, GPU utilization, memory, etc.) | `true` |
| `SWANLAB_PROBE_MONITOR_INTERVAL` | Periodic hardware monitoring collection interval (seconds) | `10` |
| `SWANLAB_PROBE_MONITOR_DISK_DIR` | Disk I/O monitoring reference directory, used to calculate disk usage | System root directory |

## Local Dashboard Configuration

| Environment Variable | Description | Default Value |
| -------------------- | ------------------------------------------------------------ | ------------- |
| `SWANLAB_DASHBOARD_HOST` | Host address for the local SwanLab dashboard server | `127.0.0.1` |
| `SWANLAB_DASHBOARD_PORT` | Port number for the local SwanLab dashboard server | `5092` |

## Logging Configuration

| Environment Variable | Description | Default Value |
| -------------------- | ---------------------------------------------------------------------------- | ------------- |
| `SWANLAB_LOG_LEVEL` | SwanLab log output level, controls log verbosity. Possible values: `debug` (most verbose), `info` (default), `warning`, `error`, `critical` (least verbose) | `info` |

## Core Behavior

| Environment Variable | Description | Default Value |
| -------------------- | ---------------------------------------------------------------------------- | ------------- |
| `SWANLAB_SKIP_SWANBOARD_VERSION_CHECK` | Whether to skip the swanboard version compatibility check. Set to `1` to skip | Not skipped |
| `SWANLAB_FS_TIMEOUT` | File system operation timeout (seconds), useful for NAS environments with high async latency | `5.0` |
| `SWANLAB_CORE_SECTION_RULE` | Metric key segmentation rule, specifies which `/` to use to split section and metric name: `0`=first, `1`=second, `-1`=last | `0` |
| `SWANLAB_CORE_RECORD_BATCH` | Maximum number of records per HTTP request | `10000` |
| `SWANLAB_TERMINAL_PROXY_TYPE` | Terminal log proxy strategy: `all` (all), `stdout`, `stderr`, `none` (no collection) | `all` |
| `SWANLAB_TERMINAL_MAX_LENGTH` | Maximum character length per line for terminal log collection | `1024` |

## Advanced Configuration

| Environment Variable | Description | Default Value |
| -------------------- | ---------------------------------------------------------------------------- | ------------- |
| `SWANLAB_SECRETS_DIR` | K8s / Docker container Secret configuration file directory, used to inject sensitive information (such as API Key) | None |
| `SWANLAB_CONFIG_DIR` | Global configuration file directory path, SwanLab reads `*.yaml` / `*.yml` config files from this directory | `/etc/swanlab` |
