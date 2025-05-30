# Environment Variables

[⚙️Complete Environment Variables 1 -> Github](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/env.py)、[⚙️Complete Environment Variables 2 -> Github](https://github.com/SwanHubX/SwanLab-Toolkit/blob/main/swankit/env.py)

## Global Configuration

| Environment Variable | Description | Default Value |
| --- | --- | --- |
| `SWANLAB_SAVE_DIR` | Path where SwanLab global folder is saved | `.swanlab` folder in the user's home directory |
| `SWANLAB_LOG_DIR` | Path where SwanLab parsed log files are saved | `swanlog` folder in the current working directory |
| `SWANLAB_MODE` | SwanLab's parsing mode, which involves callbacks registered by the operator. Currently, there are three modes: `local`, `cloud`, and `disabled`. **Note: Case-sensitive** | `cloud` |

## Service Configuration

| Environment Variable | Description | 
| --- | --- |
| `SWANLAB_BOARD_PORT` | Port for the CLI offline dashboard `swanboard` service |
| `SWANLAB_BOARD_HOST` | Address for the CLI offline dashboard `swanboard` service |
| `SWANLAB_WEB_HOST` | Web address for the SwanLab cloud environment |
| `SWANLAB_API_HOST` | API address for the SwanLab cloud environment |

## Experiment Configuration

| Environment Variable | Description |
| --- | --- |
| `SWANLAB_PROJ_NAME` | Project name, equivalent to `swanlab.init(project_name="...")` |
| `SWANLAB_WORKSPACE` | Workspace name, equivalent to `swanlab.init(workspace="...")` |
| `SWANLAB_EXP_NAME` | Experiment name, equivalent to `swanlab.init(experiment_name="...")` |

## Login Authentication

| Environment Variable | Description |
| --- | --- | 
| `SWANLAB_API_KEY` | Cloud API Key. During login, this environment variable is checked first. If it doesn't exist, the system checks if the user is already logged in. If not, the login process is initiated.<br>- If a string is passed to the `login` interface, this environment variable is ignored.<br>- If the user is already logged in, this environment variable takes precedence over locally stored login information. |

## Others

| Environment Variable | Description |
| --- | --- |
| `SWANLAB_WEBHOOK` | Webhook address.<br> When SwanLab initialization is complete, if this environment variable exists, it will be called to send a message. |