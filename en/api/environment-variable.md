# Environment Variables

[⚙️Full Environment Variables -> Github](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/env.py)

## Global Configuration

### `SWANLAB_FOLDER`
- **Description**: The path where the SwanLab global folder is saved, defaulting to the `.swanlab` folder in the user's home directory.
- **Environment Variable**: `SWANLAB_FOLDER`

### `SWANLOG_FOLDER`
- **Description**: The path where SwanLab parsed log files are saved, defaulting to the `swanlog` folder in the current running directory.
- **Environment Variable**: `SWANLOG_FOLDER`

### `SWANLAB_MODE`
- **Description**: The parsing mode of SwanLab, involving callbacks registered by the operator. Currently, there are three modes: `local`, `cloud`, and `disabled`, with the default being `cloud`. **Note: Case sensitive**.
- **Environment Variable**: `SWANLAB_MODE`

## Service Configuration

### `SWANBOARD_PROT`
- **Description**: The port for the CLI offline dashboard `swanboard` service.
- **Environment Variable**: `SWANLAB_BOARD_PORT`

### `SWANBOARD_HOST`
- **Description**: The address for the CLI offline dashboard `swanboard` service.
- **Environment Variable**: `SWANLAB_BOARD_HOST`

### `SWANLAB_WEB_HOST`
- **Description**: The web address for the SwanLab cloud environment.
- **Environment Variable**: `SWANLAB_WEB_HOST`

### `API_HOST`
- **Description**: The API address for the SwanLab cloud environment.
- **Environment Variable**: `SWANLAB_API_HOST`

## Login Authentication

### `SWANLAB_API_KEY`
- **Description**: The cloud API Key. During login, this environment variable is checked first. If it does not exist, it checks whether the user is already logged in. If not logged in, the login process is initiated.
  - If a string is passed to the `login` interface, this environment variable is invalid.
  - If the user is already logged in, this environment variable takes precedence over locally stored login information.
- **Environment Variable**: `SWANLAB_API_KEY`

## Others

### `SWANLAB_WEBHOOK`
- **Description**: Webhook address. If this environment variable exists, SwanLab will call this address to send a message upon successful initialization.
- **Environment Variable**: `SWANLAB_WEBHOOK`