# 环境变量

[⚙️完整环境变量 -> Github](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/env.py)

## 全局配置

### `SWANLAB_FOLDER`
- **描述**: SwanLab 全局文件夹保存的路径，默认为用户主目录下的 `.swanlab` 文件夹。
- **环境变量**: `SWANLAB_FOLDER`

### `SWANLOG_FOLDER`
- **描述**: SwanLab 解析日志文件保存的路径，默认为当前运行目录的 `swanlog` 文件夹。
- **环境变量**: `SWANLOG_FOLDER`

### `SWANLAB_MODE`
- **描述**: SwanLab 的解析模式，涉及操作员注册的回调。目前有三种模式：`local`、`cloud`、`disabled`，默认为 `cloud`。**注意：大小写敏感**。
- **环境变量**: `SWANLAB_MODE`

## 服务配置

### `SWANBOARD_PROT`
- **描述**: CLI 离线看板 `swanboard` 服务的端口。
- **环境变量**: `SWANLAB_BOARD_PORT`

### `SWANBOARD_HOST`
- **描述**: CLI 离线看板 `swanboard` 服务的地址。
- **环境变量**: `SWANLAB_BOARD_HOST`

### `SWANLAB_WEB_HOST`
- **描述**: SwanLab 云端环境的 Web 地址。
- **环境变量**: `SWANLAB_WEB_HOST`

### `API_HOST`
- **描述**: SwanLab 云端环境的 API 地址。
- **环境变量**: `SWANLAB_API_HOST`

## 登录认证

### `SWANLAB_API_KEY`
- **描述**: 云端 API Key。登录时会首先查找此环境变量，如果不存在，判断用户是否已登录，未登录则进入登录流程。
  - 如果 `login` 接口传入字符串，此环境变量无效。
  - 如果用户已登录，此环境变量的优先级高于本地存储的登录信息。
- **环境变量**: `SWANLAB_API_KEY`

## 其他

### `SWANLAB_WEBHOOK`
- **描述**: Webhook 地址。SwanLab 初始化完毕时，如果此环境变量存在，会调用此地址发送消息。
- **环境变量**: `SWANLAB_WEBHOOK`