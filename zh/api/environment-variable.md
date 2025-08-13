# 环境变量

[⚙️完整环境变量1 -> Github](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/env.py)、[⚙️完整环境变量2 -> Github](https://github.com/SwanHubX/SwanLab-Toolkit/blob/main/swankit/env.py)

## 全局配置

| 环境变量 | 描述 | 默认值 |
| --- | --- | --- |
| `SWANLAB_SAVE_DIR` | SwanLab 全局文件夹保存的路径 | 用户主目录下的 `.swanlab` 文件夹 |
| `SWANLAB_LOG_DIR` | SwanLab 解析日志文件保存的路径 | 当前运行目录的 `swanlog` 文件夹 |
| `SWANLAB_MODE` | SwanLab 的解析模式，涉及操作员注册的回调。目前有三种模式：`local`、`cloud`、`disabled`。**注意：大小写敏感** | `cloud` |

## 服务配置

| 环境变量 | 描述 | 
| --- | --- |
| `SWANLAB_BOARD_PORT` | CLI 离线看板 `swanboard` 服务的端口 |
| `SWANLAB_BOARD_HOST` | CLI 离线看板 `swanboard` 服务的地址 |
| `SWANLAB_WEB_HOST` | SwanLab 云端环境的 Web 地址，私有化部署仅需设置此环境变量而无需设置 `SWANLAB_API_HOST` |
| `SWANLAB_API_HOST` | SwanLab 云端环境的 API 地址 |

## 实验配置

| 环境变量 | 描述 |
| --- | --- |
| `SWANLAB_PROJ_NAME` | 项目名称，效果等价于 `swanlab.init(project="...")` |
| `SWANLAB_WORKSPACE` | 工作空间名称，效果等价于 `swanlab.init(workspace="...")` |
| `SWANLAB_EXP_NAME` | 实验名称，效果等价于 `swanlab.init(experiment_name="...")` |
| `SWANLAB_RUN_ID` | 实验运行ID，效果等价于 `swanlab.init(id="...")` |
| `SWANLAB_RESUME` | 是否断点续训，效果等价于 `swanlab.init(resume=...)`，可选值为 `must`、`allow`、`never` |

## 登录认证

| 环境变量 | 描述 |
| --- | --- | 
| `SWANLAB_API_KEY` | 云端 API Key。登录时会首先查找此环境变量，如果不存在，判断用户是否已登录，未登录则进入登录流程。<br>- 如果 `login` 接口传入字符串，此环境变量无效<br>- 如果用户已登录，此环境变量的优先级高于本地存储的登录信息 |

## 其他

| 环境变量 | 描述 |
| --- | --- |
| `SWANLAB_WEBHOOK` | Webhook 地址。<br> SwanLab 初始化完毕时，如果此环境变量存在，会调用此地址发送消息 |