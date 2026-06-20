# swanlab ping

```bash
swanlab ping [OPTIONS]
```

| 选项           | 描述                                                                                  |
| -------------- | ------------------------------------------------------------------------------------- |
| `-h`, `--host` | 指定 SwanLab 服务所在的主机地址，例如 `http://localhost:8000`。默认使用已登录的主机。 |

## 介绍

诊断当前机器的网络连通性与运行环境信息。包括 SDK 版本、Python 版本、运行模式、操作系统、CPU、内存、加速器（GPU 等）等。

```bash
swanlab ping
```

## 输出示例

```
╭─ SwanLab Diagnostics ─────────────────────────────────────╮
│ Server link: OK · https://api.swanlab.cn · 42 ms          │
│                                                           │
│ 1. SDK                                                    │
│   ✓ Version        0.8.0                                  │
│   ✓ Python         3.11.8                                 │
│   ✓ Mode           online                                 │
│                                                           │
│ 2. System                                                 │
│   ✓ OS             macOS 15.5 arm64                       │
│   ✓ Python         3.11.8                                 │
│   ✓ Executable     /usr/bin/python3                       │
│                                                           │
│ 3. Hardware                                               │
│   ✓ Apple Silicon  Apple M4 Pro                           │
│   ✓ CPU Cores      14                                     │
│   ✓ Unified Memory 48 GB                                  │
│                                                           │
│   Accelerators                                            │
│     - Not detected                                        │
╰───────────────────────────────────────────────────────────╯
```

### 有计算卡的输出

```
│   Accelerators                                            │
│     ✓ NVIDIA        2 devices, driver 550.54, CUDA 12.4  │
│       [0] RTX 4090  24 GB                                │
│       [1] RTX 4090  24 GB                                │
```

### 状态标识

| 标识 | 含义              |
| ---- | ----------------- |
| ✓    | 正常              |
| !    | 警告              |
| ✗    | 异常              |
| -    | 未检测到 / 未配置 |

## 诊断私有化部署

对于私有化部署的 SwanLab，使用 `--host` 指定服务地址：

```bash
swanlab ping --host http://localhost:8000
```
