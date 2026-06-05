# swanlab ping

```bash
swanlab ping [OPTIONS]
```

| Option | Description |
| --- | --- |
| `-h`, `--host` | Specify the SwanLab server host address, e.g. `http://localhost:8000`. Defaults to the logged-in host. |

## Introduction

Diagnose network connectivity and runtime environment information for the current machine, including SDK version, Python version, mode, OS, CPU, memory, accelerators (GPU, etc.).

```bash
swanlab ping
```

## Output Example

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

### Output with Accelerators

```
│   Accelerators                                            │
│     ✓ NVIDIA        2 devices, driver 550.54, CUDA 12.4  │
│       [0] RTX 4090  24 GB                                │
│       [1] RTX 4090  24 GB                                │
```

### Status Indicators

| Indicator | Meaning |
| --- | --- |
| ✓ | OK |
| ! | Warning |
| ✗ | Error |
| - | Not detected / Not configured |

## Diagnose Self-Hosted Deployment

For self-hosted SwanLab, use `--host` to specify the server address:

```bash
swanlab ping --host http://localhost:8000
```
