# swanlab sync

```bash
swanlab sync [options] [logdir]
```

| Option | Description |
| --- | --- |
| `-k`, `--api-key` | API key for authentication. If not specified, the default API key from the environment will be used. If specified, this API key will be used for login but won't be saved. |
| `-h`, `--host` | The host address for syncing logs. If not specified, the default host (`https://swanlab.cn`) will be used. |
| `-w`, `--workspace` | The workspace for syncing logs. If not specified, the default workspace will be used. |
| `-p`, `--project` | The project for syncing logs. If not specified, the default project will be used. |

## Introduction

Sync local logs to SwanLab cloud or private deployment.

## Examples

Locate the log directory you want to upload to the cloud (by default, it's the `run-` prefixed directory under `swanlog`), then execute the command:

```bash
swanlab sync ./swanlog/run-xxx
```

::: info
By default, logs will be synced to the `project` recorded in the log files, which is the `project` set when running the experiment.  
If you want to sync to a different project, you can use the `-p` option to specify the project.
:::

If you see the following output, it indicates a successful sync:

![swanlab sync](./cli-swanlab-sync/console.png)