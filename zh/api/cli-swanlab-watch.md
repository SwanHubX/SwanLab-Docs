# swanlab watch

``` bash
swanlab watch [OPTIONS]
```

| 选项 | 描述 | 例子 |
| --- | --- | --- |
| `-p`, `--port` | 设置实验看板Web服务运行的端口，默认为**5092**。 | `swanlab watch -p 8080`：将实验看板Web服务设置为8080端口 |
| `-h`, `--host` | 设置实验看板Web服务运行的IP地址，默认为**127.0.0.1**。 | `swanlab watch -h 0.0.0.0`：将实验看板Web服务的IP地址设置为0.0.0.0 |
| `-l`, `--logdir` | 设置实验看板Web服务读取的日志文件路径，默认为`swanlog`。 | `swanlab watch --logdir ./logs`：将当前目录下的logs文件夹设置为日志文件读取路径 |
| `--help` | 查看终端帮助信息。 | `swanlab watch --help` |

## 介绍

本地启动SwanLab[离线看板](/zh/guide_cloud/self_host/offline-board.md)。  
在创建SwanLab实验时（并设置mode="local"），会在本地目录下创建一个日志文件夹（默认名称为`swanlog`），使用`swanlab watch`可以本地离线打开实验看板，查看指标图表和配置。

## 使用案例

### 打开SwanLab离线看板

首先，我们找到日志文件夹（默认名称为`swanlog`），然后在命令行执行下面的命令：

```bash
swanlab watch -l [logfile_path]
```

其中`logfile_path`是日志文件夹的路径，可以是绝对路径或相对路径。如果你的日志文件夹名称是默认的`swanlog`，那么也可以直接用`swanlab watch`启动而无需`-l`选项。

执行命令后，会看到下面的输出：
```bash{6}
swanlab watch -l [logfile_path]

*swanlab: Try to explore the swanlab experiment logs in: [logfile_path]
*swanlab: SwanLab Experiment Dashboard ready in 465ms

        ➜  Local:   http://127.0.0.1:5092
```

访问提供的URL，即可访问SwanLab离线看板。

### 设置IP和端口号

我们可以通过`-h`参数设置IP，`-p`参数设置端口号。  
比如我们希望能够在本地访问云服务器上的离线看板，那么需要在云服务器上开启实验看板时，设置IP为0.0.0.0：

```bash
swanlab watch -h 0.0.0.0
```

如果需要设置端口的话：
```bash
swanlab watch -h 0.0.0.0 -p 8080
```