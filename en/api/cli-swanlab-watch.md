# swanlab watch

``` bash
swanlab watch [OPTIONS]
```

| Option | Description | Example |
| --- | --- | --- |
| `-p`, `--port` | Set the port for the experiment dashboard web service to run on, default is **5092**. | `swanlab watch -p 8080`: Set the experiment dashboard web service to port 8080 |
| `-h`, `--host` | Set the IP address for the experiment dashboard web service to run on, default is **127.0.0.1**. | `swanlab watch -h 0.0.0.0`: Set the experiment dashboard web service IP address to 0.0.0.0 |
| `-l`, `--logdir` | Set the log file path for the experiment dashboard web service to read from, default is `swanlog`. | `swanlab watch --logdir ./logs`: Set the logs folder in the current directory as the log file read path |
| `--help` | View terminal help information. | `swanlab watch --help` |

## Introduction

Start the SwanLab experiment dashboard locally.  
When creating a SwanLab experiment, a log folder is created locally (default name is `swanlog`). Using `swanlab watch` allows you to open the experiment dashboard offline locally to view metric charts and configurations.

## Usage Examples

### Open SwanLab Offline Dashboard

First, locate the log folder (default name is `swanlog`), then execute the following command in the terminal:

```bash
swanlab watch -l [logfile_path]
```

Where `logfile_path` is the path to the log folder, which can be an absolute or relative path. If your log folder name is the default `swanlog`, you can also start it directly with `swanlab watch` without the `-l` option.

After executing the command, you will see the following output:
```bash{6}
swanlab watch -l [logfile_path]

*swanlab: Try to explore the swanlab experiment logs in: [logfile_path]
*swanlab: SwanLab Experiment Dashboard ready in 465ms

        ➜  Local:   http://127.0.0.1:5092
```

Visit the provided URL to access the SwanLab offline dashboard.

### Set IP and Port Number

We can set the IP with the `-h` parameter and the port number with the `-p` parameter.  
For example, if we want to access the offline dashboard on a cloud server locally, we need to set the IP to 0.0.0.0 when starting the experiment dashboard on the cloud server:

```bash
swanlab watch -h 0.0.0.0
```

If you need to set the port:
```bash
swanlab watch -h 0.0.0.0 -p 8080
```