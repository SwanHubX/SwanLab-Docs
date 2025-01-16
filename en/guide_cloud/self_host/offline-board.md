# Offline Dashboard

SwanLab supports tracking experiments and accessing your experiment records without an internet connection.

## Offline Experiment Tracking

Set the `logdir` and `mode` parameters in `swanlab.init` to track experiments offline:

```python
...

swanlab.init(
  logdir='./logs',
  mode="local",
)

...
```

- The `mode` parameter is set to `local`, which disables syncing experiments to the cloud.
- The `logdir` parameter is optional and specifies the location where SwanLab log files are saved (default is the `swanlog` folder).
  - Log files will be created and updated during the experiment tracking process, and the offline dashboard will be based on these log files.

The rest of the usage is identical to cloud usage.

## Start Offline Dashboard

Open the terminal and use the following command to start a SwanLab dashboard:

```bash
swanlab watch ./logs
```

> Pronunciation hint: Use swanlab to watch the files in ./logs.

After running, a backend service will be started, and SwanLab will provide you with a local URL link (default is http://127.0.0.1:5092).

Access this link to view experiments in the browser using the offline dashboard.

[How to Set IP and Port Number](/en/api/cli-swanlab-watch.md)