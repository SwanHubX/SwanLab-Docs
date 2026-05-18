# Offline Dashboard

:::warning Note

The offline dashboard is a legacy feature of SwanLab and is currently maintained with minimal updates. It is no longer actively developed.

For private deployment needs, we recommend using the [Docker version](/guide_cloud/self_host/docker-deploy).

:::

The offline dashboard is a lightweight web-based dashboard that operates similarly to `tensorboard` in offline mode.

GitHub: https://github.com/SwanHubX/SwanLab-Dashboard

## Installation

> Starting from SwanLab version 0.5.0, the offline dashboard is no longer included by default and must be installed via the `dashboard` extension.

To use the offline dashboard, install the `dashboard` extension for `swanlab`:

```bash
pip install swanlab[dashboard]
```

## Offline Experiment Tracking

To track experiments offline, configure the `logdir` and `mode` parameters in `swanlab.init`:

```python
...

swanlab.init(
  logdir='./logs',
  mode="local",
)

...
```

• Set the `mode` parameter to `local` to disable syncing experiments to the cloud.
• The `logdir` parameter is optional and specifies the directory where SwanLab log files are saved (default is the `swanlog` folder).
  • Log files are created and updated during experiment tracking, and the offline dashboard relies on these files for operation.

All other functionalities remain consistent with the cloud version.

## Launching the Offline Dashboard

Open a terminal and use the following command to start a SwanLab dashboard:

```bash
swanlab watch ./logs
```

> Mnemonic: Use SwanLab to watch the files in `./logs`.

After execution, a backend service will be launched, and SwanLab will provide a local URL (default is http://127.0.0.1:5092).

Access this URL in your browser to view experiments using the offline dashboard.

[How to Set Port and IP](/api/cli-swanlab-watch.md#set-ip-and-port)