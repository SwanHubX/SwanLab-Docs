# Offline Dashboard API Documentation

:::info Note

This API is intended for **Offline Dashboard Mode** in Swanlab. It provides access to project, experiment, and chart data for data analysis.

:::


## Endpoint 1: Get Project Details

- **URL**: `/api/v1/project`
- **Method**: `GET`
- **Description**: Retrieve the currently loaded project in the Swanlab instance along with all associated experiments.

### Response Example

```json
{
  "code": 0,
  "message": "success",
  "data":{
    "id": 1,
    "name": "llamafactory",
    "experiments": [
        {
        "id": 1,
        "name": "Qwen2.5-7B/20250321-1130-16bed2e2",
        "run_id": "run-20250321_125806-a3b1799d",
        "status": 0,
        "config": { ... },
        "create_time": "2025-03-21T04:58:06.387383+00:00"
        },
        ...
    ]
  }
}
```

### Field Descriptions

- `id` / `name`: Unique project identifier and name.
- `experiments`: List of all experiments under the project.
- `logdir`: Path to the log directory.
- `charts`: Number of charts.
- `pinned_opened` / `hidden_opened`: Default expansion state of the dashboard panels.

---

## Endpoint 2: Get Single Experiment Details

- **URL**: `/api/v1/experiment/<experiment_id>`
- **Method**: `GET`
- **Example**: `/api/v1/experiment/1`
- **Description**: Retrieve configuration details and system environment of a specific experiment.

### Response Example

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": 1,
    "run_id": "run-20250321_125806-a3b1799d",
    "name": "Qwen2.5-7B/20250321-1130-16bed2e2",
    "config": { ... },
    "system": {
      "cpu": { "brand": "Intel...", "cores": 104 },
      "gpu": {
        "nvidia": {
          "type": ["NVIDIA A100-PCIE-40GB", ...],
          "memory": [40, 40, 40, 40],
          "cuda": "11.6"
        }
      },
      "os": "Linux...",
      "python": "3.10.14",
      "command": "/path/to/cli config.yaml",
      "swanlab": {
        "version": "0.5.2",
        "logdir": "/path/to/logs",
        "_monitor": 3
      }
    }
  }
}
```

### Field Descriptions

- `config`: Full configuration used during the experiment.
- `system`: System info where the experiment was run, including CPU, GPU, Python version, and command.
- `run_id`: Unique identifier for the experiment, usually linked to the log files.

---

## Endpoint 3: Get Chart Metadata

- **URL**: `/api/v1/experiment/<experiment_id>/chart`
- **Method**: `GET`
- **Example**: `/api/v1/experiment/1/chart`
- **Description**: Retrieve all chart definitions and metadata for the specified experiment (e.g., loss curves, learning rate curves).

### Response Example

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "charts": [
      {
        "id": 1,
        "name": "train/loss",
        "type": "line",
        "reference": "step",
        "source": ["train/loss"],
        "multi": false
      },
      ...
    ],
    "namespaces": [
      {
        "id": 1,
        "name": "train",
        "opened": 1,
        "charts": [1, 3, 5, 7]
      }
    ]
  }
}
```

### Field Descriptions

- `charts`: List of chart definitions, including name, type, and source.
- `namespaces`: Namespaces used to organize and categorize charts.
- `reference`: Reference for the X-axis, e.g., `step` (training step).

---

## Endpoint 4: Get Metric Data

- **URL**: `/api/v1/experiment/<experiment_id>/tag/<namespace>/<metric_name>`
- **Method**: `GET`
- **Example**: `/api/v1/experiment/1/tag/train/loss`
- **Description**: Retrieve historical data for a specific metric within an experiment (e.g., loss, accuracy).

### Response Example

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "sum": 207,
    "max": 1.7614,
    "min": 0.8499,
    "experiment_id": 1,
    "list": [
      {
        "index": 1,
        "data": 1.6858,
        "create_time": "2025-03-21T04:58:32.095272+00:00"
      },
      ...,
      {
        "index": 207,
        "data": 1.1845,
        "create_time": "2025-03-21T06:05:16.716693+00:00",
        "_last": true
      }
    ]
  }
}
```

### Field Descriptions

- `sum`: Total number of data points.
- `max` / `min`: Maximum and minimum metric values.
- `experiment_id`: ID of the associated experiment.
- `list`: Metric data list, each entry includes:
  - `index`: Sequence number of the data point.
  - `data`: Metric value.
  - `create_time`: Timestamp when the data was recorded.
  - `_last`: Indicates whether this is the last data point (`true` for last entry only).

---

## Endpoint 5: Get Recent Log Output

- **URL**: `/api/v1/experiment/<experiment_id>/recent_log`
- **Method**: `GET`
- **Example**: `/api/v1/experiment/1/recent_log`
- **Description**: Retrieve the latest log output of the specified experiment, including both Swanlab logs and user-defined output.

### Response Example

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "recent": ["swanlab:", "{'loss':"],
    "logs": [
      "swanlab: Tracking run with swanlab version 0.5.2",
      "swanlab: Run data will be saved locally in /data/project/...",
      "{'loss': 1.6858, 'grad_norm': ..., 'epoch': 0.02, ...}",
      "..."
    ]
  }
}
```

### Field Descriptions

- `recent`: Latest log snippets, usually for quick preview.
- `logs`: Full list of log outputs, including system logs and experiment runtime output.

---

## Endpoint 6: Get Experiment Status

- **URL**: `/api/v1/experiment/<experiment_id>/status`
- **Method**: `GET`
- **Example**: `/api/v1/experiment/1/status`
- **Description**: Retrieve the current status, update time, and chart layout of the specified experiment.

### Response Example

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "status": 0,
    "update_time": "2025-03-21T04:58:06.387487+00:00",
    "finish_time": null,
    "charts": {
      "charts": [
        {
          "id": 1,
          "name": "train/loss",
          "type": "line",
          "reference": "step",
          "status": 0,
          "source": ["train/loss"],
          "multi": false,
          "source_map": {"train/loss": 1}
        },
        ...
      ],
      "namespaces": [
        {
          "id": 1,
          "name": "train",
          "opened": 1,
          "charts": [1, 3, 5, 7, 9, 11]
        }
      ]
    }
  }
}
```

### Field Descriptions

- `status`: Current experiment status. (e.g., `0` for running)
- `update_time`: Timestamp of the latest update.
- `finish_time`: Timestamp of completion, or `null` if unfinished.
- `charts`: Chart structure under the experiment.
  - `charts`: List of chart definitions (same as in `/chart` endpoint).
  - `namespaces`: Chart categories and layout groups.

---

## Endpoint 7: Get Experiment Summary Metrics

- **URL**: `/api/v1/experiment/<experiment_id>/summary`
- **Method**: `GET`
- **Example**: `/api/v1/experiment/1/summary`
- **Description**: Retrieve a summary of the latest values for key metrics in the specified experiment.

### Response Example

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "summaries": [
      { "key": "train/loss", "value": 1.1845 },
      { "key": "train/grad_norm", "value": 1.0172306299209595 },
      { "key": "train/learning_rate", "value": 0.000037463413651718303 },
      { "key": "train/epoch", "value": 3.288 },
      { "key": "train/num_input_tokens_seen", "value": 597776 },
      { "key": "train/global_step", "value": 207 }
    ]
  }
}
```

### Field Descriptions

- `summaries`: List of key metrics and their most recent values.
  - `key`: Metric name (e.g., `train/loss`).
  - `value`: Current latest value of the metric.