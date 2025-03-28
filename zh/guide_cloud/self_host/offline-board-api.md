# 离线看板接口文档

:::info 提示

本 API 适用于 Swanlab **离线看板模式**。主要用于获取项目、实验、图表等数据，便于进行数据分析。

:::


## 接口 1：获取项目详情

- **URL**：`/api/v1/project`
- **方法**：`GET`
- **接口说明**：获取当前 Swanlab 实例中加载的项目及其所有实验信息。

### 响应示例

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

### 字段说明

- `id` / `name`：项目唯一标识与名称。
- `experiments`：该项目下的所有实验信息。
- `logdir`：日志文件存储路径。
- `charts`：图表数量。
- `pinned_opened` / `hidden_opened`：控制面板的默认展开状态。

---

## 接口 2：获取单个实验详情

- **URL**：`/api/v1/experiment/<experiment_id>`
- **方法**：`GET`
- **示例**：`/api/v1/experiment/1`
- **接口说明**：获取指定实验的详细配置信息与系统环境。

### 响应示例

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

### 字段说明

- `config`：实验的完整参数配置。
- `system`：运行时主机的系统信息，包括 CPU、GPU、Python 版本、命令等。
- `run_id`：实验的唯一标识符，通常与日志文件关联。

---

## 接口 3：获取实验图表信息

- **URL**：`/api/v1/experiment/<experiment_id>/chart`
- **方法**：`GET`
- **示例**：`/api/v1/experiment/1/chart`
- **接口说明**：获取指定实验的所有图表定义和元信息（如 loss 曲线、学习率曲线等）。

### 响应示例

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

### 字段说明

- `charts`：图表定义列表，包括图表名称、类型、数据来源等。
- `namespaces`：图表命名空间，用于分类展示。
- `reference`：图表的 X 轴参考，5982 `step`（训练步数）。

---

## 接口 4：获取指标数据

- **URL**：`/api/v1/experiment/<experiment_id>/tag/<namespace>/<metric_name>`
- **方法**：`GET`
- **示例**：`/api/v1/experiment/1/tag/train/loss`
- **接口说明**：获取指定实验中某个具体指标的历史数据（如 loss、accuracy 等）。

### 响应示例

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

### 字段说明

- `sum`：数据总条数。
- `max` / `min`：指标最大值与最小值。
- `experiment_id`：所属实验 ID。
- `list`：具体数据项，每项包括：
  - `index`：数据点序号。
  - `data`：具体数值。
  - `create_time`：记录时间。
  - `_last`：是否为最后一个数据点（仅最后一条为 true）。

---

## 接口 5：获取实验最新日志

- **URL**：`/api/v1/experiment/<experiment_id>/recent_log`
- **方法**：`GET`
- **示例**：`/api/v1/experiment/1/recent_log`
- **接口说明**：获取指定实验最新的日志输出。包括 Swanlab 自身日志信息和用户自定义的输出。

### 响应示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "recent": [
      "swanlab:",
      "{'loss':"
    ],
    "logs": [
      "swanlab: Tracking run with swanlab version 0.5.2",
      "swanlab: Run data will be saved locally in /data/project/...",
      "{'loss': 1.6858, 'grad_norm': ..., 'epoch': 0.02, ...}",
      "..."
    ]
  }
}
```

### 字段说明

- `recent`：最新日志段落，通常用于快速预览。
- `logs`：日志输出列表，包含 swanlab 系统日志和运行中的配置、输出数据。

---

## 接口 6：获取实验状态信息

- **URL**：`/api/v1/experiment/<experiment_id>/status`
- **方法**：`GET`
- **示例**：`/api/v1/experiment/1/status`
- **接口说明**：获取指定实验的最新状态、更新时间、图表结构等信息。

### 响应示例

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
        {
          "id": 3,
          "name": "train/grad_norm",
          "type": "line",
          "reference": "step",
          "status": 0,
          "source": ["train/grad_norm"],
          "multi": false,
          "source_map": {"train/grad_norm": 1}
        },
        {
          "id": 5,
          "name": "train/learning_rate",
          "type": "line",
          "reference": "step",
          "status": 0,
          "source": ["train/learning_rate"],
          "multi": false,
          "source_map": {"train/learning_rate": 1}
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

### 字段说明

- `status`：实验当前状态，整型（如 0 表示运行中）。
- `update_time`：实验状态最近更新时间。
- `finish_time`：实验完成时间，未完成为 `null`。
- `charts`：实验中的图表结构信息。
  - `charts`：图表定义数组，字段与 `/chart` 接口一致。
  - `namespaces`：图表命名空间，标识图表分类与分组。

---

## 接口 7：获取实验指标汇总

- **URL**：`/api/v1/experiment/<experiment_id>/summary`
- **方法**：`GET`
- **示例**：`/api/v1/experiment/1/summary`
- **接口说明**：获取指定实验在当前状态下的各项关键指标的最新值汇总。

### 响应示例

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

### 字段说明

- `summaries`：包含多个指标的汇总值，每项包括：
  - `key`：指标名称（如 `train/loss`）。
  - `value`：该指标当前最新值。
