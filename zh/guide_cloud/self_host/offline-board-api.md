# 离线看板接口文档

:::info 提示

本 API 适用于 **离线看板模式** 的 Swanlab 实验可视化。主要用于获取项目、实验、图表等数据，便于前端展示与交互。

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


