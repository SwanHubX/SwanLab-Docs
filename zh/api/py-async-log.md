# async_log

```python
def async_log(
    func: Callable,
    *args: Any,
    step: Optional[int] = None,
    mode: AsyncLogType = "threading",
    **kwargs: Any,
) -> Future
```

| 参数 | 描述 |
|------|------|
| `func` | (Callable) 一个返回 `dict` 的函数，返回值会被自动通过 `swanlab.log()` 记录。 |
| `*args` | 传递给 `func` 的位置参数。 |
| `step` | (Optional[int]) 可选步数。如果为 `None`，将在任务**完成时**（而非提交时）自动递增。如果步序很重要，请传入显式值。 |
| `mode` | (AsyncLogType) 执行模式。可选值为 `asyncio`、`threading`（默认）、`spawn`、`fork`。详见下方说明。 |
| `**kwargs` | 传递给 `func` 的关键字参数。 |

## 介绍

`swanlab.async_log` 异步执行一个函数并自动记录其返回值。适用于需要在训练循环中记录需要耗时计算或 I/O 的指标，而不会阻塞主流程的场景。

该调用会立即返回一个 `Future`。`swanlab.finish()` 会等待所有未完成的 `async_log` 任务完成后才刷新数据，因此不会丢失任何数据。

## 执行模式

### `threading`（默认）

在后台线程中运行 `func`。无 pickle 限制 — `func` 可以访问 `swanlab.config` 并返回 `Image`、`Audio` 等媒体对象。受 GIL 限制。

```python
import swanlab
import time

swanlab.init(project="my-project")

def fetch_score():
    time.sleep(2)
    return {"score": 0.95}

future = swanlab.async_log(fetch_score, step=1)
# 训练继续，不会被阻塞...

swanlab.finish()
```

### `asyncio`

在运行的 asyncio 事件循环上调度 `func`。`func` 必须是一个协程（`async def`）。如果没有事件循环在运行，将抛出 `RuntimeError`。

```python
import asyncio
import swanlab

swanlab.init(project="my-project")

async def slow_compute():
    await asyncio.sleep(2)
    return {"score": 0.95}

future = swanlab.async_log(slow_compute, step=1, mode="asyncio")

swanlab.finish()
```

### `spawn`

在新的子进程中运行 `func`（使用 `multiprocessing` 的 `spawn` 上下文）。绕过 GIL，适合 CPU 密集型任务。`func`、其参数和返回值**必须可被 pickle 序列化**（不支持 `Image`、`torch.Tensor` 等）。

```python
import swanlab

swanlab.init(project="my-project")

def compute_loss():
    return {"loss": 0.123, "acc": 0.95}

future = swanlab.async_log(compute_loss, step=2, mode="spawn")

swanlab.finish()
```

使用 `torch.Tensor` 时需要先转换：

```python
import torch
import swanlab

swanlab.init(project="my-project")

def compute():
    t = torch.randn(10)
    return {"value": t.item(), "arr": t.detach().cpu().numpy()}

future = swanlab.async_log(compute, step=3, mode="spawn")

swanlab.finish()
```

### `fork`（保留）

此模式暂未支持。将在 `swanlab-core` 发布后启用，届时子进程可直接调用 `swanlab.log()`。

## 注意事项

1. `func` 的返回值**必须是 `dict`**，且兼容 `swanlab.log()` 的数据格式。
2. `finish()` 会阻塞直到所有未完成的 `async_log` 任务完成，确保数据不会丢失。
3. `asyncio` 模式下需要有活跃的事件循环，否则会抛出 `RuntimeError`。
4. `spawn` 模式下，传入和返回 `func` 的所有数据必须可被 pickle 序列化。子进程无法访问活跃的 Run 实例。
