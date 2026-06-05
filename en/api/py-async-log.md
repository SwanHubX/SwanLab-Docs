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

| Parameter | Description |
|-----------|-------------|
| `func` | (Callable) A function that returns a `dict` suitable for `swanlab.log()`. The return value will be logged automatically. |
| `*args` | Positional arguments forwarded to `func`. |
| `step` | (Optional[int]) Optional step index. If `None`, it is auto-incremented when the task **completes** (not when submitted). Pass an explicit value if step ordering matters. |
| `mode` | (AsyncLogType) Execution mode. One of `asyncio`, `threading` (default), `spawn`, `fork`. See below for details. |
| `**kwargs` | Keyword arguments forwarded to `func`. |

## Introduction

`swanlab.async_log` asynchronously executes a function and automatically logs its return value. This is useful for logging metrics that require expensive computation or I/O, without blocking the training loop.

The call returns a `Future` immediately. `swanlab.finish()` waits for all outstanding `async_log` tasks before flushing, so no data is lost.

## Execution Modes

### `threading` (default)

Runs `func` in a background thread. No pickle constraints — `func` can access `swanlab.config` and return media objects such as `Image`, `Audio`, etc. Subject to the GIL.

```python
import swanlab
import time

swanlab.init(project="my-project")

def fetch_score():
    time.sleep(2)
    return {"score": 0.95}

future = swanlab.async_log(fetch_score, step=1)
# Training continues without blocking...

swanlab.finish()
```

### `asyncio`

Schedules `func` on the running asyncio event loop. `func` must be a coroutine (`async def`). Raises `RuntimeError` if no event loop is running.

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

Runs `func` in a new child process (using `multiprocessing` with `spawn` context). Bypasses the GIL, making it ideal for CPU-bound work. `func`, its arguments, and its return value **must be pickle-serializable** (no `Image`, `torch.Tensor`, etc.).

```python
import swanlab

swanlab.init(project="my-project")

def compute_loss():
    return {"loss": 0.123, "acc": 0.95}

future = swanlab.async_log(compute_loss, step=2, mode="spawn")

swanlab.finish()
```

When using `torch.Tensor`, convert it before returning:

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

### `fork` (reserved)

This mode is not yet supported. It will be enabled after the `swanlab-core` release, allowing forked children to call `swanlab.log()` directly.

## Notes

1. The return value of `func` **must be a `dict`** compatible with `swanlab.log()`.
2. `finish()` blocks until all outstanding `async_log` tasks complete, ensuring no data is lost.
3. In `asyncio` mode, an active event loop must be running; otherwise a `RuntimeError` is raised.
4. In `spawn` mode, all data passed to and returned from `func` must be pickle-serializable. The child process cannot access the active Run.
