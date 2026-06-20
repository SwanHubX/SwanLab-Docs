# swanlab.register_callbacks

:::warning Deprecated
`swanlab.register_callbacks()` is deprecated. Use [`swanlab.merge_callbacks()`](/en/api/py-merge-callback.md) instead.
:::

```python
def register_callbacks(
    callbacks: CallbacksType,
) -> None:
```

| Parameter   | Type            | Description                                                      |
| ----------- | --------------- | ---------------------------------------------------------------- |
| `callbacks` | `CallbacksType` | A single `Callback` object or an iterable of `Callback` objects. |

## Introduction

Use `swanlab.register_callbacks()` to register callback functions that will be invoked during the execution lifecycle of SwanLab.

:::tip
`register_callbacks` must be called **before** `swanlab.init()`. It will raise an error if a run is already active.
:::

```python {3}
from swanlab import Callback

class MyCallback(Callback):
    @property
    def name(self):
        return "my_callback"

    def on_run_initialized(self, run_dir, path, **kwargs):
        print("Run initialized!")

swanlab.register_callbacks(MyCallback())

swanlab.init(...)
```

This is equivalent to passing callbacks via `swanlab.init()`:

```python
from swanlab import Callback

class MyCallback(Callback):
    @property
    def name(self):
        return "my_callback"

    def on_run_initialized(self, run_dir, path, **kwargs):
        print("Run initialized!")

swanlab.init(
    ...
    callbacks=MyCallback(),
)
```

**Use Case**: For example, if you are using SwanLab integrated with Transformers, it might be difficult to locate `swanlab.init()`. In such cases, you can use `swanlab.merge_callbacks()` to register callbacks before calling `trainer.train()`, enabling the injection of plugins.

## Migration

Replace `swanlab.register_callbacks()` with `swanlab.merge_callbacks()`:

```python
# Before (deprecated)
swanlab.register_callbacks([email_callback])

# After
swanlab.merge_callbacks([email_callback])
```

For detailed usage, see the [merge_callbacks API documentation](/en/api/py-merge-callback.md).
