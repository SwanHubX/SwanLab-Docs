# merge_callbacks

```python
def merge_callbacks(
    callbacks: CallbacksType,
) -> None:
```

| Parameter   | Type            | Description                                                                                        |
| ----------- | --------------- | -------------------------------------------------------------------------------------------------- |
| `callbacks` | `CallbacksType` | A single `Callback` object or an iterable of `Callback` objects to merge into the global registry. |

## Introduction

`swanlab.merge_callbacks()` merges custom callbacks into the global SwanLab callback registry. This allows you to register callbacks **before** calling `swanlab.init()`, enabling plugin injection in frameworks where `swanlab.init()` is not directly accessible (e.g., when using SwanLab with Transformers).

:::tip
`merge_callbacks` must be called **before** `swanlab.init()`. It will raise a `RuntimeError` if called while a run is active.
:::

## Usage

### Single Callback

```python
import swanlab
from swanlab import Callback

class MyCallback(Callback):
    @property
    def name(self):
        return "my_callback"

    def on_run_initialized(self, run_dir, path, **kwargs):
        print("Run initialized!")

swanlab.merge_callbacks(MyCallback())
swanlab.init(project="my-project")
```

### Multiple Callbacks

Pass a list to merge multiple callbacks at once:

```python
import swanlab
from swanlab import Callback

class MyCallback(Callback):
    @property
    def name(self):
        return "my_callback"

    def on_run_initialized(self, run_dir, path, **kwargs):
        print("Run initialized!")

class AnotherCallback(Callback):
    @property
    def name(self):
        return "another_callback"

    def on_run_initialized(self, run_dir, path, **kwargs):
        print("Run initialized by another callback!")

swanlab.merge_callbacks([MyCallback(), AnotherCallback()])
swanlab.init(project="my-project")
```

## Callback Lifecycle

SwanLab callbacks implement the `Callback` base class. Common hook methods include:

- `on_run_initialized(run_dir, path, **kwargs)` — Called when a run is initialized.
- `on_run_finished(state, error)` — Called when a run finishes.
