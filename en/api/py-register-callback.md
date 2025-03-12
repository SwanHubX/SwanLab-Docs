# swanlab.register_callback

```python
@should_call_before_init("After calling swanlab.init(), you can't call it again.")
def register_callbacks(
    self,
    callbacks: List[SwanKitCallback]
) -> None:
```

| Parameter | Type | Description |
| --- | --- | --- |
| `callbacks` | `List[SwanKitCallback]` | A list of callback functions. |

## Introduction

Use `swanlab.register_callbacks()` to register callback functions that will be invoked during the execution lifecycle of SwanLab.

```python {3}
from swanlab.plugin.writer import EmailCallback
email_callback = EmailCallback(...)
swanlab.register_callbacks([email_callback])

swanlab.init(...)
```

This is equivalent to:

```python
from swanlab.plugin.writer import EmailCallback
email_callback = EmailCallback(...)

swanlab.init(
    ...
    callbacks=[email_callback]
)
```

**Use Case**: For example, if you are using SwanLab integrated with Transformers, it might be difficult to locate `swanlab.init()`. In such cases, you can use `swanlab.register_callbacks()` to register callback functions before calling `trainer.train()`, enabling the injection of plugins.