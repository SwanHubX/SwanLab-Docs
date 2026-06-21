# swanlab.register_callbacks

:::warning 已弃用
`swanlab.register_callbacks()` 已被弃用，请使用 [`swanlab.merge_callbacks()`](./py-merge-callback.md) 替代。
:::

```python
def register_callbacks(
    callbacks: CallbacksType,
) -> None:
```

| 参数        | 类型            | 描述                                         |
| ----------- | --------------- | -------------------------------------------- |
| `callbacks` | `CallbacksType` | 单个 `Callback` 对象或 `Callback` 对象列表。 |

## 介绍

使用 `swanlab.register_callbacks()` 注册回调函数，以在 SwanLab 的执行生命周期中调用。

:::tip
`register_callbacks` 必须在 **`swanlab.init()` 之前**调用。如果运行已激活，调用会报错。
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

效果等价于通过 `swanlab.init()` 传入 callbacks 参数：

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

**场景**：比如你使用时的是 SwanLab 与 Transformers 的集成，那么你要找到 `swanlab.init()` 是不容易的。那么，你可以在 `trainer.train()` 调用前，用 `swanlab.merge_callbacks()` 注册回调函数，实现插件的注入。

## 迁移方式

将 `swanlab.register_callbacks()` 替换为 `swanlab.merge_callbacks()`：

```python
# 之前（已弃用）
swanlab.register_callbacks([email_callback])

# 之后
swanlab.merge_callbacks([email_callback])
```

详细用法请参阅 [merge_callbacks API 文档](./py-merge-callback.md)。
