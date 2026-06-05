# merge_callbacks

```python
def merge_callbacks(
    callbacks: CallbacksType,
) -> None:
```

| 参数 | 类型 | 描述 |
| --- | --- | --- |
| `callbacks` | `CallbacksType` | 单个 `Callback` 对象或 `Callback` 对象列表，将被合并到全局回调注册表中。 |

## 介绍

`swanlab.merge_callbacks()` 将自定义回调函数合并到 SwanLab 的全局回调注册表中。这允许你在调用 `swanlab.init()` **之前**注册回调，实现在无法直接访问 `swanlab.init()` 的框架（例如与 Transformers 集成时）中注入插件。

:::tip
`merge_callbacks` 必须在 **`swanlab.init()` 之前**调用。如果运行已激活时调用，会抛出 `RuntimeError`。
:::

## 用法

### 单个回调

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

### 多个回调

传入列表可一次性合并多个回调：

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

## 回调生命周期

SwanLab 回调通过继承 `Callback` 基类实现。常用的钩子方法包括：

- `on_run_initialized(run_dir, path, **kwargs)` — 实验初始化时调用。
- `on_run_finished(state, error)` — 实验结束时调用。
