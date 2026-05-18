# swanlab.register_callback

```python
@should_call_before_init("After calling swanlab.init(), you can't call it again.")
def register_callbacks(
    self,
    callbacks: List[SwanKitCallback]
) -> None:
```

| 参数 | 类型 | 描述 |
| --- | --- | --- |
| `callbacks` | `List[SwanKitCallback]` | 回调函数列表 |


## 介绍

使用`swanlab.register_callbacks()`注册回调函数，以在SwanLab的执行生命周期中调用。

```python {3}
from swanlab.plugin.writer import EmailCallback
email_callback = EmailCallback(...)
swanlab.register_callbacks([email_callback])

swanlab.init(...)
```

效果等价于：

```python
from swanlab.plugin.writer import EmailCallback
email_callback = EmailCallback(...)

swanlab.init(
    ...
    callbacks=[email_callback]
)
```

**场景**：比如你使用时的是SwanLab与Transformers的集成，那么你要找到`swanlab.init()`是不容易的。那么，你可以在`trainer.train()`调用前，用`swanlab.register_callbacks()`注册回调函数，实现插件的注入。