如果你使用的是SwanLab与其他框架的集成，故而不太好找到`swanlab.init`，那么你可以使用`swanlab.register_callbacks`方法，在外部传入插件：

```python
import swanlab

# 等价于 swanlab.init(callbaks=[...])
swanlab.register_callbacks([...])
```