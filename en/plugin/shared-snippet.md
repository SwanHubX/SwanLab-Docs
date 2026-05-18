If you are using the integration of SwanLab with other frameworks and thus find it difficult to locate `swanlab.init`, you can use the `swanlab.register_callbacks` method to pass in plugins externally:

```python
import swanlab

# Equivalent to swanlab.init(callbacks=[...])
swanlab.register_callbacks([...])
```