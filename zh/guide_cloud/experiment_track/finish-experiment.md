# 结果一个实验

在脚本运行结束时，SwanLab自动调用`swanlab.finish`来关闭实验。同时，你也可以使用 `swanlab.finish` API手动结束实验。

以下代码示例演示了如何用`swanlab.finish`语句结束实验：

```python
import swanlab

swanlab.init()
...
swanlab.finish()
```