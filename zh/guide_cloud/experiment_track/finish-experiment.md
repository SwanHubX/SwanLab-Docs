# 结束一个实验

在一般的Python运行环境下，当脚本运行结束时，SwanLab会自动调用`swanlab.finish`来关闭实验，并将运行状态设置为「完成」。这一步无需显式调用。

但在一些特殊情况下，比如**Jupyter Notebook**中，则需要用`swanlab.finish`来显式关闭实验。

使用方式也很简单, 在`init`之后执行`finish`即可：

```python (5)
import swanlab

swanlab.init()
...
swanlab.finish()
```

## FAQ

### 在运行一次Python脚本中，我可以初始化多次实验吗？

可以，但你需要在多次`init`中间加上`finish`，如：

```python
swanlab.init()
···
swanlab.finish()
···
swanlab.init()
```