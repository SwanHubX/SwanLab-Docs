# 用 Notebook 跟踪实验

将 SwanLab 与 Jupyter 结合使用，无需离开Notebook即可获得交互式可视化效果。

![](./jupyter-notebook/swanlab-love-jupyter.jpg)

## 在Notebook中安装SwanLab

```bash
!pip install swanlab -qqq
```

ps: `-qqq`是用来控制命令执行时的输出信息量的，可选。

## 在Notebok中与SwanLab交互

```python
import swanlab

swanlab.init()
...
# 在Notebook中，需要显式关闭实验
swanlab.finish()
```

在用`swanlab.init`初始化实验时，打印信息的最后会出现一个“Display SwanLab Dashboard”按钮：

![](/assets/jupyter-notebook-1.jpg)

点击该按钮，就会在Notebook中嵌入该实验的SwanLab网页：

![](/assets/jupyter-notebook-2.jpg)

现在，你可以在这个嵌入的网页中直接看到训练过程，以及和它交互。
