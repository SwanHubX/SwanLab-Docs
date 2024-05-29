# FAQ

## 如何从一个脚本启动多个实验？

在多次创建实验之间增加`swanlab.finish()`即可。

执行了`swanlab.finish()`之后，再次执行`swanlab.init()`就会创建新的实验；  
如果不执行`swanlab.finish()`的情况下，再次执行`swanlab.init()`，将无视此次执行。

## 如何在训练时关闭swanlab记录（Debug调试）？

将`swanlab.init`的`mode`参数设置为disabled，就可以不创建实验以及不写入数据。

```python
swanlab.init(mode='disabled')
```


## 本地的训练已经结束，但SwanLab UI上仍然在运行中，要怎么改变状态？

点击实验名旁边的终止按钮，会将实验状态从“进行中”转为“中断”，并停止接收数据的上传。

![stop](/assets/stop.png)


## 如何查看折线图的局部细节？

放大折线图，长按鼠标划过目标的区域，即可放大查看该区域。

![details](/assets/faq-chart-details.png)

双击区域后复原。

## 如何取消实验的后缀名？

```python
swanlab.init(suffix=None)
```

ps: 要注意的是，目前同一个项目下，SwanLab的实验名是不允许重复的。