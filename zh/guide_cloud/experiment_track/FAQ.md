# FAQ

## 如何从一个脚本启动多个实验？

在多次创建实验之间增加`swanlab.finish()`即可。

执行了`swanlab.finish()`之后，再次执行`swanlab.init()`就会创建新的实验；  
如果不执行`swanlab.finish()`的情况下，再次执行`swanlab.init()`，将无视此次执行。

## 本地的训练已经结束，但SwanLab UI上仍然在运行中，要怎么改变状态？

点击实验名旁边的终止按钮，会将实验状态从“进行中”转为“中断”，并停止接收数据的上传。

![stop](/assets/stop.png)

