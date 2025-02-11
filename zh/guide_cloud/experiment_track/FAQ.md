# FAQ

## 登录时，API Key为什么输入不进去？

见此回答：[链接](https://www.zhihu.com/question/720308649/answer/25076837539)


## 如何从一个脚本启动多个实验？

在多次创建实验之间增加`swanlab.finish()`即可。

执行了`swanlab.finish()`之后，再次执行`swanlab.init()`就会创建新的实验；  
如果不执行`swanlab.finish()`的情况下，再次执行`swanlab.init()`，将无视此次执行。

## 如何在训练时关闭swanlab记录（Debug调试）？

将`swanlab.init`的`mode`参数设置为disabled，就可以不创建实验以及不写入数据。

```python
swanlab.init(mode='disabled')
```

## 在同一台机器上，有多个人都在使用SwanLab，应该如何配置？

`swanlab.login`登录完成之后，会在该机器上生成一个配置文件记录登录信息，以便下次不用重复登录。但如果有多人使用这一台机器的话，则需要小心日志传递到对方账号上。

**推荐的配置方式有两种：**

**方式一**：在代码开头加上`swanlab.login(api_key='你的API Key')`

**方式二**：在运行代码前，设置环境变量`SWANLAB_API_KEY="你的API Key"`


## 本地的训练已经结束，但SwanLab UI上仍然在运行中，要怎么改变状态？

点击实验名旁边的终止按钮，会将实验状态从“进行中”转为“中断”，并停止接收数据的上传。

![stop](/assets/stop.png)


## 如何查看折线图的局部细节？

放大折线图，长按鼠标划过目标的区域，即可放大查看该区域。

![details](/assets/faq-chart-details.png)

双击区域后复原。


## 内部指标名

指标名称是指`swanlab.log()`传入字典的key部分。有一部分key在内部被SwanLab用于传递系统硬件指标，所以不太建议使用。

内部指标包括：

- `__swanlab__.xxx`