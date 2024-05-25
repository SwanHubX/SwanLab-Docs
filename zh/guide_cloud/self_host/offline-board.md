# 离线看板

SwanLab支持在不联网的情况下跟踪实验，以及访问你的实验记录。

## 离线实验跟踪

在`swanlab.init`中设置`logir`和`mode`这两个参数，即可离线跟踪实验：

```python
...

swanlab.init(
  logdir='./logs',
  mode="local",
)

...
```

- 参数`mode`设置为`local`，关闭将实验同步到云端
- 参数`logdir`的设置是可选的，它的作用是指定了SwanLab日志文件的保存位置（默认保存在`swanlog`文件夹下）
  - 日志文件会在跟踪实验的过程中被创建和更新，离线看板的启动也将基于这些日志文件

其他部分和云端使用完全一致。

## 开启离线看板

打开终端，使用下面的指令，开启一个SwanLab仪表板:

```bash
swanlab watch -l ./logs
```

运行完成后，将启动一个后端服务，SwanLab会给你1个本地的URL链接（默认是http://127.0.0.1:5092）

访问该链接，就可以在浏览器用离线看板查看实验了。

[设置端口号和IP](/zh/api/cli-swanlab-watch.md#设置ip和端口号)