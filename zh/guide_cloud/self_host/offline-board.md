# 离线看板

:::warning 注意

离线看板是SwanLab的历史功能，现阶段仅做简单维护，不再更新。

如果您有私有化部署的需求，推荐使用[Docker版](/guide_cloud/self_host/docker-deploy)。

:::

离线看板是一种使用模式接近`tensorboard`的轻量级离线web看板。

Github：https://github.com/SwanHubX/SwanLab-Dashboard


## 安装

> 在swanlab>=0.5.0版本后，不再自带离线看板，需要使用dashboard扩展安装。

使用离线看板，需要安装`swanlab`的`dashboard`扩展：

```bash
pip install swanlab[dashboard]
```

## 离线实验跟踪

在`swanlab.init`中设置`logdir`和`mode`这两个参数，即可离线跟踪实验：

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
swanlab watch ./logs
```

> 谐音：用swanlab看 ./logs 里的文件

运行完成后，将启动一个后端服务，SwanLab会给你1个本地的URL链接（默认是http://127.0.0.1:5092）

访问该链接，就可以在浏览器用离线看板查看实验了。

[如何设置端口号和IP](/api/cli-swanlab-watch.md#设置ip和端口号)