# Bark

[Bark](https://github.com/Finb/Bark) 是一款专为 iOS 设备设计的免费、开源的消息推送服务应用程序。它的核心功能非常简单纯粹：让你能够从任何设备或平台，向自己的 iPhone 发送自定义推送通知。

你可以把它理解为一个“私人定制版”的推送服务，类似于 IFTTT 或 Telegram Bot，但更轻量、更专注于通知本身，并且数据完全由你自己掌控。

如果你希望在训练完成/发生错误时，第一时间发送 Bark 信息通知你，那么非常推荐你使用 Bark 通知插件。

:::warning 改进插件
SwanLab插件均为开源代码，你可以在[Github源代码](https://github.com/swanhubx/swanlab/blob/main/swanlab/plugin/notification.py)中查看，欢迎提交你的建议和PR！
:::

[[toc]]

## 准备工作

1. 打开Bark App
2. 在「服务器」页面，复制URL（url的格式为`https://api.day.app/XXXXXXX`），记下来
3. 在「设置」页面，复制Device Token，记下来


## 基本用法

使用Bark通知插件的方法非常简单，只需要初始化1个`BarkCallback`对象，将准备工作中的URL和Device Token填入：

```python
from swanlab.plugin.notification import BarkCallback

bark_callback = BarkCallback(
    key='你的Device Token',
    url='你的URL', 
)
```

然后将`bark_callback`对象传入`swanlab.init`的`callbacks`参数中：

```python
swanlab.init(callbacks=[bark_callback])
```

这样，当训练完成/发生错误时（触发`swanlab.finish()`），你将会收到Bark信息通知。

<img src="./notification-bark/show.png" width="300"/>

## 自由提醒

你还可以使用`BarkCallback`对象的`send_msg`方法，发送自定义的钉钉信息。

这在提醒你某些指标达到某个阈值时非常有用！

```python 
if accuracy > 0.95:
    # 自定义场景发送消息
    bark_callback.send_msg(
        content=f"Current Accuracy: {accuracy}",  # 通知内容
    )
```

## 外部注册插件

<!--@include: ./shared-snippet.md-->


## 限制

- Bark 通知插件的训练完成/异常通知，使用的是`SwanKitCallback`的`on_stop`生命周期回调，所以如果你的进程被突然`kill`，或者训练机异常关机，那么会因为无法触发`on_stop`回调，从而导致未发送Bark通知。