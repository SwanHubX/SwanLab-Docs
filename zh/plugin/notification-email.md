# 邮件通知

[Github源代码](https://github.com/swanhubx/swanlab/blob/main/swanlab/plugin/notification.py)

如果你希望在训练完成/发生错误时，第一时间发送邮件通知你，那么非常推荐你使用`邮件通知`插件类。

## 准备工作

在使用插件前，首先你需要准备开通：



## 使用方法

```python
from swanlab.plugin.notification import EmailCallback

swan_email = EmailCallback(
    sender_email="zeyi.lin@swanhub.co",  # 发送者邮箱
    receiver_email="xiaolin199912@gmail.com",  # 接收者邮箱
    password="7cKCiuKEQVqtI6v1",  # 发送者邮箱密码
    smtp_server="smtp.feishu.cn",  # 发送者邮箱服务器
    port=587,  # 发送者邮箱服务器端口
    language="zh",  # 语言
)

swanlab.init(callbacks=[swan_email])

...

if accuracy > 0.95:
    # 自定义场景发送邮件
    swan_email.send_email(
        subject="SwanLab | Accuracy > 0.95",
        content=f"Current Accuracy: {accuracy}",
    )
```

## 使用方法

## 限制