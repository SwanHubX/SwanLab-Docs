# swanlab.login

``` bash
login(
    api_key: str = None,
    host: str = None,
    web_host: str = None,
    save: bool = False
):
```

| 参数 | 描述 |
| --- | --- |
| `api_key` | (str) 身份验证密钥，如果未提供，密钥将从密钥文件中读取。|
| `host` | (str) SwanLab服务所在的API主机，如果未提供，将使用默认主机（即云端版）|
| `web_host` | (str) SwanLab服务所在的Web主机，如果未提供，将使用默认主机（即云端版）|
| `save` | (bool) 是否将API密钥保存到密钥文件中，默认值为False。|


## 介绍

在Python代码中登录SwanLab账号，以将实验上传到指定的云端服务器。API Key从你的SwanLab「设置」-「常规」页面中获取。

## 登录到公有云

```python
import swanlab

swanlab.login(api_key='your-api-key', save=True)
```

默认将登录到`swanlab.cn`，即SwanLab公有云服务。

如果需要登录到其他主机，可以指定`host`参数，如`http://localhost:8000`。

将`save`参数设置为`True`，会将登录凭证保存到本地（会覆盖之前保存的凭证），无需再次通过`swanlab.login`或`swanlab login`登录。

**如果你在公共机器上使用，请将`save`参数设置为`False`**，这样不会泄露你的API Key，也避免其他人不小心上传数据到你的空间。

## 登录到私有化服务

```python
swanlab.login(api_key='your-api-key', host='your-private-host')
```

