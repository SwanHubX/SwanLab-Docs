# swanlab login

``` bash
swanlab login [OPTIONS]
```

| 选项 | 描述 |
| --- | --- |
| `-r`, `--relogin` | 重新登录。|
| `-h`, `--host` | 指定SwanLab服务所在的主机。比如`http://localhost:8000`。|
| `-k`, `--api-key` | 指定API Key。如果您不喜欢使用命令行来输入 API 密钥，这将允许自动登录。|
| `-w`, `--web-host` | 指定SwanLab前端所在的Web主机。|

## 介绍

登录SwanLab账号，以同步实验到云端。

执行下面的命令后，如果第一次登录，会让你填写[API_KEY](https://swanlab.cn/settings)：

```bash
swanlab login
```

登录过一次后，凭证会保存到本地，并覆盖之前登录过的凭证，无需再次通过`swanlab.login`或`swanlab login`登录。

> 如果你不希望凭证保存在本地，请在python脚本中使用[swanlab.login()](./py-login.md)进行登录。

如果你的电脑不太适合命令行粘贴API Key（比如一些Windows CMD）的方式登录，可以使用：

```bash
swanlab login -k <api-key>
```


## 重新登录

如果需要登录一个别的账号，则用下面的命令：

```bash
swanlab login --relogin
```

这会让你输入一个新的API Key以重新登录。

## 退出登录

```bash
swanlab logout
```

## 登录到私有化服务

```bash
swanlab login --host <host>
```

