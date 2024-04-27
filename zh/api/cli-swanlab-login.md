# swanlab login

``` bash
swanlab login [OPTIONS]
```

| 选项 | 描述 |
| --- | --- |
| `--relogin` | 重新登录。|

## 介绍

登录SwanLab账号，以同步实验到云端。

执行下面的命令后，如果第一次登录，会让你填写[API_KEY](#)：

```bash
swanlab login
```

登录过一次后，凭证会保存到本地，无需再次通过`swanlab.login`或`swanlab login`登录。

## 重新登录

如果需要登录一个别的账号，则用下面的命令：

```bash
swanlab login --relogin
```

这会让你输入一个新的API Key以重新登录。

## 删除本地存储的账号信息

```bash
rm -rf ~/.swanlab
```