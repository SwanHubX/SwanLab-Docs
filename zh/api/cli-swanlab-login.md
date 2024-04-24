# swanlab login

``` bash
swanlab login [OPTIONS]
```

| 选项 | 描述 |
| --- | --- |
| `--relogin` | 如果已经登录，强制重新登录。|

## 介绍

登录SwanLab账号，以同步实验到云端。

执行下面的命令后，如果第一次登录，会让你填写[API_KEY](#)：

```bash
swanlab login
```

登录过一次后，凭证会保存到本地，无需再次通过`swanlab.login`或`swanlab login`登录。