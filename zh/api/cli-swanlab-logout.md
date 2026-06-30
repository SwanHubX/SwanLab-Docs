# swanlab logout

```bash
swanlab logout [OPTIONS]
```

| 选项            | 描述                                            |
| --------------- | ----------------------------------------------- |
| `-f`, `--force` | 强制退出登录，跳过确认提示。                    |
| `--local`       | 退出本地登录，删除当前目录下的`.swanlab/`凭证。 |

## 介绍

退出当前登录的SwanLab账号：

```bash
swanlab logout
```

执行后会提示确认，输入`y`即可退出登录。

## 强制退出

在非交互式环境（如CI/CD）中，可以使用`--force`跳过确认：

```bash
swanlab logout --force
```

## 退出本地登录

如果之前使用`swanlab login --local`进行了本地登录，退出时也需要指定`--local`：

```bash
swanlab logout --local
```
