# swanlab verify

```bash
swanlab verify [OPTIONS]
```

| 选项      | 描述                                                     |
| --------- | -------------------------------------------------------- |
| `--local` | 仅验证本地登录状态（检查当前目录下的 `.swanlab` 文件）。 |

## 介绍

验证当前登录状态。如果已登录，将显示服务器地址和当前登录用户名；如果未登录，则退出并提示错误。

```bash
swanlab verify
```

正常输出示例：

```
You are logged into https://api.swanlab.cn as my_username
```

如果未登录：

```
You are not logged in. Use `swanlab login` to login.
```

## 验证本地登录

使用 `--local` 选项可以验证当前目录下的本地登录状态：

```bash
swanlab verify --local
```
