# 多人共用服务器避免密钥冲突

**使用场景：**

1. **实验室**：多个训练者共同使用一台服务器做训练时，需要避免SwanLab API Key造成冲突，导致实验数据误传到其他人的账号上
2. **公共服务器**：在公共服务器上，避免SwanLab API Key泄露造成数据安全问题

## 临时登录

在你的Python代码中，加上下面这一行：

```python
swanlab.login(api_key="<your_api_key>")
```

使用`swanlab.login()`进行登录，不会将登录信息写入到本地，这样一方面能保证实验一定是上传到你的账号下，另一方面其他人也不能上传到你的账号。

## 本地登录

::: tip
`--local`选项仅在 SDK ≥ 0.8.0 版本中可用。
:::

使用`--local`选项，可以将登录凭证仅保存在当前项目目录下（`.swanlab/` 文件夹），而非用户主目录。这样每个项目使用独立的凭证，避免多人共用服务器时发生密钥冲突：

```bash
swanlab login --local
```

退出本地登录：

```bash
swanlab logout --local
```

## 退出登录

使用`swanlab logout`会清空本地存储的登录信息，建议在离开公共服务器前执行。

```bash
swanlab logout
```
