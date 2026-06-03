# 常见问题

## 如何修改端口？

SwanLab 自托管版本基于 [Docker](https://www.docker.com/) 部署，默认情况下使用 `8000` 端口，修改自托管服务默认访问端口实际上是修改 **swanlab-traefik** 容器的映射端口，分为以下两种情况：

### 部署前修改

安装脚本提供有一些配置可选项，包括数据存储位置和映射的端口，我们通过修改脚本启动参数来实现修改端口。

- 执行 `install.sh` 安装脚本后，命令行会提示配置可选项，可以交互式输入对应的参数。在命令行输出 `2. Use the default port  (8000)? (y/n):` 后输入 `n`，然后会提示 `Enter a custom port:`，输入对应的端口号即可，例如 `80` 。

```bash
❯ bash install.sh
🤩 Docker is installed, so let's get started.
🧐 Checking if Docker is running...

1. Use the default path  (./data)? (y/n):
   The selected path is: ./data
2. Use the default port  (8000)? (y/n):
```

- 启动脚本时添加参数，安装脚本提供有命令行参数 `-p` 可以用于修改端口，例如： `./install.sh -p 80`。

> 更多命令行参数详见：[通过 Docker 部署](https://github.com/SwanHubX/self-hosted/tree/main/docker)

### 部署后修改

如果需要 SwanLab 服务部署完成后需要修改访问端口，则需要修改生成的 `docker-compose.yaml` 配置文件。

在脚本执行的位置找到 `swanlab/` 目录，执行 `cd swanlab/` 后进入到 `swanlab` 目录下找到对应的 `docker-compose.yaml` 配置文件，然后修改 `traefik` 容器对应的端口 `ports`，如下所示：

```yaml
  traefik:
    <<: *common
    image: ccr.ccs.tencentyun.com/self-hosted/traefik:v3.0
    container_name: swanlab-traefik
    ports:
      - "8000:80" # [!code --]
      - "80:80" # [!code ++]
```

> 上面将访问端口修改为了 `80`

修改完成后执行 `docker compose up -d` 重启容器，重启完成后即可通过 `http://{ip}:80` 访问

## 上传媒体文件报错怎么办

当你使用`swanlab.log`记录媒体文件，如图像、音频时，发现报错，如：

```bash
swanlab: Upload error: An error occurred (InvalidAccessKeyId) when calling the PutObject operation: The Access Key Id you provided does not exist in our records.
```

请检查你的服务器是否开放了`9000`端口，如果未开放，请在服务器防火墙/安全组中开放`9000`端口。