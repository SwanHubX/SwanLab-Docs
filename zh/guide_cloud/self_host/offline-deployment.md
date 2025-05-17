# 离线部署 SwanLab

> [!NOTE] 
>
> 该教程适用于将 SwanLab 部署在无法联网的服务器上。

## 部署流程

### 1. 下载镜像

由于私有化版 [SwanLab](https://github.com/SwanHubX/self-hosted) 基于 Docker 部署，因此我们需要先在一台联网的机器上提前下载好所有镜像。

> [!NOTE]
>
> 注意需要在相同 CPU 架构的服务器上下载镜像。比如你的服务器为 AMD64 架构，那么也需要在 AMD64 架构的服务器上拉取镜像，不能在 MacBook 这类采用 ARM64 架构的电脑上下载镜像。

找到一台联网的电脑，确保其安装有 [Docker](https://docs.docker.com/engine/install/)，然后执行 [pull-images.sh](https://github.com/SwanHubX/self-hosted/blob/main/scripts/pull-images.sh) 脚本下载镜像包。执行完成后会得到一个 `swanlab_images.tar` 的压缩包。

::: details pull-images.sh 脚本详情

```shell
#!/bin/bash

# 定义要下载的镜像列表
images=(
  "ccr.ccs.tencentyun.com/self-hosted/traefik:v3.0"
  "ccr.ccs.tencentyun.com/self-hosted/postgres:16.1"
  "ccr.ccs.tencentyun.com/self-hosted/redis-stack-server:7.2.0-v15"
  "ccr.ccs.tencentyun.com/self-hosted/clickhouse:24.3"
  "ccr.ccs.tencentyun.com/self-hosted/logrotate:v1"
  "ccr.ccs.tencentyun.com/self-hosted/fluent-bit:3.0"
  "ccr.ccs.tencentyun.com/self-hosted/minio:RELEASE.2025-02-28T09-55-16Z"
  "ccr.ccs.tencentyun.com/self-hosted/minio-mc:RELEASE.2025-04-08T15-39-49Z"
  "ccr.ccs.tencentyun.com/self-hosted/swanlab-server:v1.1.1"
  "ccr.ccs.tencentyun.com/self-hosted/swanlab-house:v1.1"
  "ccr.ccs.tencentyun.com/self-hosted/swanlab-cloud:v1.1"
  "ccr.ccs.tencentyun.com/self-hosted/swanlab-next:v1.1"
)

# 下载镜像
for image in "${images[@]}"; do
  docker pull "$image"
done

# 保存镜像到文件
echo "正在打包所有镜像到 swanlab_images.tar..."
docker save -o ./swanlab_images.tar "${images[@]}"

echo "所有镜像都打包至 swanlab_images.tar，可直接上传该文件到目标服务器!"
```

:::

###  2. 上传镜像到目标服务器

可以使用 [sftp](https://www.ssh.com/academy/ssh/sftp-ssh-file-transfer-protocol) 等命令。例如：

- 首先连接到服务器

```bash
$ sftp username@remote_host
```

- 上传文件

```sftp
> put swanlab_images.tar swanlab_images.tar
```

> [!TIP]
>
> 借助 [Termius](https://termius.com/) 这类 SSH 工具可以更方便地向服务器上传下载文件

### 3. 加载镜像

> [!NOTE]  
>
> 需求确保服务器上安装有 [Docker](https://docs.docker.com/engine/install/)

将镜像上传到目标服务器之后，需要加载镜像，命令如下：

```bash
$ docker load -i swanlab_images.tar
```

等待加载成功后，可以通过命令 `docker images` 查看镜像列表。

```bash
(base) root@swanlab:~# docker images
REPOSITORY                                              TAG                            IMAGE ID       CREATED         SIZE
ccr.ccs.tencentyun.com/self-hosted/swanlab-server       v1.1.1                         a2b992161a68   8 days ago      1.46GB
ccr.ccs.tencentyun.com/self-hosted/swanlab-next         v1.1                           7a33e5b1afc5   3 weeks ago     265MB
ccr.ccs.tencentyun.com/self-hosted/swanlab-cloud        v1.1                           0bc15f138d79   3 weeks ago     53.3MB
ccr.ccs.tencentyun.com/self-hosted/swanlab-house        v1.1                           007b252f5b6c   3 weeks ago     48.5MB
ccr.ccs.tencentyun.com/self-hosted/minio-mc             RELEASE.2025-04-08T15-39-49Z   f33e36a42eec   5 weeks ago     84.1MB
ccr.ccs.tencentyun.com/self-hosted/clickhouse           24.3                           6ffc1e932ef1   2 months ago    942MB
ccr.ccs.tencentyun.com/self-hosted/fluent-bit           3.0                            97e65b999a4d   2 months ago    84.9MB
ccr.ccs.tencentyun.com/self-hosted/traefik              v3.0                           0f62db80c71d   2 months ago    190MB
ccr.ccs.tencentyun.com/self-hosted/minio                RELEASE.2025-02-28T09-55-16Z   377fe6127f60   2 months ago    180MB
ccr.ccs.tencentyun.com/self-hosted/redis-stack-server   7.2.0-v15                      110cc99f3057   3 months ago    520MB
ccr.ccs.tencentyun.com/self-hosted/postgres             16.1                           86414087c100   16 months ago   425MB
ccr.ccs.tencentyun.com/self-hosted/logrotate            v1                             e07b32a4bfda   6 years ago     45.6MB
```

### 4. 安装 SwanLab

首先使用 Git 克隆仓库到本地目录：

```bash
$ git clone https://github.com/SwanHubX/self-hosted.git && cd self-hosted
```

然后执行脚本 `./docker/install.sh` 用于安装，安装成功会看到以下标志：

```bash
$ ./docker/install.sh

...
   _____                    _           _     
  / ____|                  | |         | |    
 | (_____      ____ _ _ __ | |     __ _| |__  
  \___ \ \ /\ / / _` | '_ \| |    / _` | '_ \ 
  ____) \ V  V / (_| | | | | |___| (_| | |_) |
 |_____/ \_/\_/ \__,_|_| |_|______\__,_|_.__/ 
                                              
 Self-Hosted Docker v1.1 - @SwanLab

🎉 Wow, the installation is complete. Everything is perfect.
🥰 Congratulations, self-hosted SwanLab can be accessed using {IP}:8000
```

> [!TIP]
>
> 默认脚本使用的镜像源在中国，所以中国地区不需要担心网络问题
>
> 如果你需要使用 [DockerHub](https://hub.docker.com/) 作为镜像源，可以使用下面的脚本进行安装：
>
> ```bash
> $ ./docker/install-dockerhub.sh
> ```

脚本执行成功后，将会在当前目录下创建一个 `swanlab/` 目录，并在目录下生成两个文件：

- `docker-compose.yaml`：用于 Docker Compose 的配置文件
- `.env`：对应的密钥文件，保存数据库对应的初始化密码

在 `swanlab` 目录下执行 `docker compose ps -a` 可以查看所有容器的运行状态：

```bash
$ docker compose ps -a                                                                                                                                             
NAME                 IMAGE                                                                   COMMAND                  SERVICE          CREATED          STATUS                    PORTS
swanlab-clickhouse   ccr.ccs.tencentyun.com/self-hosted/clickhouse:24.3                      "/entrypoint.sh"         clickhouse       22 minutes ago   Up 22 minutes (healthy)   8123/tcp, 9000/tcp, 9009/tcp
swanlab-cloud        ccr.ccs.tencentyun.com/self-hosted/swanlab-cloud:v1                     "/docker-entrypoint.…"   swanlab-cloud    22 minutes ago   Up 21 minutes             80/tcp
swanlab-fluentbit    ccr.ccs.tencentyun.com/self-hosted/fluent-bit:3.0                       "/fluent-bit/bin/flu…"   fluent-bit       22 minutes ago   Up 22 minutes             2020/tcp
swanlab-house        ccr.ccs.tencentyun.com/self-hosted/swanlab-house:v1                     "./app"                  swanlab-house    22 minutes ago   Up 21 minutes (healthy)   3000/tcp
swanlab-logrotate    ccr.ccs.tencentyun.com/self-hosted/logrotate:v1                         "/sbin/tini -- /usr/…"   logrotate        22 minutes ago   Up 22 minutes             
swanlab-minio        ccr.ccs.tencentyun.com/self-hosted/minio:RELEASE.2025-02-28T09-55-16Z   "/usr/bin/docker-ent…"   minio            22 minutes ago   Up 22 minutes (healthy)   9000/tcp
swanlab-next         ccr.ccs.tencentyun.com/self-hosted/swanlab-next:v1                      "docker-entrypoint.s…"   swanlab-next     22 minutes ago   Up 21 minutes             3000/tcp
swanlab-postgres     ccr.ccs.tencentyun.com/self-hosted/postgres:16.1                        "docker-entrypoint.s…"   postgres         22 minutes ago   Up 22 minutes (healthy)   5432/tcp
swanlab-redis        ccr.ccs.tencentyun.com/self-hosted/redis-stack-server:7.2.0-v15         "/entrypoint.sh"         redis            22 minutes ago   Up 22 minutes (healthy)   6379/tcp
swanlab-server       ccr.ccs.tencentyun.com/self-hosted/swanlab-server:v1                    "docker-entrypoint.s…"   swanlab-server   22 minutes ago   Up 21 minutes (healthy)   3000/tcp
swanlab-traefik      ccr.ccs.tencentyun.com/self-hosted/traefik:v3.0                         "/entrypoint.sh trae…"   traefik          22 minutes ago   Up 22 minutes (healthy)   0.0.0.0:8000->80/tcp, [::]:8000->80/tcp
```

通过执行 `docker compose logs <container_name>` 可以查看每个容器的日志。

### 5. 访问 SwanLab

安装成功后，可以通过 http://localhost:8000 （默认端口为8000）直接打开网站。第一次打开需要激活主账户，流程见[文档](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html#_3-%E6%BF%80%E6%B4%BB%E4%B8%BB%E8%B4%A6%E5%8F%B7)。

### 6. 升级 SwanLab

如果你使用安装脚本部署则默认安装最新版本，不需要进行升级。升级版本的脚本为：

```bash
$ ./docker/upgrade.sh
```

