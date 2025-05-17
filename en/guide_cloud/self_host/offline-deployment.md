# Offline Deployment of SwanLab

> [!NOTE]
>
> This guide is for deploying SwanLab on servers without internet access.

## Deployment Process

### 1. Download Images

Since the private version of [SwanLab](https://github.com/SwanHubX/self-hosted) is deployed via Docker, we need to download all the images in advance on a machine with internet access.

> [!NOTE]
>
> Make sure to download the images on a server with the same CPU architecture. For example, if your server has AMD64 architecture, you need to pull the images on a server with AMD64 architecture as well. You cannot download the images on a computer like a MacBook that uses ARM64 architecture.

Find a computer with internet access, make sure it has [Docker](https://docs.docker.com/engine/install/) installed, and then execute the [pull-images.sh](https://github.com/SwanHubX/self-hosted/blob/main/scripts/pull-images.sh) script to download the image package. After execution, you will get a compressed package named `swanlab_images.tar`.

::: details pull-images.sh script details

```shell
#!/bin/bash

# Define the list of images to download
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

# Download images
for image in "${images[@]}"; do
  docker pull "$image"
done

# Save images to file
echo "Packing all images to swanlab_images.tar..."
docker save -o ./swanlab_images.tar "${images[@]}"

echo "All images are packed into swanlab_images.tar, you can directly upload this file to the target server!"
```

:::

### 2. Upload Images to the Target Server

You can use commands such as [sftp](https://www.ssh.com/academy/ssh/sftp-ssh-file-transfer-protocol). For example:

- First connect to the server

```bash
$ sftp username@remote_host
```

- Upload file

```sftp
> put swanlab_images.tar swanlab_images.tar
```

> [!TIP]
>
> Using SSH tools like [Termius](https://termius.com/) can make it easier to upload and download files to the server.

### 3. Load Images

> [!NOTE]
>
> Make sure [Docker](https://docs.docker.com/engine/install/) is installed on the server.

After uploading the image to the target server, you need to load the image. The command is as follows:

```bash
$ docker load -i swanlab_images.tar
```

After the loading is successful, you can view the image list through the command `docker images`.

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

### 4. Install SwanLab

First, use Git to clone the repository to your local directory:

```bash
$ git clone https://github.com/SwanHubX/self-hosted.git && cd self-hosted
```

Then execute the script `./docker/install.sh` for installation. When successfully installed, you will see the following banner:

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
> The default script uses image sources in China, so users in China don't need to worry about network issues.
>
> If you need to use [DockerHub](https://hub.docker.com/) as your image source, you can use the following script for installation:
>
> ```bash
> $ ./docker/install-dockerhub.sh
> ```

After the script executes successfully, a `swanlab/` directory will be created in the current directory, with two files generated in the directory:

- `docker-compose.yaml`: Configuration file for Docker Compose
- `.env`: Corresponding key file that saves the initialization passwords for the database

In the `swanlab` directory, execute `docker compose ps -a` to view the running status of all containers:

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

You can view the logs of each container by executing `docker compose logs <container_name>`.

### 5. Access SwanLab

After successful installation, you can open the website directly through http://localhost:8000 (default port is 8000). The first time you open it, you need to activate the main account. See [documentation](https://docs.swanlab.cn/en/guide_cloud/self_host/docker-deploy.html#_3-activate-the-primary-account) for the process.

### 6. Upgrade SwanLab

If you use the installation script for deployment, the latest version is installed by default, and you don't need to upgrade. The script for upgrading versions is:

```bash
$ ./docker/upgrade.sh
```
