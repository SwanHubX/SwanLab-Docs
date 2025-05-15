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

Refer to [Docker Deployment](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html)
