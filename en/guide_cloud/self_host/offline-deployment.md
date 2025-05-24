# Pure Offline Environment Deployment  

> [!NOTE]  
> This guide is for deploying SwanLab in an offline server environment.  

## Deployment Process  

### 1. Download Docker Images  

Since the private version of [SwanLab](https://github.com/SwanHubX/self-hosted) is Docker-based, you need to download all required images on an internet-connected machine first.  

> [!NOTE]  
> Ensure the download machine has the same CPU architecture as the target server. For example, if your server uses AMD64, download images on an AMD64 machine—not on ARM64 devices like MacBooks.  

On a machine with Docker installed, execute the [pull-images.sh](https://github.com/SwanHubX/self-hosted/blob/main/scripts/pull-images.sh) script to download the images. This generates a `swanlab_images.tar` archive.  

::: details pull-images.sh Script Details  

```shell
#!/bin/bash  

# Define image list  
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

# Pull images  
for image in "${images[@]}"; do  
  docker pull "$image"  
done  

# Save images to file  
echo "Packaging images into swanlab_images.tar..."  
docker save -o ./swanlab_images.tar "${images[@]}"  

echo "All images are saved to swanlab_images.tar. Ready for upload to the target server!"  
```  

:::  

### 2. Upload Images to Target Server  

Use tools like [sftp](https://www.ssh.com/academy/ssh/sftp-ssh-file-transfer-protocol). Example:  

- Connect to the server:  
```bash  
$ sftp username@remote_host  
```  

- Upload the file:  
```sftp  
> put swanlab_images.tar swanlab_images.tar  
```  

> [!TIP]  
> Tools like [Termius](https://termius.com/) simplify file transfers via SSH.  

### 3. Load Images  

> [!NOTE]  
> Docker must be installed on the target server.  

After uploading, load the images:  
```bash  
$ docker load -i swanlab_images.tar  
```  

Verify loaded images with `docker images`:  
```bash  
(base) root@swanlab:~# docker images  
REPOSITORY                                              TAG                            IMAGE ID       CREATED         SIZE  
ccr.ccs.tencentyun.com/self-hosted/swanlab-server       v1.1.1                         a2b992161a68   8 days ago      1.46GB  
ccr.ccs.tencentyun.com/self-hosted/swanlab-next         v1.1                           7a33e5b1afc5   3 weeks ago     265MB  
... [truncated for brevity]  
```  

### 4. Install SwanLab Service  

After loading images, run the installation script to deploy and start services.  

On an internet-connected machine, clone the repository:  
```bash  
$ git clone https://github.com/SwanHubX/self-hosted.git  
```  

Upload the `self-hosted` folder to the target server.  

---  

On the target server, navigate to the `self-hosted` directory and execute `./docker/install.sh`. Successful installation displays:  

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
> By default, the script uses a China-based mirror source. For [DockerHub](https://hub.docker.com/), use:  
> ```bash  
> $ ./docker/install-dockerhub.sh  
> ```  

The script creates a `swanlab/` directory containing:  
- `docker-compose.yaml`: Docker Compose configuration  
- `.env`: Secret file with initialized database passwords  

Check container status in the `swanlab` directory:  
```bash  
$ docker compose ps -a  
... [container list]  
```  

View logs with `docker compose logs <container_name>`.  

### 5. Access SwanLab  

After installation, access SwanLab at `http://localhost:8000` (default port: 8000). Follow the [activation guide](https://docs.swanlab.cn/guide_cloud/self_host/docker-deploy.html#_3-activate-admin-account) for the first login.  

### 6. Upgrade SwanLab  

To upgrade:  
1. On an internet-connected machine, sync the latest `self-hosted` repository.  
2. Run the upgrade script:  
```bash  
$ ./docker/upgrade.sh  
```  
3. Export and upload new images to the target server.  
4. Upload the updated `self-hosted` folder (⚠️ Do not overwrite existing data directories).  
5. On the offline server, execute:  
```bash  
cd self-hosted  
./docker/upgrade.sh  
```  

The upgrade completes after the script finishes.