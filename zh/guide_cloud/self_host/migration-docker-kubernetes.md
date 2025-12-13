# 从Docker版本迁移至K8S版本

![migration from docker to kubernetes](./kubernetes/migration.png)

本指南用于将 SwanLab Docker 版本的数据迁移到 **SwanLab Kubernetes（K8S）** 版本，并 仅适用于将外部服务集成至 **SwanLab Kubernetes（K8S）** 版本的场景（详见[自定义基础服务资源](/zh/guide_cloud/self_host/kubernetes-deploy.md#_3-1-自定义基础服务资源)）。

若您希望迁移至各云厂商提供的托管服务或者迁移至自部署的云原生高可用服务，本指南可作为参考。同时，请参阅相应云厂商和云原生项目的官方迁移文档，并在迁移时确保数据库名称、表名称、对象存储桶名称正确。

[[toc]]

---

**此方案要求：**
1. 先迁移数据，后部署服务
2. 您使用[自定义基础服务资源](/zh/guide_cloud/self_host/kubernetes-deploy.md#_3-1-自定义基础服务资源)功能
3. 您有一个busybox镜像用于实现迁移任务
4. 确保您的Storage Class的回收策略不会因为Pod卸载而删除数据

**我们通过打包数据、下载数据后存储在数据卷中的方式预先制作存储卷，后挂载存储卷资源**。

---

**请明确您需要迁移的资源：**

1. **PostgreSQL 单实例**：用于存储 SwanLab 的核心业务数据
2. **Clickhouse 单实例**：用于存储指标数据
3. **MinIO 单实例**：用于存储媒体资源
4. **Redis 单实例**：缓存服务

其中，SwanLab-House无需进行数据迁移。

:::warning
在迁移之前，请确保：
1. 您的Docker服务已停止，或者您需要确保迁移的数据基于某一次存储快照提取
2. 您已寻找到SwanLab Docker版本挂载的数据卷路径。默认情况下，它应该在 [self-hosted](https://github.com/SwanHubX/self-hosted) 项目的`docker/swanlab/data`目录下。如果您忘记了存储的路径，可以通过`docker inspect`命令查找对应服务容器的数据卷挂载位置。
3. 您已找到swanlab生成的`docker-compose.yaml`。这主要为了迁移账号密码，如果您忘记了`docker-compose.yaml`文件的位置，您依旧可以通过`docker inspect`查找对应的账号密码环境变量。
:::

为了方便描述，接下来docker相关命令基于`self-hosted/docker/swanlab/`请根据您的实际情况调整对应的路径。

此外，本指南中涉及的账号密码仅作为参考，请参考您实际的配置做调整。

请明确在Docker版本中各种基础服务资源的存储位置：

1. PostgreSQL的数据在`self-hosted/docker/swanlab/data/postgres`
2. Clickhouse的数据存储在`self-hosted/docker/swanlab/data/clickhouse`
3. Minio的数据存储在`self-hosted/docker/swanlab/data/minio`
4. Redis数据存储在`self-hosted/docker/swanlab/data/redis`

<img src="./kubernetes/datadir.png" alt="数据目录" width=300>

<br>

---

**术语合集：**

- `self-hosted`：部署好的SwanLab Kubernetes集群

## 1.迁移Postgresql

- 连接串示例：`postgresql://swanlab:1uuYNzZa4p@postgres:5432/app?schema=public`

### 1.1 打包并上传压缩包

:::warning
注意对于pg来讲其数据目录存放在`/var/lib/postgresql/data`下，对应外部数据卷为`/data/postgres/data`，但是pg不允许`/var/lib/postgresql`目录下有除了data以外的其他文件夹，所以对于PG而言最终的压缩包为一个`data`目录
:::

使用如下命令打包pg数据：

```bash
tar -czvf postgres-data.tar.gz -C data/postgres/ .
```

然后将其上传至对象存储或任何集群可访问的网络存储服务，本例中我们上传至aliyun对象存储，文件链接样例为：

```
https://xxx.oss-cn-beijing.aliyuncs.com/self-hosted/docker/postgres-data.tar.gz
```


### 1.2 复制数据至存储卷

参考配置如下：

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: postgres-migrate
  labels:
    swanlab: postgres
spec:
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: postgres-migrate
          image: busybox:1.37.0
          imagePullPolicy: IfNotPresent
          volumeMounts:
            - name: postgres-volume
              mountPath: /data
          env:
            - name: FILE_URL
              value: "https://xxx.oss-cn-beijing.aliyuncs.com/self-hosted/docker/postgres-data.tar.gz"
          command:
            - /bin/sh
            - -c
            - |
              wget $FILE_URL -O /tmp/postgres-data.tar.gz
              tar -xzvf /tmp/postgres-data.tar.gz -C /data
      volumes:
        - name: postgres-volume
          persistentVolumeClaim:
            claimName: postgres-docker-pvc
```

### 1.3 明确你的配置

您需要修改`dependencies.postgres`下的对应资源以绑定PVC并正确设置对应的账号密码，示例如下：

```yaml
dependencies:
  postgres:
    username: "swanlab"
    password: "1uuYNzZa4p"
    persistence:
      existingClaim: "postgres-docker-pvc"
```

> `self-hosted`本身会基于`username`和`password`创建Secret资源，不会明文存储。


## 2.迁移Redis

在 SwanLab Docker 版中，Redis连接串为默认连接串`redis://default@redis:6379`

### 2.1 打包并上传压缩包

使用如下命令打包redis数据：

```bash
tar -czvf redis-data.tar.gz -C data/redis/ .
```

然后将其上传至对象存储或任何集群可访问的网络存储服务，本例中我们上传至aliyun对象存储，文件链接样例为：

```
https://xxx.oss-cn-beijing.aliyuncs.com/self-hosted/docker/redis-data.tar.gz
```

### 2.2 复制数据至存储卷

参考配置如下：

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: redis-migrate
  labels:
    swanlab: redis
spec:
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: redis-migrate
          image: busybox:1.37.0
          imagePullPolicy: IfNotPresent
          volumeMounts:
            - name: redis-volume
              mountPath: /data
          env:
            - name: FILE_URL
              value: "https://xxx.oss-cn-beijing.aliyuncs.com/self-hosted/docker/redis-data.tar.gz"
          command:
            - /bin/sh
            - -c
            - |
              wget $FILE_URL -O /tmp/redis-data.tar.gz
              tar -xzvf /tmp/redis-data.tar.gz -C /data
      volumes:
        - name: redis-volume
          persistentVolumeClaim:
            claimName: redis-docker-pvc
```

注意PVC名称，确认无误后运行即可。

### 2.3 明确你的配置

您需要修改`dependencies.redis`下的对应资源以绑定PVC并正确设置对应的账号密码，示例如下：

```yaml
dependencies:
  redis:
    persistence:
      existingClaim: "redis-docker-pvc"
```

## 3.迁移Clickhouse

### 3.1 打包并上传压缩包

- 连接串：`tcp://swanlab:2jwnZiojEV@clickhouse-docker:9000/app`
- http端口：`8123`

使用如下命令打包clickhouse数据：

```bash
tar -czvf clickhouse-data.tar.gz -C data/clickhouse/ .
```

然后将其上传至对象存储或任何集群可访问的网络存储服务，本例中我们上传至aliyun对象存储，上传后的文件链接样例为：

```
https://xxxx.oss-cn-beijing.aliyuncs.com/self-hosted/docker/clickhouse-data.tar.gz
```

### 3.2 复制数据至存储卷

参考配置如下：

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: clickhouse-migrate
  labels:
    swanlab: clickhouse
spec:
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: clickhouse-migrate
          image: busybox:latest
          imagePullPolicy: IfNotPresent
          volumeMounts:
            - name: clickhouse-volume
              mountPath: /data
          env:
            - name: FILE_URL
              value: "https://xxx.oss-cn-beijing.aliyuncs.com/self-hosted/docker/clickhouse-data.tar.gz"
          command:
            - /bin/sh
            - -c
            - |
              wget $FILE_URL -O /tmp/clickhouse-data.tar.gz
              tar -xzvf /tmp/clickhouse-data.tar.gz -C /data
      volumes:
        - name: clickhouse-volume
          persistentVolumeClaim:
            claimName: clickhouse-docker-pvc
```

### 3.3 明确你的配置

您需要修改`dependencies.clickhouse`下的对应资源以绑定PVC并正确设置对应的账号密码，示例如下：

```yaml
dependencies:
  clickhouse:
    username: "swanlab"
    password: "2jwnZiojEV"
    persistence:
      existingClaim: "clickhouse-docker-pvc"
```

> `self-hosted`本身会基于`username`和`password`创建Secret资源，不会明文存储。

## 4.迁移Minio

- accessKey: `swanlab`
- accessSecret: `qtllV4B9KZ`

### 4.1 打包并上传压缩包

使用如下命令打包minio数据：

```bash
tar -czvf minio-data.tar.gz -C data/minio/ .
```

然后将其上传至对象存储或任何集群可访问的网络存储服务，本例中我们上传至aliyun对象存储，上传后的文件链接样例为：

```
https://xxx.oss-cn-beijing.aliyuncs.com/self-hosted/docker/minio-data.tar.gz
```

### 4.2 复制数据至存储卷

参考配置如下：

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: minio-migrate
  labels:
    swanlab: minio
spec:
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: minio-migrate
          image: busybox:latest
          imagePullPolicy: IfNotPresent
          volumeMounts:
            - name: minio-volume
              mountPath: /data
          env:
            - name: FILE_URL
              value: "https://xxx.oss-cn-beijing.aliyuncs.com/self-hosted/docker/minio-data.tar.gz"
          command:
            - /bin/sh
            - -c
            - |
              wget $FILE_URL -O /tmp/minio-data.tar.gz
              tar -xzvf /tmp/minio-data.tar.gz -C /data
      volumes:
        - name: minio-volume
          persistentVolumeClaim:
            claimName: minio-docker-pvc
```

### 4.3 明确你的配置

您需要修改`dependencies.minio`下的对应资源以绑定PVC并正确设置对应的账号密码，示例如下：

```yaml
dependencies:
  s3:
    accessKey: "swanlab"
    secretKey: "qtllV4B9KZ"
    persistence:
      existingClaim: "minio-docker-pvc"
```

## 5.部署服务

当你完成上面4个步骤之后，就可以开始部署SwanLab Kubernetes服务了。

> 部署Kubernetes服务的基本操作见：[使用Kubernetes进行部署](/zh/guide_cloud/self_host/kubernetes-deploy.md)。

你只需要在原[values.yaml](https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml)的基础上，修改`dependencies`部分。

参考values如下（请确保您的密钥正确）：

```yaml
dependencies:
  postgres:
    username: "swanlab"
    password: "1uuYNzZa4p"
    persistence:
      existingClaim: "postgres-docker-pvc"
  redis:
    persistence:
      existingClaim: "redis-docker-pvc"
  clickhouse:
    username: "swanlab"
    password: "2jwnZiojEV"
    persistence:
      existingClaim: "clickhouse-docker-pvc"
  s3:
    accessKey: "swanlab"
    secretKey: "qtllV4B9KZ"
    persistence:
      existingClaim: "minio-docker-pvc"
```

完成修改后，执行以下命令部署SwanLab Kubernetes服务：

```bash
helm install swanlab-self-hosted swanlab/self-hosted -f values.yaml
```

> 在部署以后，如果您已在您的浏览器上登录了**Swanlab Docker版**，并且迁移前后域名保持一致，您不需要再次登录。

更多 Kubernetes 部署的详细操作，请参考：[使用Kubernetes进行部署](/zh/guide_cloud/self_host/kubernetes-deploy.md)。