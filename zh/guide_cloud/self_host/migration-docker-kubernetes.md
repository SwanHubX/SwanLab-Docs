# 从Docker版本迁移至K8S版本

本指南用于将 SwanLab Docker 版本的数据迁移到 **SwanLab Kubernetes（K8S）** 版本，并 仅适用于将外部服务集成至 **SwanLab Kubernetes（K8S）** 版本的场景（详见[自定义基础服务资源](/zh/guide_cloud/self_host/kubernetes-deploy.md#自定义基础服务资源)）。

若您希望迁移至各云厂商提供的托管服务或者迁移至自部署的云原生高可用服务，本指南可作为参考。同时，请参阅相应云厂商和云原生项目的官方迁移文档，并在迁移时确保数据库名称、表名称、对象存储桶名称正确。

[[toc]]

---

**此方案要求：**
1. 先迁移数据，后部署服务
2. 您使用[自定义基础服务资源](/zh/guide_cloud/self_host/kubernetes-deploy.md#自定义基础服务资源)功能
3. 您有一个busybox镜像用于实现迁移任务
4. 使用自定义负载均衡器转发对象存储服务，参考更改[应用服务类型](/zh/guide_cloud/self_host/kubernetes-deploy.md#)。

我们通过打包数据、下载数据后存储在数据卷中的方式预先制作存储卷，后挂载存储卷资源。

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
2. 您已寻找到SwanLab Docker版本挂载的数据卷路径。默认情况下，他应该在 [self-hosted](https://github.com/SwanHubX/self-hosted) 项目的`docker/swanlab/data`目录下。如果您忘记了存储的路径，可以通过`docker inspect`命令查找对应服务容器的数据卷挂载位置。
3. 您已找到swanlab生成的`docker-compose.yaml`。这主要为了迁移账号密码，如果您忘记了`docker-compose.yaml`文件的位置，您依旧可以通过`docker inspect`查找对应的账号密码环境变量。
:::

为了方便描述，接下来docker相关命令基于`self-hosted/docker/swanlab/`请根据您的实际情况调整对应的路径。此外，本指南中涉及的账号密码仅作为参考，请参考您实际的配置做调整。

请明确在Docker版本中各种基础服务资源的存储位置：

1. PostgreSQL的数据在`self-hosted/docker/swanlab/data/postgres`
2. Clickhouse的数据存储在`self-hosted/docker/swanlab/data/clickhouse`
3. Minio的数据存储在`self-hosted/docker/swanlab/data/minio`
4. Redis数据存储在`self-hosted/docker/swanlab/data/redis`

<img src="./kebunetes/datadir.png" alt="数据目录" width=300>

## 1.迁移Redis

SwanLab Docker版默认使用内置Redis服务存储核心业务数据（如实验记录、项目配置等），连接串固定为 `redis://default@redis:6379`（其中`redis`为Docker网络内的服务名，`6379`为默认端口）。迁移核心目标是将该Redis实例中的数据完整迁移至Self-Hosted版集群的Redis服务中，确保SwanLab各项功能正常运行。

:::warning
迁移前建议停止SwanLab Docker版服务，避免数据写入导致迁移不完整；同时备份Redis数据目录，防止迁移失败造成数据丢失。
:::

### 1.1 打包并上传压缩包

本步骤需在运行SwanLab Docker版的服务器上执行，核心操作是打包Redis数据目录，再上传至集群可访问的对象存储服务（如阿里云OSS、AWS S3等），为后续集群内数据复制做准备。

SwanLab Docker版的Redis数据默认挂载在宿主机的`data/redis`目录下（该路径为Docker Compose配置中指定的宿主机挂载路径，若您自定义过挂载目录，请替换为实际路径）。可通过以下命令验证目录存在性及数据完整性：

```bash
# 查看Redis数据目录结构
ls -l data/redis
# 若目录下存在dump.rdb等文件，说明数据目录正确
```

使用tar命令打包Redis数据目录，执行命令如下：

```bash
tar -czvf redis-data.tar.gz -C data/redis .
```

执行完成后，当前目录会生成`redis-data.tar.gz`压缩包。

然后，将打包好的压缩包上传至集群内所有节点可访问的网络存储服务，本例以阿里云OSS为例，上传后的文件链接样例为：：

```bash
https://xxx.oss-cn-beijing.aliyuncs.com/self-hosted/docker/redis-data.tar.gz
```

### 1.2 复制数据至Redis存储卷

本步骤通过Kubernetes Job资源在集群内执行，核心操作是从对象存储下载数据压缩包，解压后复制到Redis服务对应的持久化存储卷（PVC）中，实现数据从对象存储到集群存储卷的迁移。

创建名为`redis-migrate.yaml`的Job配置文件，参考样例如下：

```yaml {20}
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

注意PVC名称，确认无误后在集群控制节点执行以下命令，以创建并运行Job：

```bash
# 应用Job配置文件
kubectl apply -f redis-migrate.yaml
# 查看Job执行状态（RUNNING表示正在执行，SUCCEEDED表示执行成功）
kubectl get jobs -l swanlab=redis
# 查看Job对应的Pod日志，确认数据下载和解压过程
kubectl logs -l job-name=redis-migrate
```

### 1.3 启动redis服务并定义secret

本步骤需部署Redis服务（使用已迁移数据的存储卷），并创建Secret存储Redis连接串，供SwanLab Self-Hosted版服务访问。

参考redis部署配置如下：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-docker
  labels:
    app: redis-docker
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis-docker
  template:
    metadata:
      name: redis-docker
      labels:
        app: redis-docker
    spec:
      restartPolicy: Always
      containers:
        - name: redis
          image: redis-stack:7.4.0-v8
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 6379
              protocol: TCP
          volumeMounts:
            - mountPath: /data
              name: redis-volume
      volumes:
        - name: redis-volume
          persistentVolumeClaim:
            claimName: redis-docker-pvc

---

apiVersion: v1
kind: Service
metadata:
  name: redis-docker
spec:
  selector:
    app: redis-docker
  ports:
    - protocol: TCP
      port: 6379
      targetPort: 6379
  type: ClusterIP
```

此外为了与`self-hosted`集成，需要定义一个redis的secret，内容为：

| `.data.<keys>` | 值 | 说明 |
| --- | --- | --- |
| `url` | `redis://default@redis-docker:6379` | redis连接串 |

我们将此密钥名称命名为`redis-docker`。您可以通过端口转发链接redis服务，看见服务中存在若干键值对，即代表迁移成功。


## 2.迁移Postgresql

- 连接串：`postgresql://swanlab:1uuYNzZa4p@postgres-docker:5432/app?schema=public`

### 2.1 打包并上传压缩包

::: warning
注意对于pg来讲其数据目录存放在`/var/lib/postgresql/data`下，对应外部数据卷为`/data/postgres/data`，但是pg不允许`/var/lib/postgresql`目录下有除了data以外的其他文件夹，所以对于PG而言最终的压缩包为一个`data`目录
:::

使用如下命令打包pg数据：

```bash
tar -czvf postgres-data.tar.gz -C data/postgres/ .
```

然后将其上传至对象存储或任何集群可访问的网络存储服务，本例中我们上传至aliyun对象存储，文件链接为：

```
https://xxx.oss-cn-beijing.aliyuncs.com/self-hosted/docker/postgres-data.tar.gz
```

### 2.2 复制数据至存储卷

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

### 2.3 定义secret并启动PG服务

| `.data.<key>` | 值 | 说明 |
| --- | --- | --- |
| `database` | `app` | 数据库名称 |
| `username` | `swanlab` | 用户名 |
| `password` | `1uuYNzZa4p` | 密码 |
| `port` | `5432` | 端口 |
| `primaryUrl` | `postgresql://swanlab:1uuYNzZa4p@postgres-docker:5432/app?schema=public` | 可写连接串 |
| `replicaUrl` | `postgresql://swanlab:1uuYNzZa4p@postgres-docker:5432/app?schema=public` | 只读连接串，由于本例为单实例PG，所以只读和可写连接串共用一个 |

我们将此密钥命名为`postgres-docker`。

参考pg部署配置如下：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres-docker
  labels:
    app: postgres-docker
spec:
  selector:
    matchLabels:
      app: postgres-docker
  strategy:
    type: Recreate
  template:
    metadata:
      name: postgres-docker
      labels:
        app: postgres-docker
    spec:
      restartPolicy: Always
      volumes:
        - name: postgres-volume
          persistentVolumeClaim:
            claimName: postgres-docker-pvc
      containers:
        - name: postgresql
          image: postgres:16.1
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 5432
              name: postgresql
          volumeMounts:
            # 不要直接挂载到 /var/lib/postgresql/data ，因为网络盘存在 lost+found 目录会导致 postgres 启动失败
            - mountPath: /var/lib/postgresql
              name: postgres-volume
          env:
            - name: TZ
              value: "UTC"
            - name: POSTGRES_USER
              valueFrom:
                secretKeyRef:
                  name: postgres-docker
                  key: username

            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-docker
                  key: password

            - name: POSTGRES_DB
              valueFrom:
                secretKeyRef:
                  name: postgres-docker
                  key: database

---

apiVersion: v1
kind: Service
metadata:
  name: postgres-docker
spec:
  selector:
    app: postgres-docker
  ports:
    - port: 5432
      name: postgresql
```

您可以通过端口转发链接pg服务，看见服务已生成多张表，即代表迁移成功。

## 3.迁移Clickhouse

### 3.1 打包并上传压缩包

- 连接串：`tcp://swanlab:2jwnZiojEV@clickhouse-docker:9000/app`
- http端口：`8123`

使用如下命令打包pg数据：

```bash
tar -czvf clickhouse-data.tar.gz -C data/clickhouse/ .
```

然后将其上传至对象存储或任何集群可访问的网络存储服务，本例中我们上传至aliyun对象存储，文件链接为：

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
          image: busybox:1.37.0
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

### 3.3 定义secret并启动Clickhouse服务

| `.data.<key>` | 值 | 说明 |
| --- | --- | --- |
| `database` | `app` | 数据库名称 |
| `username` | `swanlab` | 用户名 |
| `password` | `2jwnZiojEV` | 密码 |
| `host` | `clickhouse-docker` | 主机 |
| `httpPort` | `8123` | http端口 |
| `tcpPort` | `9000` | tcp端口 |
`

参考clickhouse部署配置如下：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: clickhouse-docker
  labels:
    app: clickhouse-docker
spec:
  selector:
    matchLabels:
      app: clickhouse-docker
  template:
    metadata:
      name: clickhouse-docker
      labels:
        app: clickhouse-docker
    spec:
      volumes:
        - name: clickhouse-volume
          persistentVolumeClaim:
            claimName: clickhouse-docker-pvc
      containers:
        - name: clickhouse
          image: clickhouse-server:24.3
          imagePullPolicy: IfNotPresent
          volumeMounts:
            - mountPath: /var/lib/clickhouse
              name: clickhouse-volume
          ports:
            - containerPort: 9000
              name: clickhouse-tcp
            - containerPort: 8123
              name: clickhouse-http
          env:
            - name: TZ
              value: "UTC"
            - name: CLICKHOUSE_USER
              valueFrom:
                secretKeyRef:
                  name: clickhouse-docker
                  key: username
            - name: CLICKHOUSE_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: clickhouse-docker
                  key: password
            - name: CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT
              value: "1"
      restartPolicy: Always

---

apiVersion: v1
kind: Service
metadata:
  name: clickhouse-docker
spec:
  selector:
    app: clickhouse-docker
  ports:
    - port: 9000
      name: clickhouse-tcp
    - port: 8123
      name: clickhouse-http
```

您可以通过端口转发链接clickhouse服务，看见服务已生成多张表，即代表迁移成功。


## 4.迁移Minio

- accessKey: `swanlab`
- accessSecret: `qtllV4B9KZ`

### 4.1 打包并上传压缩包

使用如下命令打包pg数据：

```bash
tar -czvf minio-data.tar.gz -C data/minio/ .
```

然后将其上传至对象存储或任何集群可访问的网络存储服务，本例中我们上传至aliyun对象存储，文件链接为：

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
          image: busybox:1.37.0
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

### 4.3 定义secret并启动Minio服务

| `.data.<key>` | 值 | 说明 |
| --- | --- | --- |
| `accessKey` | `swanlab` | 访问密钥 |
| `secretKey` | `qtllV4B9KZ` | 密钥 |
| `endpoint` | `http://minio-docker:9000` | 端点 |
| `privateBucket` | `swanlab-private` | 私有桶名称 |
| `publicBucket` | `swanlab-public` | 公共桶名称 |
| `region` | `local` | 地域 |

参考minio部署配置如下：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minio-docker
  labels:
    app: minio-docker
spec:
  selector:
    matchLabels:
      app: minio-docker
  template:
    metadata:
      name: minio-docker
      labels:
        app: minio-docker
    spec:
      volumes:
        - name: minio-volume
          persistentVolumeClaim:
            claimName: minio-docker-pvc
      restartPolicy: Always
      containers:
        - name: minio
          image: minio:RELEASE.2025-09-07T16-13-09Z
          imagePullPolicy: IfNotPresent
          volumeMounts:
            - mountPath: /data
              name: minio-volume
          args:
            - "server"
            - "/data"
            - "--console-address"
            - ":9001"
          ports:
            - containerPort: 9000
              name: minio
            - containerPort: 9001
              name: minio-console
          env:
            - name: MINIO_ROOT_USER
              valueFrom:
                secretKeyRef:
                  name: minio-docker
                  key: accessKey
            - name: MINIO_ROOT_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: minio-docker
                  key: secretKey
---


apiVersion: v1
kind: Service
metadata:
  name: minio-docker
spec:
  selector:
    app: minio-docker
  ports:
    - port: 9000
      name: minio
    - port: 9001
      name: minio-console
```

您可登录minio控制台查看`swanlab-private`和`swanlab-public`两个存储桶中的数据，有数据即代表迁移成功。

### 4.4 配置路由转发规则

您的负载均衡器需要配置`/swanlab-private`和`/swanlab-public`的路由转发规则，将流量转发至部署的minio服务，以确保前端可访问头像、媒体图表等资源。

## 5.部署服务

参考values如下：

```yaml
integrations:
  postgres:
    enabled: true
    existingSecret: "postgres-docker"
  redis:
    enabled: true
    existingSecret: "redis-docker"
  clickhouse:
    enabled: true
    existingSecret: "clickhouse-docker"
  s3:
    enabled: true
    existingSecret: "minio-docker"
```

请确保您的密钥正确。在部署以后，如果您已在您的浏览器上登录了**Swanlab Docker版**，并且迁移前后域名保持一致，您不需要再次登录。