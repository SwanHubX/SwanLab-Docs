# 数据迁移

> 本页介绍 SwanLab 私有化版本在不同 Kubernetes 集群之间进行全量数据迁移的完整流程。

:::warning
如您使用外接云数据库方案「暂不推荐」，可忽略此迁移文档的流程
:::

## 迁移流程示意图

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/cross-cluster-migration.drawio.svg"/>

集群**本地数据库部署方案**下， SwanLab 私有化服务的数据库进行跨集群数据迁移的流程，包含三个核心区域：

- **源集群 (Original)**：数据导出端，包含数据库和对象存储。
- **中转存储 (Transit)**：S3 对象存储作为临时中转站
- **目标集群 (Current)**：数据接收端，完成恢复。

| 数据类型                                  | 迁移方式              | 说明                                      |
| ----------------------------------------- | --------------------- | ----------------------------------------- |
| 数据库（PostgreSQL / ClickHouse / Redis） | 导出 → S3 中转 → 导入 | 物理迁移                                  |
| 对象存储（MiniO / S3）                    | 直接对接云端 S3       | 存算分离，免搬运，新集群直接通过 API 读写 |

**图例说明**：

- 🔵 **Phase 1: Export** — 源集群导出数据到 S3 中转存储桶（DB Export + S3 Sync）
- 🟢 **Phase 2: Import** — 数据库数据从 S3 导入到新集群（DB Import，物理迁移）
- 🟢 **Phase 2: Direct Use**（虚线）— 对象存储不移动，新集群直接通过 `value.yaml` 同样的配置访问 public 和 private 对象存储桶

## 🧾 前置条件

### 资源准备

- 兼容 S3 协议的对象存储桶，可用空间至少大于 ClickHouse + PostgreSQL + Redis + Minio（如有）的存储总和的 **1.1 倍**
- 原集群中的可用存储资源可支持上述组件的额外压缩包存储空间
- 目标集群需要额外部署一套 **全新未激活** 的 SwanLab 服务，且必须挂载云硬盘
- 如果原始集群和目标集群版本不一致，建议先对原始集群的 chart 版本进行升级

### 权限准备

- 对原始集群和目标集群均具有可写权限（原始集群和目标集群可以是同一个集群，但必须各自存在一套独立的 SwanLab 服务）
- 准备好 `access_key` 和 `secret_key`，需要具备对象存储桶的读写权限

### 配置变量

使用前请确认以下信息：

| 是否准备 | 占位变量                                | 说明                                                          |
| -------- | --------------------------------------- | ------------------------------------------------------------- |
| ✅       | `origin_namespace` / `target_namespace` | 原始集群和目标集群的命名空间                                  |
| ✅       | `S3_REGION`                             | 用于备份的对象存储桶的地域                                    |
| ✅       | `S3_BUCKET`                             | 存储桶名称                                                    |
| ✅       | `S3_ENDPOINT`                           | S3 格式的对象存储 Endpoint，如 `tos-s3-cn-beijing.volces.com` |
| ✅       | `S3_AK` / `S3_SK`                       | 对象存储可写密钥                                              |
| ✅       | `S3_PATH_PREFIX`                        | S3 中的备份路径前缀，默认 `origin-backup-datas`               |
| ✅       | 原始集群各组件 PVC 名称                 | `kubectl get pvc -n <origin_namespace>`                       |

> 💡 经实践检验，可以直接通过传入 PVC 的方式，让 K8s 自动处理挂载。

> ⚠️ S3 迁移工具默认使用 `s3cmd`，已在腾讯云 COS、火山引擎 TOS 上经过验证。阿里云 OSS 作为中转时需使用 `ossutil`，需替换传输工具。

## 🪜 操作步骤

:::warning

- 数据迁移时需要确保**两套 SwanLab 服务都保持停机状态**，由一个中间 Job 执行迁移，否则会因为状态不一致等问题，造成迁移失败。
- **迁移前必须停机！**
  :::

### 1. 修改配置文件

- 操作位置：<span style="color: red"><strong>源集群、目标集群</strong></span>

创建 S3 配置（ConfigMap + Secret），需要**复制为两份**，**分别填写源集群和目标集群的命名空间，并各自执行**：

::: details config-export.yaml / config-import.yaml

::: code-group

```yaml [config-export.yaml]
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-backup-storage-config
  namespace: <original_namespace> # ⚠️ 必填：【原始集群】K8s 命名空间
data:
  S3_REGION: "cn-beijing" # 必填：用于中转的对象存储地域（如 cn-beijing）
  S3_BUCKET: "swanlab-backup-demo" # 必填：存储桶名称
  S3_ENDPOINT: "tos-s3-cn-beijing.volces.com" # 必填：S3 格式的对象存储 Endpoint
  S3_PATH_PREFIX: "origin-backup-datas"
  # 选填：原始集群本机数据目录路径（根据实际部署填写）
  HOST_POSTGRES_PATH: "/var/lib/swanlab-postgres" # PostgreSQL 数据目录
  HOST_CLICKHOUSE_PATH: "/var/lib/swanlab-clickhouse" # ClickHouse 数据目录
  HOST_REDIS_PATH: "/var/lib/swanlab-redis" # Redis 数据目录
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-backup-storage-secret
  namespace: <original_namespace> # ⚠️ 必填：【原始集群】K8s 命名空间
type: Opaque
stringData:
  S3_AK: "xxx" # 必填：对象存储 AccessKey
  S3_SK: "xxx" # 必填：对象存储 SecretKey
```

```yaml [config-import.yaml]
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-backup-storage-config
  namespace: <target_namespace> # ⚠️ 必填：【目标集群】K8s 命名空间
data:
  S3_REGION: "cn-beijing" # 必填：用于中转的对象存储地域（如 cn-beijing）
  S3_BUCKET: "swanlab-backup-demo" # 必填：存储桶名称
  S3_ENDPOINT: "tos-s3-cn-beijing.volces.com" # 必填：S3 格式的对象存储 Endpoint
  S3_PATH_PREFIX: "origin-backup-datas"
  # 选填：原始集群本机数据目录路径（根据实际部署填写）
  HOST_POSTGRES_PATH: "/var/lib/swanlab-postgres" # PostgreSQL 数据目录
  HOST_CLICKHOUSE_PATH: "/var/lib/swanlab-clickhouse" # ClickHouse 数据目录
  HOST_REDIS_PATH: "/var/lib/swanlab-redis" # Redis 数据目录
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-backup-storage-secret
  namespace: <target_namespace> # ⚠️ 必填：与 ConfigMap 相同的【目标集群】命名空间
type: Opaque
stringData:
  S3_AK: "xxx" # 必填：对象存储 AccessKey
  S3_SK: "xxx" # 必填：对象存储 SecretKey
```

:::

::: code-group

```bash [源集群]
kubectl apply -f config-export.yaml
```

```bash [目标集群]
kubectl apply -f config-import.yaml
```

:::

同时修改导出/导入 Job YAML 中的以下字段：

- `namespace`：对应的 K8s 命名空间
- `claimName`：对应的 PVC 名称
- `nodeSelector`：本机部署场景需指定节点

### 2. 停服

- 操作位置：<span style="color: red"><strong>源集群、目标集群</strong></span>

务必按照顺序停服。

::: code-group

```bash [1. 停网关]
# 切断所有外部流量
kubectl scale deployment swanlab-self-hosted --replicas=0 -n <your_namespace>
```

```bash [2. 停应用层]
# 停后端核心服务
kubectl scale deployment swanlab-self-hosted-server --replicas=0 -n <your_namespace>
# 停后端指标OLAP服务
kubectl scale deployment swanlab-self-hosted-house --replicas=0 -n <your_namespace>
```

```bash [3. 停 Vector]
# 先等缓冲区消费完（看 logs 无新写入后 Ctrl+C）
kubectl logs -f swanlab-self-hosted-vector-0 -n <your_namespace> --tail=20
kubectl logs -f swanlab-self-hosted-vector-1 -n <your_namespace> --tail=20

# 停 Vector
kubectl scale statefulset swanlab-self-hosted-vector --replicas=0 -n <your_namespace>
```

```bash [4. 停数据库]
kubectl scale deployment swanlab-self-hosted-postgres --replicas=0 -n <your_namespace>
kubectl scale deployment swanlab-self-hosted-clickhouse --replicas=0 -n <your_namespace>
kubectl scale deployment swanlab-self-hosted-redis --replicas=0 -n <your_namespace>
```

```bash [「可选」5. 停 S3]
# 「可选」通常外接 S3 对象存储可忽略，如您使用 template 自集成的 MinIO 则需要停服
kubectl scale deployment swanlab-self-hosted-s3 --replicas=0 -n <your_namespace>
```

:::

### 3. 导出 DB 数据

- 操作位置：<span style="color: red"><strong>源集群</strong></span>

每个数据库的迁移被封装为独立的 Job，可并行执行。根据对象存储类型选择 `s3cmd` 或 `ossutil` 版本：

::: details export-postgres

::: code-group

```yaml [s3cmd]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-postgres
  namespace: <source_namespace> # ⚠️ 必填：【原始集群】K8s 命名空间
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== 本机部署场景：指定节点（跨集群 K8s 场景删除此段） =====
      # nodeSelector:
      #   kubernetes.io/hostname: <node_name>
      # tolerations:
      #   - operator: Exists
      # # ============================================================
      containers:
        - name: swanlab-export-postgres
          image: repo.swanlab.cn/public/alpine:3.19
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - secretRef:
                name: swanlab-backup-storage-secret
          command:
            - /bin/sh
            - -c
            - |
              set -e
              echo "[1/5] 配置 apk 镜像源..."
              sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
              echo "[2/5] 安装 s3cmd..."
              apk add --no-cache s3cmd

              echo "[3/5] 写入 s3cmd 配置..."
              cat > ~/.s3cfg << EOF
              [default]
              access_key = ${S3_AK}
              secret_key = ${S3_SK}
              host_base = ${S3_ENDPOINT}
              host_bucket = %(bucket)s.${S3_ENDPOINT}
              disable_path_style = True
              signature_v2 = False
              use_https = True
              encoding = utf-8
              socket_timeout = 300
              EOF

              echo "[4/5] 打包 PostgreSQL 数据..."
              tar -czf /tmp/postgres-data.tar.gz -C /mnt/postgres .
              sha256sum /tmp/postgres-data.tar.gz > /tmp/postgres-data.tar.gz.sha256

              echo "[5/5] 上传至对象存储..."
              s3cmd put /tmp/postgres-data.tar.gz s3://${S3_BUCKET}/${S3_PATH_PREFIX}/
              s3cmd put /tmp/postgres-data.tar.gz.sha256 s3://${S3_BUCKET}/${S3_PATH_PREFIX}/
              echo "--- PostgreSQL 备份完成 ---"
          volumeMounts:
            - name: swanlab-pg-data
              mountPath: /mnt/postgres
      volumes:
        - name: swanlab-pg-data
          persistentVolumeClaim:
            claimName: swanlab-postgres-pvc # ⚠️ 必填：【原始集群】K8s 命名空间下的 postgres pvc 名称
```

```yaml [ossutil]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-postgres
  namespace: <source_namespace> # ⚠️ 必填：【原始集群】K8s 命名空间
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== 本机部署场景：指定节点（跨集群 K8s 场景删除此段） =====
      # nodeSelector:
      #   kubernetes.io/hostname: <node_name>
      # tolerations:
      #   - operator: Exists
      # # ============================================================
      containers:
        - name: swanlab-export-postgres
          image: repo.swanlab.cn/public/alpine:3.19
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - secretRef:
                name: swanlab-backup-storage-secret
          command:
            - /bin/sh
            - -c
            - |
              set -e
              echo "[1/5] 安装 ossutil..."
              wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64.zip -O /tmp/ossutil.zip
              unzip -q /tmp/ossutil.zip -d /tmp/ossutil-dir
              mv /tmp/ossutil-dir/ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil
              chmod +x /usr/local/bin/ossutil

              echo "[2/5] 配置 ossutil..."
              ossutil config -e ${S3_ENDPOINT} -i ${S3_AK} -k ${S3_SK}

              echo "[3/5] 打包 PostgreSQL 数据..."
              tar -czf /tmp/postgres-data.tar.gz -C /mnt/postgres .
              sha256sum /tmp/postgres-data.tar.gz > /tmp/postgres-data.tar.gz.sha256

              echo "[4/5] 上传至对象存储..."
              ossutil cp /tmp/postgres-data.tar.gz oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f
              ossutil cp /tmp/postgres-data.tar.gz.sha256 oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f

              echo "[5/5] 记录权限信息..."
              ls -ln /mnt/postgres > /tmp/postgres-perm_info.txt
              ossutil cp /tmp/postgres-perm_info.txt oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f

              echo "--- PostgreSQL 备份完成 ---"
          volumeMounts:
            - name: swanlab-pg-data
              mountPath: /mnt/postgres
      volumes:
        - name: swanlab-pg-data
          persistentVolumeClaim:
            claimName: swanlab-postgres-pvc # ⚠️ 必填：【原始集群】K8s 命名空间下的 postgres pvc 名称
```

:::

::: details export-clickhouse

::: code-group

```yaml [s3cmd]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-clickhouse
  namespace: <source_namespace> # ⚠️ 必填：【原始集群】K8s 命名空间
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== 本机部署场景：指定节点（跨集群 K8s 场景删除此段） =====
      # nodeSelector:
      #   kubernetes.io/hostname: <node_name>
      # tolerations:
      #   - operator: Exists
      # # ============================================================
      containers:
        - name: swanlab-export-clickhouse
          image: repo.swanlab.cn/public/alpine:3.19
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - secretRef:
                name: swanlab-backup-storage-secret
          command: ["/bin/sh", "-c"]
          args:
            - |
              set -e
              echo "[1/5] 配置 apk 镜像源..."
              sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
              echo "[2/5] 安装 s3cmd..."
              apk add --no-cache s3cmd

              echo "[3/5] 写入 s3cmd 配置..."
              cat > ~/.s3cfg << EOF
              [default]
              access_key = ${S3_AK}
              secret_key = ${S3_SK}
              host_base = ${S3_ENDPOINT}
              host_bucket = %(bucket)s.${S3_ENDPOINT}
              disable_path_style = True
              signature_v2 = False
              use_https = True
              encoding = utf-8
              socket_timeout = 300
              EOF

              echo "[4/5] 打包 ClickHouse 数据..."
              tar -czf /tmp/clickhouse-data.tar.gz -C /mnt/clickhouse .
              sha256sum /tmp/clickhouse-data.tar.gz > /tmp/clickhouse-data.tar.gz.sha256

              echo "[5/5] 上传至对象存储..."
              s3cmd put /tmp/clickhouse-data.tar.gz s3://${S3_BUCKET}/${S3_PATH_PREFIX}/
              s3cmd put /tmp/clickhouse-data.tar.gz.sha256 s3://${S3_BUCKET}/${S3_PATH_PREFIX}/
              echo "--- ClickHouse 备份完成 ---"
          volumeMounts:
            - name: swanlab-ch-data
              mountPath: /mnt/clickhouse
      volumes:
        - name: swanlab-ch-data
          persistentVolumeClaim:
            claimName: swanlab-clickhouse-pvc # ⚠️ 必填：【原始集群】K8s 命名空间下的 clickhouse pvc 名称
```

```yaml [ossutil]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-clickhouse
  namespace: <source_namespace> # ⚠️ 必填：【原始集群】K8s 命名空间
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== 本机部署场景：指定节点（跨集群 K8s 场景删除此段） =====
      # nodeSelector:
      #   kubernetes.io/hostname: <node_name>
      # tolerations:
      #   - operator: Exists
      # # ============================================================
      containers:
        - name: swanlab-export-clickhouse
          image: repo.swanlab.cn/public/alpine:3.19
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - secretRef:
                name: swanlab-backup-storage-secret
          command: ["/bin/sh", "-c"]
          args:
            - |
              set -e
              echo "[1/5] 安装 ossutil..."
              wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64.zip -O /tmp/ossutil.zip
              unzip -q /tmp/ossutil.zip -d /tmp/ossutil-dir
              mv /tmp/ossutil-dir/ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil
              chmod +x /usr/local/bin/ossutil

              echo "[2/5] 配置 ossutil..."
              ossutil config -e ${S3_ENDPOINT} -i ${S3_AK} -k ${S3_SK}

              echo "[3/5] 打包 ClickHouse 数据..."
              tar -czf /tmp/clickhouse-data.tar.gz -C /mnt/clickhouse .
              sha256sum /tmp/clickhouse-data.tar.gz > /tmp/clickhouse-data.tar.gz.sha256

              echo "[4/5] 上传至对象存储..."
              ossutil cp /tmp/clickhouse-data.tar.gz oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f
              ossutil cp /tmp/clickhouse-data.tar.gz.sha256 oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f

              echo "[5/5] 记录权限信息..."
              ls -ln /mnt/clickhouse > /tmp/clickhouse-perm_info.txt
              ossutil cp /tmp/clickhouse-perm_info.txt oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f

              echo "--- ClickHouse 备份完成 ---"
          volumeMounts:
            - name: swanlab-ch-data
              mountPath: /mnt/clickhouse
      volumes:
        - name: swanlab-ch-data
          persistentVolumeClaim:
            claimName: swanlab-clickhouse-pvc # ⚠️ 必填：【原始集群】K8s 命名空间下的 clickhouse pvc 名称
```

:::

::: details export-redis

::: code-group

```yaml [s3cmd]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-redis
  namespace: <source_namespace> # ⚠️ 必填：【原始集群】K8s 命名空间
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== 本机部署场景：指定节点（跨集群 K8s 场景删除此段） =====
      # nodeSelector:
      #   kubernetes.io/hostname: <node_name>
      # tolerations:
      #   - operator: Exists
      # # ============================================================
      containers:
        - name: swanlab-export-redis
          image: repo.swanlab.cn/public/alpine:3.19
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - secretRef:
                name: swanlab-backup-storage-secret
          command: ["/bin/sh", "-c"]
          args:
            - |
              set -e
              echo "[1/5] 配置 apk 镜像源..."
              sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
              echo "[2/5] 安装 s3cmd..."
              apk add --no-cache s3cmd

              echo "[3/5] 写入 s3cmd 配置..."
              cat > ~/.s3cfg << EOF
              [default]
              access_key = ${S3_AK}
              secret_key = ${S3_SK}
              host_base = ${S3_ENDPOINT}
              host_bucket = %(bucket)s.${S3_ENDPOINT}
              disable_path_style = True
              signature_v2 = False
              use_https = True
              encoding = utf-8
              socket_timeout = 300
              EOF

              echo "[4/5] 打包 Redis 数据..."
              tar -czf /tmp/redis-data.tar.gz -C /mnt/redis .
              sha256sum /tmp/redis-data.tar.gz > /tmp/redis-data.tar.gz.sha256

              echo "[5/5] 上传至对象存储..."
              s3cmd put /tmp/redis-data.tar.gz s3://${S3_BUCKET}/${S3_PATH_PREFIX}/
              s3cmd put /tmp/redis-data.tar.gz.sha256 s3://${S3_BUCKET}/${S3_PATH_PREFIX}/
              echo "--- Redis 备份完成 ---"
          volumeMounts:
            - name: swanlab-redis-data
              mountPath: /mnt/redis
      volumes:
        - name: swanlab-redis-data
          persistentVolumeClaim:
            claimName: swanlab-redis-pvc # ⚠️ 必填：【原始集群】K8s 命名空间下的 redis pvc 名称
```

```yaml [ossutil]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-redis
  namespace: <source_namespace> # ⚠️ 必填：【原始集群】K8s 命名空间
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== 本机部署场景：指定节点（跨集群 K8s 场景删除此段） =====
      # nodeSelector:
      #   kubernetes.io/hostname: <node_name>
      # tolerations:
      #   - operator: Exists
      # # ============================================================
      containers:
        - name: swanlab-export-redis
          image: repo.swanlab.cn/public/alpine:3.19
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - secretRef:
                name: swanlab-backup-storage-secret
          command: ["/bin/sh", "-c"]
          args:
            - |
              set -e
              echo "[1/5] 安装 ossutil..."
              wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64.zip -O /tmp/ossutil.zip
              unzip -q /tmp/ossutil.zip -d /tmp/ossutil-dir
              mv /tmp/ossutil-dir/ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil
              chmod +x /usr/local/bin/ossutil

              echo "[2/5] 配置 ossutil..."
              ossutil config -e ${S3_ENDPOINT} -i ${S3_AK} -k ${S3_SK}

              echo "[3/5] 打包 Redis 数据..."
              tar -czf /tmp/redis-data.tar.gz -C /mnt/redis .
              sha256sum /tmp/redis-data.tar.gz > /tmp/redis-data.tar.gz.sha256

              echo "[4/5] 上传至对象存储..."
              ossutil cp /tmp/redis-data.tar.gz oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f
              ossutil cp /tmp/redis-data.tar.gz.sha256 oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f

              echo "[5/5] 记录权限信息..."
              ls -ln /mnt/redis > /tmp/redis-perm_info.txt
              ossutil cp /tmp/redis-perm_info.txt oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f

              echo "--- Redis 备份完成 ---"
          volumeMounts:
            - name: swanlab-redis-data
              mountPath: /mnt/redis
      volumes:
        - name: swanlab-redis-data
          persistentVolumeClaim:
            claimName: swanlab-redis-pvc # ⚠️ 必填：【原始集群】K8s 命名空间下的 redis pvc 名称
```

:::

::: details export-vector

::: code-group

```yaml [s3cmd]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-vector
  namespace: <source_namespace> # ⚠️ 必填：【原始集群】K8s 命名空间
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== 本机部署场景：指定节点（跨集群 K8s 场景删除此段） =====
      # nodeSelector:
      #   kubernetes.io/hostname: <node_name>
      # tolerations:
      #   - operator: Exists
      # # ============================================================
      containers:
        - name: swanlab-export-vector
          image: repo.swanlab.cn/public/alpine:3.19
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - secretRef:
                name: swanlab-backup-storage-secret
          command:
            - /bin/sh
            - -c
            - |
              set -e
              echo "[1/5] 配置 apk 镜像源..."
              sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
              echo "[2/5] 安装 s3cmd..."
              apk add --no-cache s3cmd

              echo "[3/5] 写入 s3cmd 配置..."
              cat > ~/.s3cfg << EOF
              [default]
              access_key = ${S3_AK}
              secret_key = ${S3_SK}
              host_base = ${S3_ENDPOINT}
              host_bucket = %(bucket)s.${S3_ENDPOINT}
              disable_path_style = True
              signature_v2 = False
              use_https = True
              encoding = utf-8
              socket_timeout = 300
              EOF

              echo "[4/5] 打包 Vector 数据..."
              tar -czf /tmp/vector-data.tar.gz -C /mnt vector-0 vector-1
              sha256sum /tmp/vector-data.tar.gz > /tmp/vector-data.tar.gz.sha256

              echo "[5/5] 上传至对象存储..."
              s3cmd put /tmp/vector-data.tar.gz s3://${S3_BUCKET}/${S3_PATH_PREFIX}/
              s3cmd put /tmp/vector-data.tar.gz.sha256 s3://${S3_BUCKET}/${S3_PATH_PREFIX}/
              echo "--- Vector 备份完成 ---"
          volumeMounts:
            - name: vector-data-0
              mountPath: /mnt/vector-0
            - name: vector-data-1
              mountPath: /mnt/vector-1
      volumes:
        - name: vector-data-0
          persistentVolumeClaim:
            claimName: data-swanlab-self-hosted-vector-0 # ⚠️ 必填：【原始集群】K8s 命名空间下的 vector-0 pvc 名称
        - name: vector-data-1
          persistentVolumeClaim:
            claimName: data-swanlab-self-hosted-vector-1 # ⚠️ 必填：【原始集群】K8s 命名空间下的 vector-1 pvc 名称
```

```yaml [ossutil]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-vector
  namespace: <source_namespace> # ⚠️ 必填：【原始集群】K8s 命名空间
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== 本机部署场景：指定节点（跨集群 K8s 场景删除此段） =====
      # nodeSelector:
      #   kubernetes.io/hostname: <node_name>
      # tolerations:
      #   - operator: Exists
      # # ============================================================
      containers:
        - name: swanlab-export-vector
          image: repo.swanlab.cn/public/alpine:3.19
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - secretRef:
                name: swanlab-backup-storage-secret
          command:
            - /bin/sh
            - -c
            - |
              set -e
              echo "[1/5] 安装 ossutil..."
              wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64.zip -O /tmp/ossutil.zip
              unzip -q /tmp/ossutil.zip -d /tmp/ossutil-dir
              mv /tmp/ossutil-dir/ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil
              chmod +x /usr/local/bin/ossutil

              echo "[2/5] 配置 ossutil..."
              ossutil config -e ${S3_ENDPOINT} -i ${S3_AK} -k ${S3_SK}

              echo "[3/5] 打包 Vector 数据..."
              tar -czf /tmp/vector-data.tar.gz -C /mnt vector-0 vector-1
              sha256sum /tmp/vector-data.tar.gz > /tmp/vector-data.tar.gz.sha256

              echo "[4/5] 上传至对象存储..."
              ossutil cp /tmp/vector-data.tar.gz oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f
              ossutil cp /tmp/vector-data.tar.gz.sha256 oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f

              echo "[5/5] 记录权限信息..."
              echo "=== vector-0 ===" > /tmp/vector-perm_info.txt
              ls -ln /mnt/vector-0 >> /tmp/vector-perm_info.txt
              echo "=== vector-1 ===" >> /tmp/vector-perm_info.txt
              ls -ln /mnt/vector-1 >> /tmp/vector-perm_info.txt
              ossutil cp /tmp/vector-perm_info.txt oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f

              echo "--- Vector 备份完成 ---"
          volumeMounts:
            - name: vector-data-0
              mountPath: /mnt/vector-0
            - name: vector-data-1
              mountPath: /mnt/vector-1
      volumes:
        - name: vector-data-0
          persistentVolumeClaim:
            claimName: data-swanlab-self-hosted-vector-0 # ⚠️ 必填：【原始集群】K8s 命名空间下的 vector-0 pvc 名称
        - name: vector-data-1
          persistentVolumeClaim:
            claimName: data-swanlab-self-hosted-vector-1 # ⚠️ 必填：【原始集群】K8s 命名空间下的 vector-1 pvc 名称
```

:::

```bash
# 并行执行所有导出 Job
kubectl apply -f export/

# 查看执行状态
kubectl logs -f job/swanlab-export-postgres -n <your_namespace>
kubectl logs -f job/swanlab-export-clickhouse -n <your_namespace>
kubectl logs -f job/swanlab-export-redis -n <your_namespace>

# 确认所有 Job 完成
kubectl get jobs -n <your_namespace>
```

### 4. 导出 S3 数据（可选）

- 操作位置：<span style="color: red"><strong>源集群</strong></span>

#### 情况 1：原始集群已集成 S3 URL

如果原本已经挂载好 S3 接入点，只需配置源集群 `value.yaml` 中相同的 S3接入点配置，详见 [外部 S3 集成配置](./configuration.md#外部-s3-集成-integrations-s3)。

#### 情况 2：原始集群使用 MiniO 挂载 PVC

MiniO 采用分片存储，一般不可直接上传到对象存储，需要 `rclone` 处理后同步到公有云对象存储：

::: details export-s3-pod YAML

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: export-s3-pod
  namespace: <your_namespace>
spec:
  restartPolicy: Never
  volumes:
    - name: s3-data
      persistentVolumeClaim:
        claimName: <your_s3_pvc>
  containers:
    # 容器 1：本地 MinIO（解析原始物理文件）
    - name: local-minio
      image: minio/minio:latest
      env:
        - name: MINIO_ROOT_USER
          value: "<original_minio_ak>"
        - name: MINIO_ROOT_PASSWORD
          value: "<original_minio_sk>"
      command: ["/bin/sh", "-c"]
      args: ["minio server /mnt/s3 --address :9000"]
      volumeMounts:
        - name: s3-data
          mountPath: /mnt/s3

    # 容器 2：Rclone 迁移工具
    - name: rclone-worker
      image: rclone/rclone:latest
      env:
        - name: MINIO_AK
          value: "<original_minio_ak>"
        - name: MINIO_SK
          value: "<original_minio_sk>"
        - name: S3_AK
          value: "<your_cloud_ak>"
        - name: S3_SK
          value: "<your_cloud_sk>"
        - name: S3_ENDPOINT
          value: "<your_cloud_endpoint>"
        - name: S3_PUBLIC_BUCKET
          value: "<your_public_bucket>"
        - name: S3_PRIVATE_BUCKET
          value: "<your_private_bucket>"
      command: ["/bin/sh", "-c"]
      args:
        - |
          cat <<EOF > /tmp/rclone.conf
          [local_minio]
          type = s3
          provider = Minio
          access_key_id = ${MINIO_AK}
          secret_access_key = ${MINIO_SK}
          endpoint = http://127.0.0.1:9000

          [remote_s3]
          type = s3
          provider = Auto
          access_key_id = ${S3_AK}
          secret_access_key = ${S3_SK}
          endpoint = ${S3_ENDPOINT}
          EOF
          sleep 10
          rclone sync local_minio:swanlab-public remote_s3:${S3_PUBLIC_BUCKET} --config /tmp/rclone.conf -v
          rclone sync local_minio:swanlab-private remote_s3:${S3_PRIVATE_BUCKET} --config /tmp/rclone.conf -v
          sleep 864000
```

:::

```bash
kubectl apply -f export-s3-pod.yaml -n <your_namespace>

# 查看 rclone 日志
kubectl logs -f export-s3-pod -c rclone-worker
```

### 5. 导入 DB 数据

- 操作位置：<span style="color: red"><strong>目标集群</strong></span>

与导出类似，每个数据库有独立的导入 Job，可并行执行。根据对象存储类型选择 `s3cmd` 或 `ossutil` 版本：

::: details import-postgres

::: code-group

```yaml [s3cmd]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-postgres
  namespace: <target_namespace> # ⚠️ 必填：【目标集群】K8s 命名空间
  labels:
    swanlab: postgres
spec:
  backoffLimit: 3
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: swanlab-import-postgres
          image: repo.swanlab.cn/public/alpine:3.19
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - secretRef:
                name: swanlab-backup-storage-secret
          volumeMounts:
            - name: swanlab-pg-data
              mountPath: /data
          command:
            - /bin/sh
            - -c
            - |
              set -e
              echo "[1/6] 配置 apk 镜像源..."
              sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
              echo "[2/6] 安装 s3cmd..."
              apk add --no-cache s3cmd

              echo "[3/6] 写入 s3cmd 配置..."
              cat > ~/.s3cfg << EOF
              [default]
              access_key = ${S3_AK}
              secret_key = ${S3_SK}
              host_base = ${S3_ENDPOINT}
              host_bucket = %(bucket)s.${S3_ENDPOINT}
              disable_path_style = True
              signature_v2 = No
              use_https = True
              encoding = utf-8
              socket_timeout = 300
              EOF

              echo "[4/6] 验证 Bucket 访问..."
              s3cmd ls s3://${S3_BUCKET}/${S3_PATH_PREFIX}/

              echo "[5/6] 下载备份文件..."
              s3cmd get s3://${S3_BUCKET}/${S3_PATH_PREFIX}/postgres-data.tar.gz /tmp/postgres-data.tar.gz
              s3cmd get s3://${S3_BUCKET}/${S3_PATH_PREFIX}/postgres-data.tar.gz.sha256 /tmp/postgres-data.tar.gz.sha256

              echo "[6/6] 校验并解压..."
              cd /tmp && sha256sum -c postgres-data.tar.gz.sha256
              tar -xzf /tmp/postgres-data.tar.gz -C /data
              ls -lh /data
      volumes:
        - name: swanlab-pg-data
          persistentVolumeClaim:
            claimName: swanlab-postgres-pvc # ⚠️ 必填：【目标集群】K8s 命名空间下的 postgres pvc 名称
```

```yaml [ossutil]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-postgres
  namespace: <target_namespace> # ⚠️ 必填：【目标集群】K8s 命名空间
  labels:
    swanlab: postgres
spec:
  backoffLimit: 3
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: swanlab-import-postgres
          image: repo.swanlab.cn/public/alpine:3.19
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - secretRef:
                name: swanlab-backup-storage-secret
          volumeMounts:
            - name: swanlab-pg-data
              mountPath: /data
          command:
            - /bin/sh
            - -c
            - |
              set -e
              echo "[1/5] 安装 ossutil..."
              wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64.zip -O /tmp/ossutil.zip
              unzip -q /tmp/ossutil.zip -d /tmp/ossutil-dir
              mv /tmp/ossutil-dir/ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil
              chmod +x /usr/local/bin/ossutil

              echo "[2/5] 配置 ossutil..."
              ossutil config -e ${S3_ENDPOINT} -i ${S3_AK} -k ${S3_SK}

              echo "[3/5] 下载备份文件..."
              ossutil cp oss://${S3_BUCKET}/${S3_PATH_PREFIX}/postgres-data.tar.gz /tmp/postgres-data.tar.gz -f
              ossutil cp oss://${S3_BUCKET}/${S3_PATH_PREFIX}/postgres-data.tar.gz.sha256 /tmp/postgres-data.tar.gz.sha256 -f

              echo "[4/5] 校验并解压..."
              cd /tmp && sha256sum -c postgres-data.tar.gz.sha256
              tar -xzf /tmp/postgres-data.tar.gz -C /data
              ls -lh /data

              echo "--- PostgreSQL 恢复完成 ---"
      volumes:
        - name: swanlab-pg-data
          persistentVolumeClaim:
            claimName: swanlab-postgres-pvc # ⚠️ 必填：【目标集群】K8s 命名空间下的 postgres pvc 名称
```

:::

::: details import-clickhouse

::: code-group

```yaml [s3cmd]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-clickhouse
  namespace: <target_namespace> # ⚠️ 必填：【目标集群】K8s 命名空间
  labels:
    swanlab: clickhouse
spec:
  backoffLimit: 3
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: swanlab-import-clickhouse
          image: repo.swanlab.cn/public/alpine:3.19
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - secretRef:
                name: swanlab-backup-storage-secret
          volumeMounts:
            - name: swanlab-ch-data
              mountPath: /data
          command:
            - /bin/sh
            - -c
            - |
              set -e
              echo "[1/6] 配置 apk 镜像源..."
              sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
              echo "[2/6] 安装 s3cmd..."
              apk add --no-cache s3cmd

              echo "[3/6] 写入 s3cmd 配置..."
              cat > ~/.s3cfg << EOF
              [default]
              access_key = ${S3_AK}
              secret_key = ${S3_SK}
              host_base = ${S3_ENDPOINT}
              host_bucket = %(bucket)s.${S3_ENDPOINT}
              disable_path_style = True
              signature_v2 = No
              use_https = True
              encoding = utf-8
              socket_timeout = 300
              EOF

              echo "[4/6] 验证 Bucket 访问..."
              s3cmd ls s3://${S3_BUCKET}/${S3_PATH_PREFIX}/

              echo "[5/6] 下载备份文件..."
              s3cmd get s3://${S3_BUCKET}/${S3_PATH_PREFIX}/clickhouse-data.tar.gz /tmp/clickhouse-data.tar.gz
              s3cmd get s3://${S3_BUCKET}/${S3_PATH_PREFIX}/clickhouse-data.tar.gz.sha256 /tmp/clickhouse-data.tar.gz.sha256

              echo "[6/6] 校验并解压..."
              cd /tmp && sha256sum -c clickhouse-data.tar.gz.sha256
              tar -xzf /tmp/clickhouse-data.tar.gz -C /data
              ls -lh /data
      volumes:
        - name: swanlab-ch-data
          persistentVolumeClaim:
            claimName: swanlab-clickhouse-pvc # ⚠️ 必填：【目标集群】K8s 命名空间下的 clickhouse pvc 名称
```

```yaml [ossutil]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-clickhouse
  namespace: <target_namespace> # ⚠️ 必填：【目标集群】K8s 命名空间
  labels:
    swanlab: clickhouse
spec:
  backoffLimit: 3
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: swanlab-import-clickhouse
          image: repo.swanlab.cn/public/alpine:3.19
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - secretRef:
                name: swanlab-backup-storage-secret
          volumeMounts:
            - name: swanlab-ch-data
              mountPath: /data
          command:
            - /bin/sh
            - -c
            - |
              set -e
              echo "[1/5] 安装 ossutil..."
              wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64.zip -O /tmp/ossutil.zip
              unzip -q /tmp/ossutil.zip -d /tmp/ossutil-dir
              mv /tmp/ossutil-dir/ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil
              chmod +x /usr/local/bin/ossutil

              echo "[2/5] 配置 ossutil..."
              ossutil config -e ${S3_ENDPOINT} -i ${S3_AK} -k ${S3_SK}

              echo "[3/5] 下载备份文件..."
              ossutil cp oss://${S3_BUCKET}/${S3_PATH_PREFIX}/clickhouse-data.tar.gz /tmp/clickhouse-data.tar.gz -f
              ossutil cp oss://${S3_BUCKET}/${S3_PATH_PREFIX}/clickhouse-data.tar.gz.sha256 /tmp/clickhouse-data.tar.gz.sha256 -f

              echo "[4/5] 校验并解压..."
              cd /tmp && sha256sum -c clickhouse-data.tar.gz.sha256
              tar -xzf /tmp/clickhouse-data.tar.gz -C /data
              ls -lh /data

              echo "--- ClickHouse 恢复完成 ---"
      volumes:
        - name: swanlab-ch-data
          persistentVolumeClaim:
            claimName: swanlab-clickhouse-pvc # ⚠️ 必填：【目标集群】K8s 命名空间下的 clickhouse pvc 名称
```

:::

::: details import-redis

::: code-group

```yaml [s3cmd]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-redis
  namespace: <target_namespace> # ⚠️ 必填：【目标集群】K8s 命名空间
  labels:
    swanlab: redis
spec:
  backoffLimit: 3
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: swanlab-import-redis
          image: repo.swanlab.cn/public/alpine:3.19
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - secretRef:
                name: swanlab-backup-storage-secret
          volumeMounts:
            - name: swanlab-redis-data
              mountPath: /data
          command:
            - /bin/sh
            - -c
            - |
              set -e
              echo "[1/6] 配置 apk 镜像源..."
              sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
              echo "[2/6] 安装 s3cmd..."
              apk add --no-cache s3cmd

              echo "[3/6] 写入 s3cmd 配置..."
              cat > ~/.s3cfg << EOF
              [default]
              access_key = ${S3_AK}
              secret_key = ${S3_SK}
              host_base = ${S3_ENDPOINT}
              host_bucket = %(bucket)s.${S3_ENDPOINT}
              disable_path_style = True
              signature_v2 = No
              use_https = True
              encoding = utf-8
              socket_timeout = 300
              EOF

              echo "[4/6] 验证 Bucket 访问..."
              s3cmd ls s3://${S3_BUCKET}/${S3_PATH_PREFIX}/

              echo "[5/6] 下载备份文件..."
              s3cmd get s3://${S3_BUCKET}/${S3_PATH_PREFIX}/redis-data.tar.gz /tmp/redis-data.tar.gz
              s3cmd get s3://${S3_BUCKET}/${S3_PATH_PREFIX}/redis-data.tar.gz.sha256 /tmp/redis-data.tar.gz.sha256

              echo "[6/6] 校验并解压..."
              cd /tmp && sha256sum -c redis-data.tar.gz.sha256
              tar -xzf /tmp/redis-data.tar.gz -C /data
              ls -lh /data
      volumes:
        - name: swanlab-redis-data
          persistentVolumeClaim:
            claimName: swanlab-redis-pvc # ⚠️ 必填：【目标集群】K8s 命名空间下的 redis pvc 名称
```

```yaml [ossutil]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-redis
  namespace: <target_namespace> # ⚠️ 必填：【目标集群】K8s 命名空间
  labels:
    swanlab: redis
spec:
  backoffLimit: 3
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: swanlab-import-redis
          image: repo.swanlab.cn/public/alpine:3.19
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - secretRef:
                name: swanlab-backup-storage-secret
          volumeMounts:
            - name: swanlab-redis-data
              mountPath: /data
          command:
            - /bin/sh
            - -c
            - |
              set -e
              echo "[1/5] 安装 ossutil..."
              wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64.zip -O /tmp/ossutil.zip
              unzip -q /tmp/ossutil.zip -d /tmp/ossutil-dir
              mv /tmp/ossutil-dir/ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil
              chmod +x /usr/local/bin/ossutil

              echo "[2/5] 配置 ossutil..."
              ossutil config -e ${S3_ENDPOINT} -i ${S3_AK} -k ${S3_SK}

              echo "[3/5] 下载备份文件..."
              ossutil cp oss://${S3_BUCKET}/${S3_PATH_PREFIX}/redis-data.tar.gz /tmp/redis-data.tar.gz -f
              ossutil cp oss://${S3_BUCKET}/${S3_PATH_PREFIX}/redis-data.tar.gz.sha256 /tmp/redis-data.tar.gz.sha256 -f

              echo "[4/5] 校验并解压..."
              cd /tmp && sha256sum -c redis-data.tar.gz.sha256
              tar -xzf /tmp/redis-data.tar.gz -C /data
              ls -lh /data

              echo "--- Redis 恢复完成 ---"
      volumes:
        - name: swanlab-redis-data
          persistentVolumeClaim:
            claimName: swanlab-redis-pvc # ⚠️ 必填：【目标集群】K8s 命名空间下的 redis pvc 名称
```

:::

::: details import-vector

::: code-group

```yaml [s3cmd]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-vector
  namespace: <target_namespace> # ⚠️ 必填：【目标集群】K8s 命名空间
  labels:
    swanlab: vector
spec:
  backoffLimit: 3
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: swanlab-import-vector
          image: repo.swanlab.cn/public/alpine:3.19
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - secretRef:
                name: swanlab-backup-storage-secret
          volumeMounts:
            - name: vector-data-0
              mountPath: /data/vector-0
            - name: vector-data-1
              mountPath: /data/vector-1
          command:
            - /bin/sh
            - -c
            - |
              set -e
              echo "[1/6] 配置 apk 镜像源..."
              sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
              echo "[2/6] 安装 s3cmd..."
              apk add --no-cache s3cmd

              echo "[3/6] 写入 s3cmd 配置..."
              cat > ~/.s3cfg << EOF
              [default]
              access_key = ${S3_AK}
              secret_key = ${S3_SK}
              host_base = ${S3_ENDPOINT}
              host_bucket = %(bucket)s.${S3_ENDPOINT}
              disable_path_style = True
              signature_v2 = No
              use_https = True
              encoding = utf-8
              socket_timeout = 300
              EOF

              echo "[4/6] 验证 Bucket 访问..."
              s3cmd ls s3://${S3_BUCKET}/${S3_PATH_PREFIX}/

              echo "[5/6] 下载备份文件..."
              s3cmd get s3://${S3_BUCKET}/${S3_PATH_PREFIX}/vector-data.tar.gz /tmp/vector-data.tar.gz
              s3cmd get s3://${S3_BUCKET}/${S3_PATH_PREFIX}/vector-data.tar.gz.sha256 /tmp/vector-data.tar.gz.sha256

              echo "[6/6] 校验并解压..."
              cd /tmp && sha256sum -c vector-data.tar.gz.sha256
              tar -xzf /tmp/vector-data.tar.gz -C /data
              ls -lh /data/vector-0
              ls -lh /data/vector-1
      volumes:
        - name: vector-data-0
          persistentVolumeClaim:
            claimName: data-swanlab-self-hosted-vector-0 # ⚠️ 必填：【目标集群】K8s 命名空间下的 vector-0 pvc 名称
        - name: vector-data-1
          persistentVolumeClaim:
            claimName: data-swanlab-self-hosted-vector-1 # ⚠️ 必填：【目标集群】K8s 命名空间下的 vector-1 pvc 名称
```

```yaml [ossutil]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-vector
  namespace: <target_namespace> # ⚠️ 必填：【目标集群】K8s 命名空间
  labels:
    swanlab: vector
spec:
  backoffLimit: 3
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: swanlab-import-vector
          image: repo.swanlab.cn/public/alpine:3.19
          imagePullPolicy: IfNotPresent
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - secretRef:
                name: swanlab-backup-storage-secret
          volumeMounts:
            - name: vector-data-0
              mountPath: /data/vector-0
            - name: vector-data-1
              mountPath: /data/vector-1
          command:
            - /bin/sh
            - -c
            - |
              set -e
              echo "[1/5] 安装 ossutil..."
              wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64.zip -O /tmp/ossutil.zip
              unzip -q /tmp/ossutil.zip -d /tmp/ossutil-dir
              mv /tmp/ossutil-dir/ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil
              chmod +x /usr/local/bin/ossutil

              echo "[2/5] 配置 ossutil..."
              ossutil config -e ${S3_ENDPOINT} -i ${S3_AK} -k ${S3_SK}

              echo "[3/5] 下载备份文件..."
              ossutil cp oss://${S3_BUCKET}/${S3_PATH_PREFIX}/vector-data.tar.gz /tmp/vector-data.tar.gz -f
              ossutil cp oss://${S3_BUCKET}/${S3_PATH_PREFIX}/vector-data.tar.gz.sha256 /tmp/vector-data.tar.gz.sha256 -f

              echo "[4/5] 校验并解压..."
              cd /tmp && sha256sum -c vector-data.tar.gz.sha256
              tar -xzf /tmp/vector-data.tar.gz -C /data
              ls -lh /data/vector-0
              ls -lh /data/vector-1

              echo "--- Vector 恢复完成 ---"
      volumes:
        - name: vector-data-0
          persistentVolumeClaim:
            claimName: data-swanlab-self-hosted-vector-0 # ⚠️ 必填：【目标集群】K8s 命名空间下的 vector-0 pvc 名称
        - name: vector-data-1
          persistentVolumeClaim:
            claimName: data-swanlab-self-hosted-vector-1 # ⚠️ 必填：【目标集群】K8s 命名空间下的 vector-1 pvc 名称
```

:::

```bash
# 并行执行所有导入 Job
kubectl apply -f import/

# 查看执行状态
kubectl logs -f job/swanlab-import-postgres -n <your_namespace>
kubectl logs -f job/swanlab-import-clickhouse -n <your_namespace>
kubectl logs -f job/swanlab-import-redis -n <your_namespace>

# 确认所有 Job 完成
kubectl get jobs -n <your_namespace>
```

### 6. 重新开服

- 操作位置：<span style="color: red"><strong>目标集群</strong></span>

务必按照顺序开服。

::: code-group

```bash [1. 恢复数据库]
# 恢复数据库服务 (replicas 必须为 1)
kubectl scale deployment swanlab-self-hosted-clickhouse --replicas=1 -n <your_namespace>
kubectl scale deployment swanlab-self-hosted-postgres --replicas=1 -n <your_namespace>
kubectl scale deployment swanlab-self-hosted-redis --replicas=1 -n <your_namespace>

# 确认数据库就绪
kubectl get pods -n <your_namespace> -w
```

```bash [2. 恢复 Vector]
# StatefulSet，双副本
kubectl scale statefulset swanlab-self-hosted-vector --replicas=2 -n <your_namespace>
```

```bash [3. 恢复应用层]
# 先恢复副本，再按需扩容
kubectl scale deployment swanlab-self-hosted-house --replicas=1 -n <your_namespace>
kubectl scale deployment swanlab-self-hosted-server --replicas=1 -n <your_namespace>
```

```bash [4. 恢复网关]
# 恢复网关
kubectl scale deployment swanlab-self-hosted --replicas=2 -n <your_namespace>
```

```bash [「可选」5. 恢复 S3]
# 「可选」如外接 S3 可忽略，如果使用 template 内置 MinIO 需要手动恢复 S3
kubectl scale deployment swanlab-self-hosted-s3 --replicas=1 -n <your_namespace>
```

:::

恢复后可以观测 pod 健康状况与线上服务验证数据恢复情况。

## 🧹 Job 清理

原始和目标集群的 Job 完成后 **24 小时自动清理**（`ttlSecondsAfterFinished: 86400`）。如需手动清理：

```bash
# 源集群
kubectl delete job swanlab-export-postgres swanlab-export-clickhouse swanlab-export-redis swanlab-export-vector -n <original_namespace>

# 目标集群
kubectl delete job swanlab-import-postgres swanlab-import-clickhouse swanlab-import-redis swanlab-import-vector -n <target_namespace>
```
