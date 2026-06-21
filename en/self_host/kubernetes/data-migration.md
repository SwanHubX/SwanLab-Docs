# Data Migration

> This page introduces the complete process of full data migration between different Kubernetes clusters for the SwanLab self-hosted version.

:::warning
If you are using an external cloud database scheme [Not Recommended], you can ignore the process in this migration document
:::

## Migration Process Diagram

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/cross-cluster-migration.drawio.svg"/>

Under the **local database deployment scheme** of the cluster, the process of cross-cluster data migration for the SwanLab self-hosted service database includes three core areas:

- **Source Cluster (Original)**: Data export side, containing databases and object storage.
- **Transit Storage (Transit)**: S3 object storage as a temporary transit station
- **Target Cluster (Current)**: Data receiving side, completing the restore.

| Data Type                                  | Migration Method              | Description                                                                                |
| ------------------------------------------ | ----------------------------- | ------------------------------------------------------------------------------------------ |
| Database (PostgreSQL / ClickHouse / Redis) | Export → S3 Transit → Import  | Physical migration                                                                         |
| Object Storage (MiniO / S3)                | Direct connection to cloud S3 | Storage-compute separation, no need to move, new cluster directly reads and writes via API |

**Legend Explanation**:

- 🔵 **Phase 1: Export** — Source cluster exports data to S3 transit storage bucket (DB Export + S3 Sync)
- 🟢 **Phase 2: Import** — Database data imported from S3 to new cluster (DB Import, physical migration)
- 🟢 **Phase 2: Direct Use** (dashed line) — Object storage is not moved, new cluster directly accesses public and private object storage buckets through the same `value.yaml` configuration

## 🧾 Prerequisites

### Resource Preparation

- S3 protocol compatible object storage bucket, available space must be at least **1.1 times** greater than the total storage of ClickHouse + PostgreSQL + Redis + MinIO (if applicable)
- Available storage resources in the original cluster can support the additional compressed package storage space for the above components
- The target cluster needs an additional deployment of a **brand new unactivated** SwanLab service, and must mount cloud disks
- If the original cluster and target cluster versions are inconsistent, it is recommended to first upgrade the chart version of the original cluster

### Permission Preparation

- Have write permissions for both the original cluster and target cluster (the original and target clusters can be the same cluster, but each must have an independent SwanLab service)
- Prepare `access_key` and `secret_key` with read-write permissions for the object storage bucket

### Configuration Variables

Please confirm the following information before use:

| Prepared | Placeholder Variable                    | Description                                                             |
| -------- | --------------------------------------- | ----------------------------------------------------------------------- |
| ✅       | `origin_namespace` / `target_namespace` | Namespaces of the original and target clusters                          |
| ✅       | `S3_REGION`                             | Region of the object storage bucket used for backup                     |
| ✅       | `S3_BUCKET`                             | Bucket name                                                             |
| ✅       | `S3_ENDPOINT`                           | S3 format object storage Endpoint, e.g., `tos-s3-cn-beijing.volces.com` |
| ✅       | `S3_AK` / `S3_SK`                       | Object storage writable credentials                                     |
| ✅       | `S3_PATH_PREFIX`                        | Backup path prefix in S3, default `origin-backup-datas`                 |
| ✅       | Original cluster component PVC names    | `kubectl get pvc -n <origin_namespace>`                                 |

> 💡 Through practice, it has been verified that you can directly pass PVCs and let K8s handle the mounting automatically.

> ⚠️ The S3 migration tool uses `s3cmd` by default, which has been verified on Tencent Cloud COS and Volcano Engine TOS. When using Alibaba Cloud OSS as a transit, you need to use `ossutil` and replace the transfer tool.

## 🪜 Operation Steps

:::warning

- During data migration, you need to ensure that **both SwanLab services are kept in a stopped state**, with a single intermediate Job performing the migration. Otherwise, migration will fail due to state inconsistency issues.
- **You must stop services before migration!**
  :::

### 1. Modify Configuration Files

- Operation location: <span style="color: red"><strong>Source Cluster, Target Cluster</strong></span>

Create S3 configuration (ConfigMap + Secret). You need to **make two copies**, **fill in the namespaces of the source and target clusters respectively, and execute each**:

::: details config-export.yaml / config-import.yaml

::: code-group

```yaml [config-export.yaml]
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-backup-storage-config
  namespace: <original_namespace> # ⚠️ Required: [Source Cluster] K8s namespace
data:
  S3_REGION: "cn-beijing" # Required: Object storage region for transit (e.g., cn-beijing)
  S3_BUCKET: "swanlab-backup-demo" # Required: Bucket name
  S3_ENDPOINT: "tos-s3-cn-beijing.volces.com" # Required: S3 format object storage Endpoint
  S3_PATH_PREFIX: "origin-backup-datas"
  # Optional: Local data directory path of the original cluster (fill in based on actual deployment)
  HOST_POSTGRES_PATH: "/var/lib/swanlab-postgres" # PostgreSQL data directory
  HOST_CLICKHOUSE_PATH: "/var/lib/swanlab-clickhouse" # ClickHouse data directory
  HOST_REDIS_PATH: "/var/lib/swanlab-redis" # Redis data directory
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-backup-storage-secret
  namespace: <original_namespace> # ⚠️ Required: [Source Cluster] K8s namespace
type: Opaque
stringData:
  S3_AK: "xxx" # Required: Object storage AccessKey
  S3_SK: "xxx" # Required: Object storage SecretKey
```

```yaml [config-import.yaml]
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-backup-storage-config
  namespace: <target_namespace> # ⚠️ Required: [Target Cluster] K8s namespace
data:
  S3_REGION: "cn-beijing" # Required: Object storage region for transit (e.g., cn-beijing)
  S3_BUCKET: "swanlab-backup-demo" # Required: Bucket name
  S3_ENDPOINT: "tos-s3-cn-beijing.volces.com" # Required: S3 format object storage Endpoint
  S3_PATH_PREFIX: "origin-backup-datas"
  # Optional: Local data directory path of the original cluster (fill in based on actual deployment)
  HOST_POSTGRES_PATH: "/var/lib/swanlab-postgres" # PostgreSQL data directory
  HOST_CLICKHOUSE_PATH: "/var/lib/swanlab-clickhouse" # ClickHouse data directory
  HOST_REDIS_PATH: "/var/lib/swanlab-redis" # Redis data directory
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-backup-storage-secret
  namespace: <target_namespace> # ⚠️ Required: Same [Target Cluster] namespace as ConfigMap
type: Opaque
stringData:
  S3_AK: "xxx" # Required: Object storage AccessKey
  S3_SK: "xxx" # Required: Object storage SecretKey
```

:::

::: code-group

```bash [Source Cluster]
kubectl apply -f config-export.yaml
```

```bash [Target Cluster]
kubectl apply -f config-import.yaml
```

:::

Also modify the following fields in the export/import Job YAML:

- `namespace`: Corresponding K8s namespace
- `claimName`: Corresponding PVC name
- `nodeSelector`: Needs to specify a node for local deployment scenarios

### 2. Stop Services

- Operation location: <span style="color: red"><strong>Source Cluster, Target Cluster</strong></span>

Be sure to stop services in order.

::: code-group

```bash [1. Stop Gateway]
# Cut off all external traffic
kubectl scale deployment swanlab-self-hosted --replicas=0 -n <your_namespace>
```

```bash [2. Stop Application Layer]
# Stop backend core service
kubectl scale deployment swanlab-self-hosted-server --replicas=0 -n <your_namespace>
# Stop backend metrics OLAP service
kubectl scale deployment swanlab-self-hosted-house --replicas=0 -n <your_namespace>
```

```bash [3. Stop Vector]
# Wait for the buffer to consume (watch logs for no new writes, then Ctrl+C)
kubectl logs -f swanlab-self-hosted-vector-0 -n <your_namespace> --tail=20
kubectl logs -f swanlab-self-hosted-vector-1 -n <your_namespace> --tail=20

# Stop Vector
kubectl scale statefulset swanlab-self-hosted-vector --replicas=0 -n <your_namespace>
```

```bash [4. Stop Databases]
kubectl scale deployment swanlab-self-hosted-postgres --replicas=0 -n <your_namespace>
kubectl scale deployment swanlab-self-hosted-clickhouse --replicas=0 -n <your_namespace>
kubectl scale deployment swanlab-self-hosted-redis --replicas=0 -n <your_namespace>
```

```bash [「Optional」5. Stop S3]
# 「Optional」Usually external S3 object storage can be ignored. If you use the template's built-in MinIO, you need to stop it
kubectl scale deployment swanlab-self-hosted-s3 --replicas=0 -n <your_namespace>
```

:::

### 3. Export DB Data

- Operation location: <span style="color: red"><strong>Source Cluster</strong></span>

Each database migration is encapsulated as an independent Job and can be executed in parallel. Choose `s3cmd` or `ossutil` version based on the object storage type:

::: details export-postgres

::: code-group

```yaml [s3cmd]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-postgres
  namespace: <source_namespace> # ⚠️ Required: [Source Cluster] K8s namespace
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== Local deployment scenario: specify node (delete this section for cross-cluster K8s) =====
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
              echo "[1/5] Configuring apk mirror source..."
              sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
              echo "[2/5] Installing s3cmd..."
              apk add --no-cache s3cmd

              echo "[3/5] Writing s3cmd configuration..."
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

              echo "[4/5] Packaging PostgreSQL data..."
              tar -czf /tmp/postgres-data.tar.gz -C /mnt/postgres .
              sha256sum /tmp/postgres-data.tar.gz > /tmp/postgres-data.tar.gz.sha256

              echo "[5/5] Uploading to object storage..."
              s3cmd put /tmp/postgres-data.tar.gz s3://${S3_BUCKET}/${S3_PATH_PREFIX}/
              s3cmd put /tmp/postgres-data.tar.gz.sha256 s3://${S3_BUCKET}/${S3_PATH_PREFIX}/
              echo "--- PostgreSQL backup complete ---"
          volumeMounts:
            - name: swanlab-pg-data
              mountPath: /mnt/postgres
      volumes:
        - name: swanlab-pg-data
          persistentVolumeClaim:
            claimName: swanlab-postgres-pvc # ⚠️ Required: postgres pvc name in [Source Cluster] K8s namespace
```

```yaml [ossutil]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-postgres
  namespace: <source_namespace> # ⚠️ Required: [Source Cluster] K8s namespace
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== Local deployment scenario: specify node (delete this section for cross-cluster K8s) =====
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
              echo "[1/5] Installing ossutil..."
              wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64.zip -O /tmp/ossutil.zip
              unzip -q /tmp/ossutil.zip -d /tmp/ossutil-dir
              mv /tmp/ossutil-dir/ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil
              chmod +x /usr/local/bin/ossutil

              echo "[2/5] Configuring ossutil..."
              ossutil config -e ${S3_ENDPOINT} -i ${S3_AK} -k ${S3_SK}

              echo "[3/5] Packaging PostgreSQL data..."
              tar -czf /tmp/postgres-data.tar.gz -C /mnt/postgres .
              sha256sum /tmp/postgres-data.tar.gz > /tmp/postgres-data.tar.gz.sha256

              echo "[4/5] Uploading to object storage..."
              ossutil cp /tmp/postgres-data.tar.gz oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f
              ossutil cp /tmp/postgres-data.tar.gz.sha256 oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f

              echo "[5/5] Recording permission info..."
              ls -ln /mnt/postgres > /tmp/postgres-perm_info.txt
              ossutil cp /tmp/postgres-perm_info.txt oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f

              echo "--- PostgreSQL backup complete ---"
          volumeMounts:
            - name: swanlab-pg-data
              mountPath: /mnt/postgres
      volumes:
        - name: swanlab-pg-data
          persistentVolumeClaim:
            claimName: swanlab-postgres-pvc # ⚠️ Required: postgres pvc name in [Source Cluster] K8s namespace
```

:::

::: details export-clickhouse

::: code-group

```yaml [s3cmd]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-clickhouse
  namespace: <source_namespace> # ⚠️ Required: [Source Cluster] K8s namespace
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== Local deployment scenario: specify node (delete this section for cross-cluster K8s) =====
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
              echo "[1/5] Configuring apk mirror source..."
              sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
              echo "[2/5] Installing s3cmd..."
              apk add --no-cache s3cmd

              echo "[3/5] Writing s3cmd configuration..."
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

              echo "[4/5] Packaging ClickHouse data..."
              tar -czf /tmp/clickhouse-data.tar.gz -C /mnt/clickhouse .
              sha256sum /tmp/clickhouse-data.tar.gz > /tmp/clickhouse-data.tar.gz.sha256

              echo "[5/5] Uploading to object storage..."
              s3cmd put /tmp/clickhouse-data.tar.gz s3://${S3_BUCKET}/${S3_PATH_PREFIX}/
              s3cmd put /tmp/clickhouse-data.tar.gz.sha256 s3://${S3_BUCKET}/${S3_PATH_PREFIX}/
              echo "--- ClickHouse backup complete ---"
          volumeMounts:
            - name: swanlab-ch-data
              mountPath: /mnt/clickhouse
      volumes:
        - name: swanlab-ch-data
          persistentVolumeClaim:
            claimName: swanlab-clickhouse-pvc # ⚠️ Required: clickhouse pvc name in [Source Cluster] K8s namespace
```

```yaml [ossutil]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-clickhouse
  namespace: <source_namespace> # ⚠️ Required: [Source Cluster] K8s namespace
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== Local deployment scenario: specify node (delete this section for cross-cluster K8s) =====
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
              echo "[1/5] Installing ossutil..."
              wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64.zip -O /tmp/ossutil.zip
              unzip -q /tmp/ossutil.zip -d /tmp/ossutil-dir
              mv /tmp/ossutil-dir/ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil
              chmod +x /usr/local/bin/ossutil

              echo "[2/5] Configuring ossutil..."
              ossutil config -e ${S3_ENDPOINT} -i ${S3_AK} -k ${S3_SK}

              echo "[3/5] Packaging ClickHouse data..."
              tar -czf /tmp/clickhouse-data.tar.gz -C /mnt/clickhouse .
              sha256sum /tmp/clickhouse-data.tar.gz > /tmp/clickhouse-data.tar.gz.sha256

              echo "[4/5] Uploading to object storage..."
              ossutil cp /tmp/clickhouse-data.tar.gz oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f
              ossutil cp /tmp/clickhouse-data.tar.gz.sha256 oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f

              echo "[5/5] Recording permission info..."
              ls -ln /mnt/clickhouse > /tmp/clickhouse-perm_info.txt
              ossutil cp /tmp/clickhouse-perm_info.txt oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f

              echo "--- ClickHouse backup complete ---"
          volumeMounts:
            - name: swanlab-ch-data
              mountPath: /mnt/clickhouse
      volumes:
        - name: swanlab-ch-data
          persistentVolumeClaim:
            claimName: swanlab-clickhouse-pvc # ⚠️ Required: clickhouse pvc name in [Source Cluster] K8s namespace
```

:::

::: details export-redis

::: code-group

```yaml [s3cmd]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-redis
  namespace: <source_namespace> # ⚠️ Required: [Source Cluster] K8s namespace
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== Local deployment scenario: specify node (delete this section for cross-cluster K8s) =====
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
              echo "[1/5] Configuring apk mirror source..."
              sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
              echo "[2/5] Installing s3cmd..."
              apk add --no-cache s3cmd

              echo "[3/5] Writing s3cmd configuration..."
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

              echo "[4/5] Packaging Redis data..."
              tar -czf /tmp/redis-data.tar.gz -C /mnt/redis .
              sha256sum /tmp/redis-data.tar.gz > /tmp/redis-data.tar.gz.sha256

              echo "[5/5] Uploading to object storage..."
              s3cmd put /tmp/redis-data.tar.gz s3://${S3_BUCKET}/${S3_PATH_PREFIX}/
              s3cmd put /tmp/redis-data.tar.gz.sha256 s3://${S3_BUCKET}/${S3_PATH_PREFIX}/
              echo "--- Redis backup complete ---"
          volumeMounts:
            - name: swanlab-redis-data
              mountPath: /mnt/redis
      volumes:
        - name: swanlab-redis-data
          persistentVolumeClaim:
            claimName: swanlab-redis-pvc # ⚠️ Required: redis pvc name in [Source Cluster] K8s namespace
```

```yaml [ossutil]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-redis
  namespace: <source_namespace> # ⚠️ Required: [Source Cluster] K8s namespace
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== Local deployment scenario: specify node (delete this section for cross-cluster K8s) =====
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
              echo "[1/5] Installing ossutil..."
              wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64.zip -O /tmp/ossutil.zip
              unzip -q /tmp/ossutil.zip -d /tmp/ossutil-dir
              mv /tmp/ossutil-dir/ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil
              chmod +x /usr/local/bin/ossutil

              echo "[2/5] Configuring ossutil..."
              ossutil config -e ${S3_ENDPOINT} -i ${S3_AK} -k ${S3_SK}

              echo "[3/5] Packaging Redis data..."
              tar -czf /tmp/redis-data.tar.gz -C /mnt/redis .
              sha256sum /tmp/redis-data.tar.gz > /tmp/redis-data.tar.gz.sha256

              echo "[4/5] Uploading to object storage..."
              ossutil cp /tmp/redis-data.tar.gz oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f
              ossutil cp /tmp/redis-data.tar.gz.sha256 oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f

              echo "[5/5] Recording permission info..."
              ls -ln /mnt/redis > /tmp/redis-perm_info.txt
              ossutil cp /tmp/redis-perm_info.txt oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f

              echo "--- Redis backup complete ---"
          volumeMounts:
            - name: swanlab-redis-data
              mountPath: /mnt/redis
      volumes:
        - name: swanlab-redis-data
          persistentVolumeClaim:
            claimName: swanlab-redis-pvc # ⚠️ Required: redis pvc name in [Source Cluster] K8s namespace
```

:::

::: details export-vector

::: code-group

```yaml [s3cmd]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-vector
  namespace: <source_namespace> # ⚠️ Required: [Source Cluster] K8s namespace
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== Local deployment scenario: specify node (delete this section for cross-cluster K8s) =====
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
              echo "[1/5] Configuring apk mirror source..."
              sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
              echo "[2/5] Installing s3cmd..."
              apk add --no-cache s3cmd

              echo "[3/5] Writing s3cmd configuration..."
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

              echo "[4/5] Packaging Vector data..."
              tar -czf /tmp/vector-data.tar.gz -C /mnt vector-0 vector-1
              sha256sum /tmp/vector-data.tar.gz > /tmp/vector-data.tar.gz.sha256

              echo "[5/5] Uploading to object storage..."
              s3cmd put /tmp/vector-data.tar.gz s3://${S3_BUCKET}/${S3_PATH_PREFIX}/
              s3cmd put /tmp/vector-data.tar.gz.sha256 s3://${S3_BUCKET}/${S3_PATH_PREFIX}/
              echo "--- Vector backup complete ---"
          volumeMounts:
            - name: vector-data-0
              mountPath: /mnt/vector-0
            - name: vector-data-1
              mountPath: /mnt/vector-1
      volumes:
        - name: vector-data-0
          persistentVolumeClaim:
            claimName: data-swanlab-self-hosted-vector-0 # ⚠️ Required: vector-0 pvc name in [Source Cluster] K8s namespace
        - name: vector-data-1
          persistentVolumeClaim:
            claimName: data-swanlab-self-hosted-vector-1 # ⚠️ Required: vector-1 pvc name in [Source Cluster] K8s namespace
```

```yaml [ossutil]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-vector
  namespace: <source_namespace> # ⚠️ Required: [Source Cluster] K8s namespace
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== Local deployment scenario: specify node (delete this section for cross-cluster K8s) =====
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
              echo "[1/5] Installing ossutil..."
              wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64.zip -O /tmp/ossutil.zip
              unzip -q /tmp/ossutil.zip -d /tmp/ossutil-dir
              mv /tmp/ossutil-dir/ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil
              chmod +x /usr/local/bin/ossutil

              echo "[2/5] Configuring ossutil..."
              ossutil config -e ${S3_ENDPOINT} -i ${S3_AK} -k ${S3_SK}

              echo "[3/5] Packaging Vector data..."
              tar -czf /tmp/vector-data.tar.gz -C /mnt vector-0 vector-1
              sha256sum /tmp/vector-data.tar.gz > /tmp/vector-data.tar.gz.sha256

              echo "[4/5] Uploading to object storage..."
              ossutil cp /tmp/vector-data.tar.gz oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f
              ossutil cp /tmp/vector-data.tar.gz.sha256 oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f

              echo "[5/5] Recording permission info..."
              echo "=== vector-0 ===" > /tmp/vector-perm_info.txt
              ls -ln /mnt/vector-0 >> /tmp/vector-perm_info.txt
              echo "=== vector-1 ===" >> /tmp/vector-perm_info.txt
              ls -ln /mnt/vector-1 >> /tmp/vector-perm_info.txt
              ossutil cp /tmp/vector-perm_info.txt oss://${S3_BUCKET}/${S3_PATH_PREFIX}/ -f

              echo "--- Vector backup complete ---"
          volumeMounts:
            - name: vector-data-0
              mountPath: /mnt/vector-0
            - name: vector-data-1
              mountPath: /mnt/vector-1
      volumes:
        - name: vector-data-0
          persistentVolumeClaim:
            claimName: data-swanlab-self-hosted-vector-0 # ⚠️ Required: vector-0 pvc name in [Source Cluster] K8s namespace
        - name: vector-data-1
          persistentVolumeClaim:
            claimName: data-swanlab-self-hosted-vector-1 # ⚠️ Required: vector-1 pvc name in [Source Cluster] K8s namespace
```

:::

```bash
# Execute all export Jobs in parallel
kubectl apply -f export/

# View execution status
kubectl logs -f job/swanlab-export-postgres -n <your_namespace>
kubectl logs -f job/swanlab-export-clickhouse -n <your_namespace>
kubectl logs -f job/swanlab-export-redis -n <your_namespace>

# Confirm all Jobs are complete
kubectl get jobs -n <your_namespace>
```

### 4. Export S3 Data (Optional)

- Operation location: <span style="color: red"><strong>Source Cluster</strong></span>

#### Case 1: Original Cluster Already Integrated with S3 URL

If S3 endpoint is already mounted, you just need to configure the same S3 endpoint configuration in the source cluster's `value.yaml`. For details, see [External S3 Integration Configuration](./configuration.md#external-object-storage-s3-integrations-s3).

#### Case 2: Original Cluster Uses MiniO Mounted PVC

MiniIO uses sharded storage and generally cannot be directly uploaded to object storage. It needs to be processed with `rclone` before syncing to public cloud object storage:

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
    # Container 1: Local MinIO (resolve original physical files)
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

    # Container 2: Rclone migration tool
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

# View rclone logs
kubectl logs -f export-s3-pod -c rclone-worker
```

### 5. Import DB Data

- Operation location: <span style="color: red"><strong>Target Cluster</strong></span>

Similar to export, each database has an independent import Job and can be executed in parallel. Choose `s3cmd` or `ossutil` version based on the object storage type:

::: details import-postgres

::: code-group

```yaml [s3cmd]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-postgres
  namespace: <target_namespace> # ⚠️ Required: [Target Cluster] K8s namespace
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
              echo "[1/6] Configuring apk mirror source..."
              sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
              echo "[2/6] Installing s3cmd..."
              apk add --no-cache s3cmd

              echo "[3/6] Writing s3cmd configuration..."
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

              echo "[4/6] Verifying Bucket access..."
              s3cmd ls s3://${S3_BUCKET}/${S3_PATH_PREFIX}/

              echo "[5/6] Downloading backup files..."
              s3cmd get s3://${S3_BUCKET}/${S3_PATH_PREFIX}/postgres-data.tar.gz /tmp/postgres-data.tar.gz
              s3cmd get s3://${S3_BUCKET}/${S3_PATH_PREFIX}/postgres-data.tar.gz.sha256 /tmp/postgres-data.tar.gz.sha256

              echo "[6/6] Verifying and extracting..."
              cd /tmp && sha256sum -c postgres-data.tar.gz.sha256
              tar -xzf /tmp/postgres-data.tar.gz -C /data
              ls -lh /data
      volumes:
        - name: swanlab-pg-data
          persistentVolumeClaim:
            claimName: swanlab-postgres-pvc # ⚠️ Required: postgres pvc name in [Target Cluster] K8s namespace
```

```yaml [ossutil]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-postgres
  namespace: <target_namespace> # ⚠️ Required: [Target Cluster] K8s namespace
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
              echo "[1/5] Installing ossutil..."
              wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64.zip -O /tmp/ossutil.zip
              unzip -q /tmp/ossutil.zip -d /tmp/ossutil-dir
              mv /tmp/ossutil-dir/ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil
              chmod +x /usr/local/bin/ossutil

              echo "[2/5] Configuring ossutil..."
              ossutil config -e ${S3_ENDPOINT} -i ${S3_AK} -k ${S3_SK}

              echo "[3/5] Downloading backup files..."
              ossutil cp oss://${S3_BUCKET}/${S3_PATH_PREFIX}/postgres-data.tar.gz /tmp/postgres-data.tar.gz -f
              ossutil cp oss://${S3_BUCKET}/${S3_PATH_PREFIX}/postgres-data.tar.gz.sha256 /tmp/postgres-data.tar.gz.sha256 -f

              echo "[4/5] Verifying and extracting..."
              cd /tmp && sha256sum -c postgres-data.tar.gz.sha256
              tar -xzf /tmp/postgres-data.tar.gz -C /data
              ls -lh /data

              echo "--- PostgreSQL restore complete ---"
      volumes:
        - name: swanlab-pg-data
          persistentVolumeClaim:
            claimName: swanlab-postgres-pvc # ⚠️ Required: postgres pvc name in [Target Cluster] K8s namespace
```

:::

::: details import-clickhouse

::: code-group

```yaml [s3cmd]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-clickhouse
  namespace: <target_namespace> # ⚠️ Required: [Target Cluster] K8s namespace
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
              echo "[1/6] Configuring apk mirror source..."
              sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
              echo "[2/6] Installing s3cmd..."
              apk add --no-cache s3cmd

              echo "[3/6] Writing s3cmd configuration..."
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

              echo "[4/6] Verifying Bucket access..."
              s3cmd ls s3://${S3_BUCKET}/${S3_PATH_PREFIX}/

              echo "[5/6] Downloading backup files..."
              s3cmd get s3://${S3_BUCKET}/${S3_PATH_PREFIX}/clickhouse-data.tar.gz /tmp/clickhouse-data.tar.gz
              s3cmd get s3://${S3_BUCKET}/${S3_PATH_PREFIX}/clickhouse-data.tar.gz.sha256 /tmp/clickhouse-data.tar.gz.sha256

              echo "[6/6] Verifying and extracting..."
              cd /tmp && sha256sum -c clickhouse-data.tar.gz.sha256
              tar -xzf /tmp/clickhouse-data.tar.gz -C /data
              ls -lh /data
      volumes:
        - name: swanlab-ch-data
          persistentVolumeClaim:
            claimName: swanlab-clickhouse-pvc # ⚠️ Required: clickhouse pvc name in [Target Cluster] K8s namespace
```

```yaml [ossutil]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-clickhouse
  namespace: <target_namespace> # ⚠️ Required: [Target Cluster] K8s namespace
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
              echo "[1/5] Installing ossutil..."
              wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64.zip -O /tmp/ossutil.zip
              unzip -q /tmp/ossutil.zip -d /tmp/ossutil-dir
              mv /tmp/ossutil-dir/ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil
              chmod +x /usr/local/bin/ossutil

              echo "[2/5] Configuring ossutil..."
              ossutil config -e ${S3_ENDPOINT} -i ${S3_AK} -k ${S3_SK}

              echo "[3/5] Downloading backup files..."
              ossutil cp oss://${S3_BUCKET}/${S3_PATH_PREFIX}/clickhouse-data.tar.gz /tmp/clickhouse-data.tar.gz -f
              ossutil cp oss://${S3_BUCKET}/${S3_PATH_PREFIX}/clickhouse-data.tar.gz.sha256 /tmp/clickhouse-data.tar.gz.sha256 -f

              echo "[4/5] Verifying and extracting..."
              cd /tmp && sha256sum -c clickhouse-data.tar.gz.sha256
              tar -xzf /tmp/clickhouse-data.tar.gz -C /data
              ls -lh /data

              echo "--- ClickHouse restore complete ---"
      volumes:
        - name: swanlab-ch-data
          persistentVolumeClaim:
            claimName: swanlab-clickhouse-pvc # ⚠️ Required: clickhouse pvc name in [Target Cluster] K8s namespace
```

:::

::: details import-redis

::: code-group

```yaml [s3cmd]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-redis
  namespace: <target_namespace> # ⚠️ Required: [Target Cluster] K8s namespace
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
              echo "[1/6] Configuring apk mirror source..."
              sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
              echo "[2/6] Installing s3cmd..."
              apk add --no-cache s3cmd

              echo "[3/6] Writing s3cmd configuration..."
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

              echo "[4/6] Verifying Bucket access..."
              s3cmd ls s3://${S3_BUCKET}/${S3_PATH_PREFIX}/

              echo "[5/6] Downloading backup files..."
              s3cmd get s3://${S3_BUCKET}/${S3_PATH_PREFIX}/redis-data.tar.gz /tmp/redis-data.tar.gz
              s3cmd get s3://${S3_BUCKET}/${S3_PATH_PREFIX}/redis-data.tar.gz.sha256 /tmp/redis-data.tar.gz.sha256

              echo "[6/6] Verifying and extracting..."
              cd /tmp && sha256sum -c redis-data.tar.gz.sha256
              tar -xzf /tmp/redis-data.tar.gz -C /data
              ls -lh /data
      volumes:
        - name: swanlab-redis-data
          persistentVolumeClaim:
            claimName: swanlab-redis-pvc # ⚠️ Required: redis pvc name in [Target Cluster] K8s namespace
```

```yaml [ossutil]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-redis
  namespace: <target_namespace> # ⚠️ Required: [Target Cluster] K8s namespace
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
              echo "[1/5] Installing ossutil..."
              wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64.zip -O /tmp/ossutil.zip
              unzip -q /tmp/ossutil.zip -d /tmp/ossutil-dir
              mv /tmp/ossutil-dir/ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil
              chmod +x /usr/local/bin/ossutil

              echo "[2/5] Configuring ossutil..."
              ossutil config -e ${S3_ENDPOINT} -i ${S3_AK} -k ${S3_SK}

              echo "[3/5] Downloading backup files..."
              ossutil cp oss://${S3_BUCKET}/${S3_PATH_PREFIX}/redis-data.tar.gz /tmp/redis-data.tar.gz -f
              ossutil cp oss://${S3_BUCKET}/${S3_PATH_PREFIX}/redis-data.tar.gz.sha256 /tmp/redis-data.tar.gz.sha256 -f

              echo "[4/5] Verifying and extracting..."
              cd /tmp && sha256sum -c redis-data.tar.gz.sha256
              tar -xzf /tmp/redis-data.tar.gz -C /data
              ls -lh /data

              echo "--- Redis restore complete ---"
      volumes:
        - name: swanlab-redis-data
          persistentVolumeClaim:
            claimName: swanlab-redis-pvc # ⚠️ Required: redis pvc name in [Target Cluster] K8s namespace
```

:::

::: details import-vector

::: code-group

```yaml [s3cmd]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-vector
  namespace: <target_namespace> # ⚠️ Required: [Target Cluster] K8s namespace
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
              echo "[1/6] Configuring apk mirror source..."
              sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories
              echo "[2/6] Installing s3cmd..."
              apk add --no-cache s3cmd

              echo "[3/6] Writing s3cmd configuration..."
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

              echo "[4/6] Verifying Bucket access..."
              s3cmd ls s3://${S3_BUCKET}/${S3_PATH_PREFIX}/

              echo "[5/6] Downloading backup files..."
              s3cmd get s3://${S3_BUCKET}/${S3_PATH_PREFIX}/vector-data.tar.gz /tmp/vector-data.tar.gz
              s3cmd get s3://${S3_BUCKET}/${S3_PATH_PREFIX}/vector-data.tar.gz.sha256 /tmp/vector-data.tar.gz.sha256

              echo "[6/6] Verifying and extracting..."
              cd /tmp && sha256sum -c vector-data.tar.gz.sha256
              tar -xzf /tmp/vector-data.tar.gz -C /data
              ls -lh /data/vector-0
              ls -lh /data/vector-1
      volumes:
        - name: vector-data-0
          persistentVolumeClaim:
            claimName: data-swanlab-self-hosted-vector-0 # ⚠️ Required: vector-0 pvc name in [Target Cluster] K8s namespace
        - name: vector-data-1
          persistentVolumeClaim:
            claimName: data-swanlab-self-hosted-vector-1 # ⚠️ Required: vector-1 pvc name in [Target Cluster] K8s namespace
```

```yaml [ossutil]
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-vector
  namespace: <target_namespace> # ⚠️ Required: [Target Cluster] K8s namespace
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
              echo "[1/5] Installing ossutil..."
              wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil-v1.7.18-linux-amd64.zip -O /tmp/ossutil.zip
              unzip -q /tmp/ossutil.zip -d /tmp/ossutil-dir
              mv /tmp/ossutil-dir/ossutil-v1.7.18-linux-amd64/ossutil /usr/local/bin/ossutil
              chmod +x /usr/local/bin/ossutil

              echo "[2/5] Configuring ossutil..."
              ossutil config -e ${S3_ENDPOINT} -i ${S3_AK} -k ${S3_SK}

              echo "[3/5] Downloading backup files..."
              ossutil cp oss://${S3_BUCKET}/${S3_PATH_PREFIX}/vector-data.tar.gz /tmp/vector-data.tar.gz -f
              ossutil cp oss://${S3_BUCKET}/${S3_PATH_PREFIX}/vector-data.tar.gz.sha256 /tmp/vector-data.tar.gz.sha256 -f

              echo "[4/5] Verifying and extracting..."
              cd /tmp && sha256sum -c vector-data.tar.gz.sha256
              tar -xzf /tmp/vector-data.tar.gz -C /data
              ls -lh /data/vector-0
              ls -lh /data/vector-1

              echo "--- Vector restore complete ---"
      volumes:
        - name: vector-data-0
          persistentVolumeClaim:
            claimName: data-swanlab-self-hosted-vector-0 # ⚠️ Required: vector-0 pvc name in [Target Cluster] K8s namespace
        - name: vector-data-1
          persistentVolumeClaim:
            claimName: data-swanlab-self-hosted-vector-1 # ⚠️ Required: vector-1 pvc name in [Target Cluster] K8s namespace
```

:::

```bash
# Execute all import Jobs in parallel
kubectl apply -f import/

# View execution status
kubectl logs -f job/swanlab-import-postgres -n <your_namespace>
kubectl logs -f job/swanlab-import-clickhouse -n <your_namespace>
kubectl logs -f job/swanlab-import-redis -n <your_namespace>

# Confirm all Jobs are complete
kubectl get jobs -n <your_namespace>
```

### 6. Restart Services

- Operation location: <span style="color: red"><strong>Target Cluster</strong></span>

Be sure to restart services in order.

::: code-group

```bash [1. Restore Databases]
# Restore database services (replicas must be 1)
kubectl scale deployment swanlab-self-hosted-clickhouse --replicas=1 -n <your_namespace>
kubectl scale deployment swanlab-self-hosted-postgres --replicas=1 -n <your_namespace>
kubectl scale deployment swanlab-self-hosted-redis --replicas=1 -n <your_namespace>

# Confirm databases are ready
kubectl get pods -n <your_namespace> -w
```

```bash [2. Restore Vector]
# StatefulSet, dual replicas
kubectl scale statefulset swanlab-self-hosted-vector --replicas=2 -n <your_namespace>
```

```bash [3. Restore Application Layer]
# Restore replicas first, then scale as needed
kubectl scale deployment swanlab-self-hosted-house --replicas=1 -n <your_namespace>
kubectl scale deployment swanlab-self-hosted-server --replicas=1 -n <your_namespace>
```

```bash [4. Restore Gateway]
# Restore gateway
kubectl scale deployment swanlab-self-hosted --replicas=2 -n <your_namespace>
```

```bash [「Optional」5. Restore S3]
# 「Optional」If using external S3, this can be ignored. If using the template's built-in MinIO, you need to manually restore S3
kubectl scale deployment swanlab-self-hosted-s3 --replicas=1 -n <your_namespace>
```

:::

After restoration, you can observe pod health status and verify data recovery through online service.

## 🧹 Job Cleanup

Jobs on the original and target clusters will be **automatically cleaned up after 24 hours** (`ttlSecondsAfterFinished: 86400`). If manual cleanup is needed:

```bash
# Source cluster
kubectl delete job swanlab-export-postgres swanlab-export-clickhouse swanlab-export-redis swanlab-export-vector -n <original_namespace>

# Target cluster
kubectl delete job swanlab-import-postgres swanlab-import-clickhouse swanlab-import-redis swanlab-import-vector -n <target_namespace>
```
