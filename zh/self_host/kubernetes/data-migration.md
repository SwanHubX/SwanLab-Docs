# 数据迁移

> 本页介绍 SwanLab 私有化版本在不同 Kubernetes 集群之间进行全量数据迁移的完整流程。

## 迁移流程示意图

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/cross-cluster-migration-v2.drawio.svg"/>

SwanLab 私有化服务的数据库进行跨集群数据迁移的流程，包含三个核心区域：

- **源集群 (Source)**：数据导出端，包含数据库和对象存储。
- **中转存储 (Transit)**：S3 对象存储作为临时中转站
- **目标集群 (Target)**：数据接收端，完成恢复。

| 数据类型               | 迁移方式                                         | 说明                                  |
| ---------------------- | ------------------------------------------------ | ------------------------------------- |
| PostgreSQL             | `pg_dump` → S3 中转 → `pg_restore`               | 客户端直连 Service，不需要挂载 PVC    |
| ClickHouse             | 原生 `BACKUP DATABASE` → S3 → `RESTORE DATABASE` | 客户端直连 Service，原生 S3 备份/恢复 |
| Redis                  | `redis-cli --rdb` → S3 中转 → 落盘 PVC           | 导出直连 Service，导入写入目标 PVC    |
| 对象存储（MinIO / S3） | `rclone sync` 直传云端 S3 或直接对接云端 S3      | 存算分离时免搬运                      |

**图例说明**：

- 🔵 **Phase 1: Export** — 源集群导出数据到 S3 中转存储桶（DB Export + S3 Sync）
- 🟢 **Phase 2: Import** — 数据库数据从 S3 导入到新集群（DB Import，物理迁移）
- 🟢 **Phase 2: Direct Use**（虚线）— 对象存储不移动，新集群直接通过 `values.yaml` 同样的配置访问 public 和 private 对象存储桶

## 🧾 前置条件

### 资源准备

- 兼容 S3 协议的对象存储桶，可用空间至少大于 ClickHouse + PostgreSQL + Redis 存储总和的 **1.1 倍**
- 目标集群需要额外部署一套 **全新未激活** 的 SwanLab 服务，且必须挂载云硬盘
- 如果原始集群和目标集群版本不一致，建议先对原始集群的 chart 版本进行升级

### 镜像准备

| 镜像                                                 | 用途            | 说明                                      |
| ---------------------------------------------------- | --------------- | ----------------------------------------- |
| `repo.swanlab.cn/public/pg-migrator:16.1`            | PostgreSQL 迁移 | 含 pg_dump / pg_restore / rclone          |
| `repo.swanlab.cn/self-hosted/clickhouse-server:24.3` | ClickHouse 迁移 | 含 clickhouse-client，原生 BACKUP/RESTORE |
| `repo.swanlab.cn/public/redis-migrator:7.4.0-v8`     | Redis 迁移      | 含 redis-cli / rclone                     |
| `repo.swanlab.cn/public/s3-migrator:bookworm-slim`   | S3 迁移         | 含 rclone，用于 MinIO → 云端 S3 同步      |

### 权限准备

- 对原始集群和目标集群均具有可写权限（原始集群和目标集群可以是同一个集群，但必须各自存在一套独立的 SwanLab 服务）
- 准备好 `access_key` 和 `secret_key`，需要具备对象存储桶的读写权限
- ClickHouse 用户需具备 `BACKUP` / `RESTORE` 权限
- PostgreSQL 用户需具备 `pg_dump` / `pg_restore` 权限

### 配置变量

使用前请确认以下信息：

| 是否准备 | 占位变量                                | 说明                                                          |
| -------- | --------------------------------------- | ------------------------------------------------------------- |
| ✅       | `SOURCE_NAMESPACE` / `TARGET_NAMESPACE` | 源集群和目标集群的命名空间                                    |
| ✅       | `S3_REGION`                             | 用于备份的对象存储桶的地域                                    |
| ✅       | `S3_BUCKET`                             | 存储桶名称                                                    |
| ✅       | `S3_ENDPOINT`                           | S3 格式的对象存储 Endpoint，如 `tos-s3-cn-beijing.volces.com` |
| ✅       | `S3_AK` / `S3_SK`                       | 对象存储可写密钥                                              |
| ✅       | `S3_PATH_PREFIX`                        | S3 中的备份路径前缀，默认 `origin-backup-datas`               |

> ⚠️ S3 迁移工具使用 `rclone`，已验证兼容腾讯云 COS、火山引擎 TOS、阿里云 OSS 等主流 S3 兼容存储。

## 🪜 操作步骤

:::warning

**⚠️ 迁移前必须停机！** 数据迁移时需要确保**两套 SwanLab 应用层服务都保持停机状态**，由中间 Job 执行迁移，否则会因为数据写入状态不一致等问题，造成迁移失败。
:::

### 1. 修改配置文件

- 操作位置：<span style="color: red"><strong>源集群、目标集群</strong></span>

迁移配置按组件拆分为独立的 ConfigMap / Secret，需**分别在源集群和目标集群创建**。

#### 源集群配置（config-export.yaml）

源集群需要创建以下配置：

- `swanlab-backup-storage-config` / `swanlab-backup-storage-secret` — S3 中转桶凭据
- `swanlab-pg-backup-config` / `swanlab-pg-backup-secret` — PG 连接信息
- `swanlab-ch-backup-config` / `swanlab-ch-backup-secret` — CH 连接信息
- `swanlab-redis-backup-config` — Redis 连接信息

::: details config-export.yaml

```yaml
## ============================================================
## 源集群：迁移配置（export 专用）
## ============================================================

## Part 1: 备份中转桶（DB export/import 共用）
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-backup-storage-config
  namespace: <SOURCE_NAMESPACE> # ⚠️ 必填：【源集群】K8s 命名空间
data:
  S3_ENDPOINT: "<BACKUP_S3_ENDPOINT>" # ⚠️ 备份中转桶 endpoint，无 bucket 前缀
  S3_REGION: "<BACKUP_S3_REGION>" # ⚠️ 备份中转桶 region
  S3_BUCKET: "<BACKUP_S3_BUCKET>" # ⚠️ 备份中转桶名
  S3_PATH_PREFIX: "<S3_PATH_PREFIX>" # ⚠️ 桶内路径前缀（import 侧需保持一致）
  S3_FORCE_PATH_STYLE: "false"
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-backup-storage-secret
  namespace: <SOURCE_NAMESPACE>
type: Opaque
stringData:
  S3_AK: "<BACKUP_S3_AK>" # ⚠️ 备份中转桶 AK
  S3_SK: "<BACKUP_S3_SK>" # ⚠️ 备份中转桶 SK
---
## Part 2: Redis 连接配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-redis-backup-config
  namespace: <SOURCE_NAMESPACE>
data:
  REDIS_HOST: "swanlab-self-hosted-redis" # ⚠️ 【源集群】Redis Service 名
  REDIS_PORT: "6379" # ⚠️ Redis 端口
  REDIS_BACKUP_OBJECT: "redis-restore.rdb" # ⚠️ S3 中转对象名（import 侧需一致）
---
## Part 3: PostgreSQL 连接配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-pg-backup-config
  namespace: <SOURCE_NAMESPACE>
data:
  PG_HOST: "postgres" # ⚠️ 【源集群】PG Service 名
  PG_USER: "postgres" # ⚠️ PG 用户名
  PG_DB: "app" # ⚠️ PG 数据库名
  PG_BACKUP_OBJECT: "postgres-restore.dump" # ⚠️ S3 中转对象名（import 侧需一致）
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-pg-backup-secret
  namespace: <SOURCE_NAMESPACE>
type: Opaque
stringData:
  PG_PASSWORD: "<PG_PASSWORD>" # ⚠️ 与 Helm values 中 postgres password 一致
---
## Part 4: ClickHouse 连接配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-ch-backup-config
  namespace: <SOURCE_NAMESPACE>
data:
  CLICKHOUSE_HOST: "<CLICKHOUSE_HOST>" # ⚠️ 【源集群】CH Service 名
  CLICKHOUSE_DB: "app" # ⚠️ 待备份的数据库名
  CLICKHOUSE_BACKUP_PATH: "clickhouse-backup" # ⚠️ S3 中转桶内备份子路径（import 侧需一致）
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-ch-backup-secret
  namespace: <SOURCE_NAMESPACE>
type: Opaque
stringData:
  CLICKHOUSE_USER: "<CLICKHOUSE_USER>" # ⚠️ 与 Helm values 中 CH username 一致
  CLICKHOUSE_PASSWORD: "<CLICKHOUSE_PASSWORD>" # ⚠️ 与 Helm values 中 CH password 一致
```

:::

::: tip
如果源集群使用的是 MinIO 内置存储，还需要额外创建 `swanlab-cloud-s3-config` / `swanlab-cloud-s3-secret`，详见 [导出 S3 数据](#_4-导出-s3-数据-可选) 章节。
:::

```bash [源集群]
kubectl apply -f config-export.yaml
```

#### 目标集群配置（config-import.yaml）

目标集群需要创建以下配置：

- `swanlab-backup-storage-config` / `swanlab-backup-storage-secret` — S3 中转桶凭据
- `swanlab-pg-backup-config` / `swanlab-pg-backup-secret` — 目标 PG 连接信息
- `swanlab-ch-backup-config` / `swanlab-ch-backup-secret` — 目标 CH 连接信息
- `swanlab-redis-backup-config` — Redis 对象名

::: details config-import.yaml

```yaml
## ============================================================
## 目标集群：迁移配置（import 专用）
## ============================================================
## 只有 DB 数据（PG/CH/Redis）需要从中转桶 import 回来
## MinIO 数据直接上云，不经过中转桶，不需要 import 配置
## ============================================================

apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-backup-storage-config
  namespace: <TARGET_NAMESPACE> # ⚠️ 必填：【目标集群】K8s 命名空间
data:
  S3_ENDPOINT: "<BACKUP_S3_ENDPOINT>" # ⚠️ 备份中转桶 endpoint，无 bucket 前缀
  S3_REGION: "<BACKUP_S3_REGION>" # ⚠️ 备份中转桶 region
  S3_BUCKET: "<BACKUP_S3_BUCKET>" # ⚠️ 备份中转桶名
  S3_PATH_PREFIX: "<S3_PATH_PREFIX>" # ⚠️ 桶内路径前缀
  S3_FORCE_PATH_STYLE: "false"
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-backup-storage-secret
  namespace: <TARGET_NAMESPACE>
type: Opaque
stringData:
  S3_AK: "<BACKUP_S3_AK>" # ⚠️ 备份中转桶 AK
  S3_SK: "<BACKUP_S3_SK>" # ⚠️ 备份中转桶 SK
---
## Redis 对象名配置（import 仅下载落盘，不连接 Redis）
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-redis-backup-config
  namespace: <TARGET_NAMESPACE>
data:
  REDIS_BACKUP_OBJECT: "redis-restore.rdb" # ⚠️ 需与 export 侧一致
---
## PostgreSQL 连接配置（import 需连接目标集群 PG 执行 pg_restore）
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-pg-backup-config
  namespace: <TARGET_NAMESPACE>
data:
  ## 迁移目标二选一：
  ##   A) 内置 PG：填 K8s Service 名，例如 swanlab-self-hosted-postgres
  ##   B) 云端 RDS：填 RDS 域名，例如 rm-xxx.pg.rds.aliyuncs.com
  ##      ⚠️ 选 B 时，需将 import Job 所在集群出口 IP 加入 RDS 白名单
  PG_HOST: "<PG_HOST>" # ⚠️ K8s Service 名 或 云端 RDS 域名
  PG_PORT: "5432" # ⚠️ PG 端口（云端 RDS 按实际填）
  PG_USER: "<PG_USER>" # ⚠️ PG 用户名
  PG_DB: "app" # ⚠️ PG 数据库名
  PG_BACKUP_OBJECT: "postgres-restore.dump" # ⚠️ 需与 export 侧一致
  PGSSLMODE: "prefer" # ⚠️ 云端 RDS 通常需 require 或更高
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-pg-backup-secret
  namespace: <TARGET_NAMESPACE>
type: Opaque
stringData:
  PG_PASSWORD: "<PG_PASSWORD>" # ⚠️ 与目标集群 Helm values 中 postgres password 一致
---
## ClickHouse 连接配置（Job 通过 client 连接目标 CH 执行 RESTORE）
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-ch-backup-config
  namespace: <TARGET_NAMESPACE>
data:
  CLICKHOUSE_HOST: "<CLICKHOUSE_HOST>" # ⚠️ 【目标集群】CH Service 名
  CLICKHOUSE_DB: "app" # ⚠️ 待恢复的数据库名（需与 export 侧一致）
  CLICKHOUSE_BACKUP_PATH: "clickhouse-backup" # ⚠️ 需与 export 侧一致
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-ch-backup-secret
  namespace: <TARGET_NAMESPACE>
type: Opaque
stringData:
  CLICKHOUSE_USER: "<CLICKHOUSE_USER>" # ⚠️ 目标集群 CH 用户名
  CLICKHOUSE_PASSWORD: "<CLICKHOUSE_PASSWORD>" # ⚠️ 目标集群 CH 密码
```

:::

```bash [目标集群]
kubectl apply -f config-import.yaml
```

同时修改导出/导入 Job YAML 中的以下字段：

- `namespace`：对应的 K8s 命名空间
- `claimName`：对应的 PVC 名称（Redis 导入时需要）
- `nodeSelector`：本机部署场景需指定节点

### 2. 停服

- 操作位置：<span style="color: red"><strong>源集群、目标集群</strong></span>

务必按照顺序停服。

::: code-group

```bash [1. 停网关]
# 切断所有外部流量
kubectl scale deploy/swanlab-self-hosted --replicas=0 -n <your_namespace>
```

```bash [2. 停应用层]
# 停后端核心服务
kubectl scale deploy/swanlab-self-hosted-server --replicas=0 -n <your_namespace>
# 停后端指标OLAP服务
kubectl scale deploy/swanlab-self-hosted-house --replicas=0 -n <your_namespace>
```

```bash [3. 等待 Vector 消费]
# 先等缓冲区消费完（看 logs 无新写入后 Ctrl+C）
kubectl logs -f swanlab-self-hosted-vector-0 -n <your_namespace> --tail=20
kubectl logs -f swanlab-self-hosted-vector-1 -n <your_namespace> --tail=20

```

:::

:::tip
针对 **目标集群** 的 Redis 数据库，由于需要用 rdb 快照恢复服务，因此需要单独针对目标集群的Reids数据库停服:

`kubectl scale deploy/swanlab-self-hosted-redis --replicas=0 -n <your_namespace>`
:::

### 3. 导出 DB 数据

- 操作位置：<span style="color: red"><strong>源集群</strong></span>

每个数据库的迁移被封装为独立的 Job，可并行执行。

::: info 导出说明

- **PostgreSQL**：Job 作为客户端直连 PG Service，使用 `pg_dump -Fc`（custom 格式）导出后通过 `rclone` 上传至 S3 中转桶
- **ClickHouse**：Job 作为客户端直连 CH Service，使用原生 `BACKUP DATABASE ... TO S3()` 命令，CH 服务端保证 parts 一致性。
- **Redis**：Job 作为客户端直连 Redis Service，使用 `redis-cli --rdb` 导出 RDB 后通过 `rclone` 上传。

:::

::: details export-postgres

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-postgres
  namespace: <SOURCE_NAMESPACE> # ⚠️ 必填：【源集群】K8s 命名空间
  labels:
    swanlab: postgres
spec:
  backoffLimit: 1
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      ## ⚠️ /tmp 走 emptyDir + sizeLimit，避免 dump 落到节点根盘写爆触发节点级驱逐。
      ##    sizeLimit 超限时只杀本 Pod，不殃及邻居。
      volumes:
        - name: tmp
          emptyDir:
            sizeLimit: 20Gi # ⚠️ 约 10-15 GB dump；DB 更大时按实际调，勿超节点可分配
      containers:
        - name: swanlab-export-postgres
          image: repo.swanlab.cn/public/pg-migrator:16.1
          imagePullPolicy: IfNotPresent
          resources:
            limits:
              memory: 4Gi
              ephemeral-storage: 20Gi
          volumeMounts:
            - name: tmp
              mountPath: /tmp
          env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: swanlab-pg-backup-secret
                  key: PG_PASSWORD
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - configMapRef:
                name: swanlab-pg-backup-config
            - secretRef:
                name: swanlab-backup-storage-secret
          command:
            - /bin/bash
            - -c
            - |
              set -e
              set -o pipefail

              echo "[1/5] 配置 rclone S3..."
              export RCLONE_CONFIG_S3_TYPE=s3
              export RCLONE_CONFIG_S3_PROVIDER=Other
              export RCLONE_CONFIG_S3_ENDPOINT="${S3_ENDPOINT}"
              export RCLONE_CONFIG_S3_REGION="${S3_REGION}"
              export RCLONE_CONFIG_S3_ACCESS_KEY_ID="${S3_AK}"
              export RCLONE_CONFIG_S3_SECRET_ACCESS_KEY="${S3_SK}"
              export RCLONE_CONFIG_S3_NO_CHECK_BUCKET=true
              export RCLONE_CONFIG_S3_FORCE_PATH_STYLE="${S3_FORCE_PATH_STYLE:-false}"

              echo "[2/5] 连接检查..."
              pg_isready -h "${PG_HOST}" -U "${PG_USER}" -d "${PG_DB}"

              DUMP=/tmp/${PG_BACKUP_OBJECT}
              echo "[3/5] pg_dump 导出中 (custom format)..."
              (while true; do sleep 5; [ -f "${DUMP}" ] && echo "  导出中... $(du -h "${DUMP}" | cut -f1)"; done) &
              PROGRESS_PID=$!
              pg_dump \
                -h "${PG_HOST}" \
                -U "${PG_USER}" \
                -d "${PG_DB}" \
                -Fc \
                --no-owner \
                --no-acl \
                -f "${DUMP}"
              kill $PROGRESS_PID 2>/dev/null || true
              echo "  导出完成: $(du -h "${DUMP}" | cut -f1)"

              ls -lh "${DUMP}"
              pg_restore -l "${DUMP}" | sed -n '1,50p'

              PREFIX="${S3_PATH_PREFIX%/}"
              TARGET="s3:${S3_BUCKET}/${PREFIX}"

              echo "[4/5] 上传 dump 到 ${TARGET}/..."
              rclone copyto "${DUMP}" "${TARGET}/${PG_BACKUP_OBJECT}" \
                --s3-no-check-bucket \
                --contimeout 10s \
                --timeout 5m \
                --low-level-retries 3 \
                --retries 3 \
                --transfers 1 \
                --checkers 1 \
                --progress \
                -vv

              echo "[5/5] 上传后列表："
              rclone lsf "${TARGET}/" --s3-no-check-bucket
              echo "--- PostgreSQL dump 导出 + 上传完成 ---"
```

:::

::: details export-clickhouse

```yaml
## ============================================================
## ClickHouse 原生 BACKUP 导出 Job
## ============================================================
## 原理：Job 作为纯 client，连接源集群运行中的 CH Service，
##       通过原生 BACKUP DATABASE 命令将数据写入 S3 备份中转桶。
##
## ✅ 不需要 scale down CH Deployment，不挂 PVC。
##    BACKUP 在服务端执行，CH 自身保证 parts 一致性。
##
## ⚠️ 前置条件：
##   - 源集群 CH Service 对 Job 可达（9000 端口通）
##   - CH 用户具备 BACKUP 权限
##   - 源/目标 CH 版本一致
## ============================================================
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-clickhouse
  namespace: <SOURCE_NAMESPACE> # ⚠️ 必填：【源集群】K8s 命名空间
  labels:
    swanlab: clickhouse
spec:
  backoffLimit: 0 # ⚠️ TB 级备份失败不自动重跑，避免数据状态混乱
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== 本机部署场景：指定节点 =====
      # nodeSelector:
      #   kubernetes.io/hostname: <node_name>
      # tolerations:
      #   - operator: Exists
      # # ============================================================
      containers:
        - name: swanlab-export-clickhouse
          image: repo.swanlab.cn/self-hosted/clickhouse-server:24.3
          imagePullPolicy: IfNotPresent
          env:
            - name: TZ
              value: "UTC"
            - name: CLICKHOUSE_USER
              valueFrom:
                secretKeyRef:
                  name: swanlab-ch-backup-secret
                  key: CLICKHOUSE_USER
            - name: CLICKHOUSE_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: swanlab-ch-backup-secret
                  key: CLICKHOUSE_PASSWORD
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - configMapRef:
                name: swanlab-ch-backup-config
            - secretRef:
                name: swanlab-backup-storage-secret
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -e

              CH_CLIENT=(clickhouse-client
                --host "${CLICKHOUSE_HOST}"
                --user "${CLICKHOUSE_USER}"
                --password "${CLICKHOUSE_PASSWORD}")

              echo "[1/5] 连接源 ClickHouse..."
              "${CH_CLIENT[@]}" --query "SELECT version(), uptime()"
              echo "  ${CLICKHOUSE_DB} 下表列表："
              "${CH_CLIENT[@]}" --query "SHOW TABLES FROM ${CLICKHOUSE_DB}" || true

              echo "[2/5] 预估数据量..."
              "${CH_CLIENT[@]}" --query "
                SELECT
                  formatReadableSize(sum(bytes_on_disk)) AS disk_size,
                  formatReadableQuantity(sum(rows)) AS total_rows,
                  count() AS parts
                FROM system.parts
                WHERE database='${CLICKHOUSE_DB}' AND active
              "

              ## 阿里云 OSS 要求 virtual hosted style
              S3_URL="https://${S3_BUCKET}.${S3_ENDPOINT}"
              PREFIX="${S3_PATH_PREFIX%/}"
              BACKUP_PATH="${PREFIX:+${PREFIX}/}${CLICKHOUSE_BACKUP_PATH}"

              echo "[3/5] 发起 BACKUP DATABASE ${CLICKHOUSE_DB} → s3://${S3_BUCKET}/${BACKUP_PATH}..."
              ## ASYNC 模式发起，轮询 system.backups 跟踪进度
              BACKUP_RESULT=$("${CH_CLIENT[@]}" --format=TabSeparated --query "
                BACKUP DATABASE ${CLICKHOUSE_DB}
                TO S3(
                  '${S3_URL}/${BACKUP_PATH}',
                  '${S3_AK}',
                  '${S3_SK}'
                )
                SETTINGS
                  compression_method='lz4'
                ASYNC
              ")
              BACKUP_ID=$(echo "${BACKUP_RESULT}" | awk '{print $1}')
              echo "  备份任务已提交，id=${BACKUP_ID}"

              echo "[4/5] 轮询备份进度..."
              while true; do
                ROW=$("${CH_CLIENT[@]}" --format=TabSeparated --query "
                  SELECT status, num_files, uncompressed_size, compressed_size, error
                  FROM system.backups
                  WHERE id='${BACKUP_ID}'
                ")
                STATUS=$(echo "${ROW}" | awk -F'\t' '{print $1}')
                NUM_FILES=$(echo "${ROW}" | awk -F'\t' '{print $2}')
                UNCOMPRESSED=$(echo "${ROW}" | awk -F'\t' '{print $3}')
                COMPRESSED=$(echo "${ROW}" | awk -F'\t' '{print $4}')
                ERROR=$(echo "${ROW}" | awk -F'\t' '{print $5}')

                UNCOMPRESSED_H=$(awk "BEGIN{printf \"%.2f GB\", ${UNCOMPRESSED:-0}/1073741824}")
                COMPRESSED_H=$(awk "BEGIN{printf \"%.2f GB\", ${COMPRESSED:-0}/1073741824}")

                echo "  [$(date +%H:%M:%S)] status=${STATUS} files=${NUM_FILES} uncompressed=${UNCOMPRESSED_H} compressed=${COMPRESSED_H}"

                case "${STATUS}" in
                  BACKUP_CREATED)
                    echo "--- ClickHouse 原生备份完成 ---"
                    echo "备份位置: s3://${S3_BUCKET}/${BACKUP_PATH}"
                    echo "总文件数: ${NUM_FILES}，压缩后: ${COMPRESSED_H}，未压缩: ${UNCOMPRESSED_H}"
                    break
                    ;;
                  BACKUP_FAILED)
                    echo "ERROR: 备份失败：${ERROR}"
                    exit 1
                    ;;
                esac
                sleep 10
              done

              echo "[5/5] 源侧行数统计（供 import 侧校验对比）..."
              "${CH_CLIENT[@]}" --query "
                SELECT
                  '${CLICKHOUSE_DB}' AS database,
                  sum(rows) AS total_rows,
                  formatReadableQuantity(sum(rows)) AS rows_readable
                FROM system.parts
                WHERE database='${CLICKHOUSE_DB}' AND active
              "
              echo "--- 导出侧完成，请记录上述行数供 import 侧校验 ---"
```

:::

::: details export-redis

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-redis
  namespace: <SOURCE_NAMESPACE> # ⚠️ 必填：【源集群】K8s 命名空间
  labels:
    swanlab: redis
spec:
  backoffLimit: 1
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: swanlab-export-redis
          image: repo.swanlab.cn/public/redis-migrator:7.4.0-v8
          imagePullPolicy: IfNotPresent
          resources:
            limits:
              memory: 4Gi
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - configMapRef:
                name: swanlab-redis-backup-config
            - secretRef:
                name: swanlab-backup-storage-secret
          command:
            - /bin/bash
            - -c
            - |
              set -e
              set -o pipefail

              : "${REDIS_HOST:?REDIS_HOST 未配置}"
              : "${REDIS_PORT:?REDIS_PORT 未配置}"
              : "${REDIS_BACKUP_OBJECT:?REDIS_BACKUP_OBJECT 未配置}"

              echo "[1/5] 配置 rclone S3..."
              export RCLONE_CONFIG_S3_TYPE=s3
              export RCLONE_CONFIG_S3_PROVIDER=Other
              export RCLONE_CONFIG_S3_ENDPOINT="${S3_ENDPOINT}"
              export RCLONE_CONFIG_S3_REGION="${S3_REGION}"
              export RCLONE_CONFIG_S3_ACCESS_KEY_ID="${S3_AK}"
              export RCLONE_CONFIG_S3_SECRET_ACCESS_KEY="${S3_SK}"
              export RCLONE_CONFIG_S3_NO_CHECK_BUCKET=true
              export RCLONE_CONFIG_S3_FORCE_PATH_STYLE="${S3_FORCE_PATH_STYLE:-false}"

              echo "[2/5] 连接检查..."
              redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" PING

              DUMP=/tmp/${REDIS_BACKUP_OBJECT}
              echo "[3/5] redis-cli --rdb 导出中..."
              (while true; do sleep 5; [ -f "${DUMP}" ] && echo "  导出中... $(du -h "${DUMP}" | cut -f1)"; done) &
              PROGRESS_PID=$!
              redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" --rdb "${DUMP}"
              kill $PROGRESS_PID 2>/dev/null || true
              echo "  导出完成: $(du -h "${DUMP}" | cut -f1)"

              ls -lh "${DUMP}"

              PREFIX="${S3_PATH_PREFIX%/}"
              TARGET="s3:${S3_BUCKET}/${PREFIX}"

              echo "[4/5] 上传 RDB 到 ${TARGET}/..."
              rclone copyto "${DUMP}" "${TARGET}/${REDIS_BACKUP_OBJECT}" \
                --s3-no-check-bucket \
                --contimeout 10s \
                --timeout 5m \
                --low-level-retries 3 \
                --retries 3 \
                --transfers 1 \
                --checkers 1 \
                --progress \
                -vv

              echo "[5/5] 上传后列表："
              rclone lsf "${TARGET}/" --s3-no-check-bucket
              echo "--- Redis RDB 导出 + 上传完成 ---"
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

如果原本已经挂载好 S3 接入点，只需配置源集群 `values.yaml` 中相同的 S3 接入点配置，详见 [外部 S3 集成配置](./configuration.md#外部-s3-集成integrations-s3)。

#### 情况 2：原始集群使用 MinIO 挂载 PVC

MinIO 数据需要通过 `rclone sync` 同步到公有云对象存储。此 Job 直连 MinIO Service。

::: info 前置条件

- `swanlab-cloud-s3-config` 中的 `MINIO_ENDPOINT` 指向现役 MinIO 的集群内 Service 地址
- `swanlab-cloud-s3-secret` 中的 `MINIO_AK/SK` 为现役 MinIO 的 root 凭据
- 业务已停写，保证读到的对象集稳定
  :::

首先创建 S3 迁移专用配置：

::: details config-s3-export.yaml

```yaml
## MinIO → 云端 S3 迁移专用配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-cloud-s3-config
  namespace: <SOURCE_NAMESPACE>
data:
  ## MinIO 本地源（直连 Service，同 namespace 用短名）
  MINIO_ENDPOINT: "http://swanlab-self-hosted-s3:9000" # ⚠️ 现役 MinIO Service
  MINIO_PUBLIC_BUCKET: "swanlab-public" # MinIO 中 public 桶名
  MINIO_PRIVATE_BUCKET: "swanlab-private" # MinIO 中 private 桶名
  ## 云端 S3 目标（支持 bucket/path 格式或独立桶）
  CLOUD_S3_ENDPOINT: "<CLOUD_S3_ENDPOINT>" # ⚠️ 云端 S3 endpoint
  CLOUD_S3_REGION: "<CLOUD_S3_REGION>" # ⚠️ 云端 S3 region
  CLOUD_S3_PUBLIC_DEST: "<CLOUD_S3_PUBLIC_DEST>" # ⚠️ 云端 public 桶/路径
  CLOUD_S3_PRIVATE_DEST: "<CLOUD_S3_PRIVATE_DEST>" # ⚠️ 云端 private 桶/路径
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-cloud-s3-secret
  namespace: <SOURCE_NAMESPACE>
type: Opaque
stringData:
  MINIO_AK: "<MINIO_AK>" # ⚠️ 源集群 MinIO AK
  MINIO_SK: "<MINIO_SK>" # ⚠️ 源集群 MinIO SK
  CLOUD_S3_AK: "<CLOUD_S3_AK>" # ⚠️ 云端 S3 AK
  CLOUD_S3_SK: "<CLOUD_S3_SK>" # ⚠️ 云端 S3 SK
```

:::

```bash
kubectl apply -f config-s3-export.yaml
```

::: details export-s3 Job YAML

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-migrate-s3-export
  namespace: <SOURCE_NAMESPACE> # ⚠️ 必填：【源集群】K8s 命名空间
  labels:
    swanlab: minio
spec:
  backoffLimit: 0 # ⚠️ TB 级迁移失败不自动重跑
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: rclone-worker
          image: repo.swanlab.cn/public/s3-migrator:bookworm-slim
          imagePullPolicy: IfNotPresent
          resources:
            limits:
              memory: 4Gi
          env:
            - name: MINIO_AK
              valueFrom:
                secretKeyRef:
                  name: swanlab-cloud-s3-secret
                  key: MINIO_AK
            - name: MINIO_SK
              valueFrom:
                secretKeyRef:
                  name: swanlab-cloud-s3-secret
                  key: MINIO_SK
            - name: MINIO_ENDPOINT
              valueFrom:
                configMapKeyRef:
                  name: swanlab-cloud-s3-config
                  key: MINIO_ENDPOINT
            - name: MINIO_PUBLIC_BUCKET
              valueFrom:
                configMapKeyRef:
                  name: swanlab-cloud-s3-config
                  key: MINIO_PUBLIC_BUCKET
            - name: MINIO_PRIVATE_BUCKET
              valueFrom:
                configMapKeyRef:
                  name: swanlab-cloud-s3-config
                  key: MINIO_PRIVATE_BUCKET
            - name: CLOUD_S3_AK
              valueFrom:
                secretKeyRef:
                  name: swanlab-cloud-s3-secret
                  key: CLOUD_S3_AK
            - name: CLOUD_S3_SK
              valueFrom:
                secretKeyRef:
                  name: swanlab-cloud-s3-secret
                  key: CLOUD_S3_SK
            - name: CLOUD_S3_ENDPOINT
              valueFrom:
                configMapKeyRef:
                  name: swanlab-cloud-s3-config
                  key: CLOUD_S3_ENDPOINT
            - name: CLOUD_S3_REGION
              valueFrom:
                configMapKeyRef:
                  name: swanlab-cloud-s3-config
                  key: CLOUD_S3_REGION
            - name: CLOUD_S3_PUBLIC_DEST
              valueFrom:
                configMapKeyRef:
                  name: swanlab-cloud-s3-config
                  key: CLOUD_S3_PUBLIC_DEST
            - name: CLOUD_S3_PRIVATE_DEST
              valueFrom:
                configMapKeyRef:
                  name: swanlab-cloud-s3-config
                  key: CLOUD_S3_PRIVATE_DEST
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -e
              set -o pipefail

              echo "[1/4] 探活现役 MinIO（${MINIO_ENDPOINT}）..."
              for i in $(seq 1 30); do
                if curl -sf -o /dev/null --max-time 2 "${MINIO_ENDPOINT}/minio/health/live"; then
                  echo "  MinIO 可达"
                  break
                fi
                [ "$i" -eq 30 ] && { echo "ERROR: MinIO 不可达"; exit 1; }
                echo "  probing minio... ($i/30)"
                sleep 3
              done

              echo "[2/4] 配置 rclone S3..."
              export RCLONE_CONFIG_LOCALMINIO_TYPE=s3
              export RCLONE_CONFIG_LOCALMINIO_PROVIDER=Minio
              export RCLONE_CONFIG_LOCALMINIO_ACCESS_KEY_ID="${MINIO_AK}"
              export RCLONE_CONFIG_LOCALMINIO_SECRET_ACCESS_KEY="${MINIO_SK}"
              export RCLONE_CONFIG_LOCALMINIO_ENDPOINT="${MINIO_ENDPOINT}"
              export RCLONE_CONFIG_LOCALMINIO_FORCE_PATH_STYLE=true

              export RCLONE_CONFIG_CLOUDS3_TYPE=s3
              export RCLONE_CONFIG_CLOUDS3_PROVIDER=Other
              export RCLONE_CONFIG_CLOUDS3_ENV_AUTH=false
              export RCLONE_CONFIG_CLOUDS3_ACCESS_KEY_ID="${CLOUD_S3_AK}"
              export RCLONE_CONFIG_CLOUDS3_SECRET_ACCESS_KEY="${CLOUD_S3_SK}"
              export RCLONE_CONFIG_CLOUDS3_ENDPOINT="${CLOUD_S3_ENDPOINT}"
              export RCLONE_CONFIG_CLOUDS3_REGION="${CLOUD_S3_REGION}"
              export RCLONE_CONFIG_CLOUDS3_NO_CHECK_BUCKET=true
              export RCLONE_CONFIG_CLOUDS3_FORCE_PATH_STYLE=false

              echo "[3/4] 同步 public..."
              rclone sync localminio:${MINIO_PUBLIC_BUCKET} clouds3:${CLOUD_S3_PUBLIC_DEST} \
                --transfers 16 --checkers 8 --buffer-size 256M \
                --multi-thread-streams 4 --s3-upload-concurrency 4 \
                --retries 3 --retries-sleep 5s --progress -vv

              echo "[4/4] 同步 private..."
              rclone sync localminio:${MINIO_PRIVATE_BUCKET} clouds3:${CLOUD_S3_PRIVATE_DEST} \
                --transfers 16 --checkers 8 --buffer-size 256M \
                --multi-thread-streams 4 --s3-upload-concurrency 4 \
                --retries 3 --retries-sleep 5s --progress -vv

              echo "=== S3 Export 完成 ==="
```

:::

```bash
kubectl apply -f export-s3.yaml -n <your_namespace>

# 查看 rclone 日志
kubectl logs -f job/swanlab-migrate-s3-export -n <your_namespace>
```

### 5. 导入 DB 数据

- 操作位置：<span style="color: red"><strong>目标集群</strong></span>

与导出类似，每个数据库有独立的导入 Job，可并行执行。

::: info 导入架构说明

- **PostgreSQL**：Job 从 S3 中转桶下载 dump 文件后，通过 `pg_restore` 连接目标集群 PG Service 恢复。不挂载 PVC，不需要 scale down PG。
- **ClickHouse**：Job 连接目标集群 CH Service，使用原生 `RESTORE DATABASE ... FROM S3()` 命令。恢复前自动 `DROP DATABASE IF EXISTS` 再重建空库（不可逆操作）。
- **Redis**：Job 从 S3 下载 RDB 落盘到目标 Redis PVC 的 `/data/dump.rdb`，导入完成后需手动 scale 回 Redis Deployment，Redis 启动时自动加载。

:::

::: warning ClickHouse 恢复注意

- CH 的 `RESTORE` 没有 `pg_restore --clean` 的等价选项，`allow_non_empty_tables=1` 是 merge 不是 replace，会导致行数翻倍，因此脚本在 RESTORE 前执行 `DROP DATABASE IF EXISTS`
- 若目标库单表 > 50GB，CH 默认禁止 DROP，需先在 CH config 里设 `<max_table_size_to_drop>0</max_table_size_to_drop>` 解锁
  :::

::: details import-postgres

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-postgres
  namespace: <TARGET_NAMESPACE> # ⚠️ 必填：【目标集群】K8s 命名空间
  labels:
    swanlab: postgres
spec:
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      volumes:
        - name: tmp
          emptyDir:
            sizeLimit: 20Gi
      containers:
        - name: swanlab-import-postgres
          image: repo.swanlab.cn/public/pg-migrator:16.1
          imagePullPolicy: IfNotPresent
          resources:
            limits:
              memory: 4Gi
              ephemeral-storage: 20Gi
          volumeMounts:
            - name: tmp
              mountPath: /tmp
          env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: swanlab-pg-backup-secret
                  key: PG_PASSWORD
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - configMapRef:
                name: swanlab-pg-backup-config
            - secretRef:
                name: swanlab-backup-storage-secret
          command:
            - /bin/bash
            - -c
            - |
              set -e
              set -o pipefail
              echo "[1/4] 配置 rclone S3..."
              export RCLONE_CONFIG_S3_TYPE=s3
              export RCLONE_CONFIG_S3_PROVIDER=Other
              export RCLONE_CONFIG_S3_ENDPOINT="${S3_ENDPOINT}"
              export RCLONE_CONFIG_S3_REGION="${S3_REGION}"
              export RCLONE_CONFIG_S3_ACCESS_KEY_ID="${S3_AK}"
              export RCLONE_CONFIG_S3_SECRET_ACCESS_KEY="${S3_SK}"
              export RCLONE_CONFIG_S3_NO_CHECK_BUCKET=true
              export RCLONE_CONFIG_S3_FORCE_PATH_STYLE="${S3_FORCE_PATH_STYLE:-false}"

              PREFIX="${S3_PATH_PREFIX%/}"
              TARGET="s3:${S3_BUCKET}/${PREFIX}"
              DUMP=/tmp/${PG_BACKUP_OBJECT}

              echo "[2/4] 从 ${TARGET}/ 下载..."
              rclone copyto "${TARGET}/${PG_BACKUP_OBJECT}" "${DUMP}" \
                --s3-no-check-bucket \
                --contimeout 10s \
                --timeout 5m \
                --low-level-retries 3 \
                --retries 3 \
                --transfers 1 \
                --checkers 1 \
                --progress \
                -vv

              echo "[3/4] 连接检查..."
              ls -lh "${DUMP}"
              pg_isready -h "${PG_HOST}" -p "${PG_PORT:-5432}" -U "${PG_USER}" -d "${PG_DB}"

              echo "[4/4] pg_restore 恢复中..."
              pg_restore \
                -h "${PG_HOST}" \
                -p "${PG_PORT:-5432}" \
                -U "${PG_USER}" \
                -d "${PG_DB}" \
                --no-owner \
                --no-acl \
                --clean \
                --if-exists \
                -v \
                "${DUMP}"
              echo "--- PostgreSQL 恢复完成 ---"
```

:::

::: details import-clickhouse

```yaml
## ============================================================
## ClickHouse 原生 RESTORE 导入 Job
## ============================================================
## ⚠️ 清库行为（不可逆）：
##   - RESTORE 前会 DROP DATABASE IF EXISTS 再重建空库
##   - 若目标库单表 > 50GB，需预先解锁 DROP 保护
## ============================================================
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-clickhouse
  namespace: <TARGET_NAMESPACE> # ⚠️ 必填：【目标集群】K8s 命名空间
  labels:
    swanlab: clickhouse
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: swanlab-import-clickhouse
          image: repo.swanlab.cn/self-hosted/clickhouse-server:24.3
          imagePullPolicy: IfNotPresent
          env:
            - name: TZ
              value: "UTC"
            - name: CLICKHOUSE_USER
              valueFrom:
                secretKeyRef:
                  name: swanlab-ch-backup-secret
                  key: CLICKHOUSE_USER
            - name: CLICKHOUSE_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: swanlab-ch-backup-secret
                  key: CLICKHOUSE_PASSWORD
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - configMapRef:
                name: swanlab-ch-backup-config
            - secretRef:
                name: swanlab-backup-storage-secret
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -e

              CH_CLIENT=(clickhouse-client
                --host "${CLICKHOUSE_HOST}"
                --user "${CLICKHOUSE_USER}"
                --password "${CLICKHOUSE_PASSWORD}")

              echo "[1/5] 连接目标 ClickHouse 并清理目标库..."
              "${CH_CLIENT[@]}" --query "SELECT version(), uptime()"

              ## 清库：DROP + CREATE 空库
              EXISTING_TABLES=$("${CH_CLIENT[@]}" --query "SHOW TABLES FROM ${CLICKHOUSE_DB}" 2>/dev/null || true)
              if [ -n "${EXISTING_TABLES}" ]; then
                echo "  目标库 ${CLICKHOUSE_DB} 非空，执行 DROP..."
                "${CH_CLIENT[@]}" --query "DROP DATABASE IF EXISTS ${CLICKHOUSE_DB}"
              fi
              "${CH_CLIENT[@]}" --query "CREATE DATABASE IF NOT EXISTS ${CLICKHOUSE_DB}"
              echo "  目标库 ${CLICKHOUSE_DB} 已就绪（空）"

              S3_URL="https://${S3_BUCKET}.${S3_ENDPOINT}"
              PREFIX="${S3_PATH_PREFIX%/}"
              BACKUP_PATH="${PREFIX:+${PREFIX}/}${CLICKHOUSE_BACKUP_PATH}"

              echo "[2/5] 发起 RESTORE DATABASE ${CLICKHOUSE_DB} ← s3://${S3_BUCKET}/${BACKUP_PATH}..."
              RESTORE_RESULT=$("${CH_CLIENT[@]}" --format=TabSeparated --query "
                RESTORE DATABASE ${CLICKHOUSE_DB}
                FROM S3(
                  '${S3_URL}/${BACKUP_PATH}',
                  '${S3_AK}',
                  '${S3_SK}'
                )
                ASYNC
              ")
              RESTORE_ID=$(echo "${RESTORE_RESULT}" | awk '{print $1}')
              echo "  恢复任务已提交，id=${RESTORE_ID}"

              echo "[3/5] 轮询恢复进度..."
              while true; do
                ROW=$("${CH_CLIENT[@]}" --format=TabSeparated --query "
                  SELECT status, num_files, uncompressed_size, compressed_size, error
                  FROM system.backups
                  WHERE id='${RESTORE_ID}'
                ")
                STATUS=$(echo "${ROW}" | awk -F'\t' '{print $1}')
                NUM_FILES=$(echo "${ROW}" | awk -F'\t' '{print $2}')
                UNCOMPRESSED=$(echo "${ROW}" | awk -F'\t' '{print $3}')
                COMPRESSED=$(echo "${ROW}" | awk -F'\t' '{print $4}')
                ERROR=$(echo "${ROW}" | awk -F'\t' '{print $5}')

                UNCOMPRESSED_H=$(awk "BEGIN{printf \"%.2f GB\", ${UNCOMPRESSED:-0}/1073741824}")
                COMPRESSED_H=$(awk "BEGIN{printf \"%.2f GB\", ${COMPRESSED:-0}/1073741824}")

                echo "  [$(date +%H:%M:%S)] status=${STATUS} files=${NUM_FILES} uncompressed=${UNCOMPRESSED_H} compressed=${COMPRESSED_H}"

                case "${STATUS}" in
                  RESTORED)
                    echo "--- ClickHouse 原生恢复完成 ---"
                    break
                    ;;
                  RESTORE_FAILED)
                    echo "ERROR: 恢复失败：${ERROR}"
                    exit 1
                    ;;
                esac
                sleep 10
              done

              echo "[4/5] 恢复后表列表及大小..."
              "${CH_CLIENT[@]}" --query "SHOW TABLES FROM ${CLICKHOUSE_DB}"
              "${CH_CLIENT[@]}" --query "
                SELECT
                  table,
                  formatReadableSize(total_bytes) AS size,
                  formatReadableQuantity(total_rows) AS rows
                FROM system.tables
                WHERE database = '${CLICKHOUSE_DB}'
                ORDER BY total_bytes DESC
              "

              echo "[5/5] 恢复后行数统计（与 export 侧对比校验）..."
              "${CH_CLIENT[@]}" --query "
                SELECT
                  '${CLICKHOUSE_DB}' AS database,
                  sum(rows) AS total_rows,
                  formatReadableQuantity(sum(rows)) AS rows_readable
                FROM system.parts
                WHERE database='${CLICKHOUSE_DB}' AND active
              "
              echo "--- 导入侧完成，请与 export 侧输出的行数对比 ---"
```

:::

::: details import-redis

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-redis
  namespace: <TARGET_NAMESPACE> # ⚠️ 必填：【目标集群】K8s 命名空间
  labels:
    swanlab: redis
spec:
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      volumes:
        - name: swanlab-redis-data
          persistentVolumeClaim:
            claimName: swanlab-redis-pvc # ⚠️ 必填：【目标集群】Redis PVC 名称
      containers:
        - name: swanlab-import-redis
          image: repo.swanlab.cn/public/redis-migrator:7.4.0-v8
          imagePullPolicy: IfNotPresent
          resources:
            limits:
              memory: 4Gi
          envFrom:
            - configMapRef:
                name: swanlab-backup-storage-config
            - configMapRef:
                name: swanlab-redis-backup-config
            - secretRef:
                name: swanlab-backup-storage-secret
          volumeMounts:
            - name: swanlab-redis-data
              mountPath: /data
          command:
            - /bin/bash
            - -c
            - |
              set -e
              set -o pipefail
              echo "[1/4] 配置 rclone S3..."
              export RCLONE_CONFIG_S3_TYPE=s3
              export RCLONE_CONFIG_S3_PROVIDER=Other
              export RCLONE_CONFIG_S3_ENDPOINT="${S3_ENDPOINT}"
              export RCLONE_CONFIG_S3_REGION="${S3_REGION}"
              export RCLONE_CONFIG_S3_ACCESS_KEY_ID="${S3_AK}"
              export RCLONE_CONFIG_S3_SECRET_ACCESS_KEY="${S3_SK}"
              export RCLONE_CONFIG_S3_NO_CHECK_BUCKET=true
              export RCLONE_CONFIG_S3_FORCE_PATH_STYLE="${S3_FORCE_PATH_STYLE:-false}"

              : "${REDIS_BACKUP_OBJECT:?REDIS_BACKUP_OBJECT 未配置}"
              PREFIX="${S3_PATH_PREFIX%/}"
              TARGET="s3:${S3_BUCKET}/${PREFIX}"
              DUMP=/tmp/${REDIS_BACKUP_OBJECT}

              echo "[2/4] 从 ${TARGET}/ 下载..."
              rclone copyto "${TARGET}/${REDIS_BACKUP_OBJECT}" "${DUMP}" \
                --s3-no-check-bucket \
                --contimeout 10s \
                --timeout 5m \
                --low-level-retries 3 \
                --retries 3 \
                --transfers 1 \
                --checkers 1 \
                --progress \
                -vv

              echo "[3/4] 落盘至 /data/dump.rdb..."
              ls -lh "${DUMP}"
              cp -f "${DUMP}" /data/dump.rdb
              # 清理 AOF 残留，确保 Redis 启动时优先从 dump.rdb 加载
              rm -f /data/appendonly.aof* 2>/dev/null || true
              rm -rf /data/appendonlydir 2>/dev/null || true
              ls -lh /data

              echo "[4/4] --- Redis 恢复完成 ---"
              echo "⚠️ 本 Job 完成后需 scale 回 Redis Deployment，Redis 启动时会自动加载 dump.rdb"
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
# 恢复数据库服务（主要是 Redis） (replicas 必须为 1)
kubectl scale deployment swanlab-self-hosted-redis --replicas=1 -n <your_namespace>

# 确认数据库就绪
kubectl get pods -n <your_namespace> -w
```

```bash [2. 恢复应用层]
# 先恢复副本，再按需扩容
kubectl scale deploy/swanlab-self-hosted-house deploy/swanlab-self-hosted-server --replicas=1 -n <your_namespace>
```

```bash [4. 恢复网关]
# 恢复网关
kubectl scale deploy/swanlab-self-hosted --replicas=2 -n <your_namespace>
```

:::

恢复后可以观测 pod 健康状况与线上服务验证数据恢复情况。

## 🧹 Job 清理

原始和目标集群的 Job 完成后 **24 小时自动清理**（`ttlSecondsAfterFinished: 86400`）。如需手动清理：

```bash
# 源集群
kubectl delete job swanlab-export-postgres swanlab-export-clickhouse swanlab-export-redis -n <SOURCE_NAMESPACE>

# 目标集群
kubectl delete job swanlab-import-postgres swanlab-import-clickhouse swanlab-import-redis -n <TARGET_NAMESPACE>
```
