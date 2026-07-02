# Data Migration

> This page describes the complete process of full data migration between different Kubernetes clusters for the SwanLab self-hosted version.

## Migration Process Diagram

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/cross-cluster-migration-v2.drawio.svg"/>

The cross-cluster data migration process for the SwanLab self-hosted service database includes three core areas:

- **Source Cluster**: Data export side, containing databases and object storage.
- **Transit Storage**: S3 object storage as a temporary transit station.
- **Target Cluster**: Data receiving side, completing the restore.

| Data Type                   | Migration Method                                                | Description                                                      |
| --------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------- |
| PostgreSQL                  | `pg_dump` → S3 transit → `pg_restore`                           | Client connects directly to Service, no PVC mounting required    |
| ClickHouse                  | Native `BACKUP DATABASE` → S3 → `RESTORE DATABASE`              | Client connects directly to Service, native S3 backup/restore    |
| Redis                       | `redis-cli --rdb` → S3 transit → write to PVC                   | Export connects directly to Service, import writes to target PVC |
| Object Storage (MinIO / S3) | `rclone sync` direct to cloud S3 or direct cloud S3 integration | No migration needed with storage-compute separation              |

**Legend**:

- 🔵 **Phase 1: Export** — Source cluster exports data to S3 transit bucket (DB Export + S3 Sync)
- 🟢 **Phase 2: Import** — Database data imported from S3 to new cluster (DB Import, physical migration)
- 🟢 **Phase 2: Direct Use** (dashed line) — Object storage is not moved, new cluster directly accesses public and private object storage buckets through the same `values.yaml` configuration

## 🧾 Prerequisites

### Resource Preparation

- S3-compatible object storage bucket with available space at least **1.1 times** greater than the total storage of ClickHouse + PostgreSQL + Redis
- Target cluster requires an additional deployment of a **brand new unactivated** SwanLab service, and must mount cloud disks
- If the original and target cluster versions are inconsistent, it is recommended to first upgrade the chart version of the original cluster

### Image Preparation

| Image                                                | Purpose              | Description                                       |
| ---------------------------------------------------- | -------------------- | ------------------------------------------------- |
| `repo.swanlab.cn/public/pg-migrator:16.1`            | PostgreSQL migration | Contains pg_dump / pg_restore / rclone            |
| `repo.swanlab.cn/self-hosted/clickhouse-server:24.3` | ClickHouse migration | Contains clickhouse-client, native BACKUP/RESTORE |
| `repo.swanlab.cn/public/redis-migrator:7.4.0-v8`     | Redis migration      | Contains redis-cli / rclone                       |
| `repo.swanlab.cn/public/s3-migrator:bookworm-slim`   | S3 migration         | Contains rclone, for MinIO → cloud S3 sync        |

### Permission Preparation

- Have write permissions for both the original and target clusters (they can be the same cluster, but each must have an independent SwanLab service)
- Prepare `access_key` and `secret_key` with read-write permissions for the object storage bucket
- ClickHouse user must have `BACKUP` / `RESTORE` permissions
- PostgreSQL user must have `pg_dump` / `pg_restore` permissions

### Configuration Variables

Please confirm the following information before use:

| Prepared | Placeholder Variable                    | Description                                                             |
| -------- | --------------------------------------- | ----------------------------------------------------------------------- |
| ✅       | `SOURCE_NAMESPACE` / `TARGET_NAMESPACE` | Namespaces of the source and target clusters                            |
| ✅       | `S3_REGION`                             | Region of the object storage bucket used for backup                     |
| ✅       | `S3_BUCKET`                             | Bucket name                                                             |
| ✅       | `S3_ENDPOINT`                           | S3 format object storage Endpoint, e.g., `tos-s3-cn-beijing.volces.com` |
| ✅       | `S3_AK` / `S3_SK`                       | Object storage writable credentials                                     |
| ✅       | `S3_PATH_PREFIX`                        | Backup path prefix in S3, default `origin-backup-datas`                 |

## 🪜 Operation Steps

:::warning

**⚠️ Services must be stopped before migration!** During data migration, **both SwanLab application services must be kept in a stopped state**, with the migration Jobs executing in between. Otherwise, migration will fail due to data write state inconsistency issues.
:::

### 1. Modify Configuration Files

- Operation location: <span style="color: red"><strong>Source Cluster, Target Cluster</strong></span>

Migration configurations are split into independent ConfigMaps / Secrets per component, and must be **created separately on both the source and target clusters**.

#### Source Cluster Configuration (config-export.yaml)

The source cluster requires the following configurations:

- `swanlab-backup-storage-config` / `swanlab-backup-storage-secret` — S3 transit bucket credentials
- `swanlab-pg-backup-config` / `swanlab-pg-backup-secret` — PG connection info
- `swanlab-ch-backup-config` / `swanlab-ch-backup-secret` — CH connection info
- `swanlab-redis-backup-config` — Redis connection info

::: details config-export.yaml

```yaml
## ============================================================
## Source Cluster: Migration Configuration (export only)
## ============================================================

## Part 1: Backup Transit Bucket (shared by DB export/import)
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-backup-storage-config
  namespace: <SOURCE_NAMESPACE> # ⚠️ Required: [Source Cluster] K8s namespace
data:
  S3_ENDPOINT: "<BACKUP_S3_ENDPOINT>" # ⚠️ Backup transit bucket endpoint, no bucket prefix
  S3_REGION: "<BACKUP_S3_REGION>" # ⚠️ Backup transit bucket region
  S3_BUCKET: "<BACKUP_S3_BUCKET>" # ⚠️ Backup transit bucket name
  S3_PATH_PREFIX: "<S3_PATH_PREFIX>" # ⚠️ Path prefix within the bucket (must match import side)
  S3_FORCE_PATH_STYLE: "false"
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-backup-storage-secret
  namespace: <SOURCE_NAMESPACE>
type: Opaque
stringData:
  S3_AK: "<BACKUP_S3_AK>" # ⚠️ Backup transit bucket AK
  S3_SK: "<BACKUP_S3_SK>" # ⚠️ Backup transit bucket SK
---
## Part 2: Redis Connection Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-redis-backup-config
  namespace: <SOURCE_NAMESPACE>
data:
  REDIS_HOST: "swanlab-self-hosted-redis" # ⚠️ [Source Cluster] Redis Service name
  REDIS_PORT: "6379" # ⚠️ Redis port
  REDIS_BACKUP_OBJECT: "redis-restore.rdb" # ⚠️ S3 transit object name (must match import side)
---
## Part 3: PostgreSQL Connection Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-pg-backup-config
  namespace: <SOURCE_NAMESPACE>
data:
  PG_HOST: "swanlab-self-hosted-postgres" # ⚠️ [Source Cluster] PG Service name
  PG_USER: "swanlab" # ⚠️ PG username
  PG_DB: "app" # ⚠️ PG database name
  PG_BACKUP_OBJECT: "postgres-restore.dump" # ⚠️ S3 transit object name (must match import side)
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-pg-backup-secret
  namespace: <SOURCE_NAMESPACE>
type: Opaque
stringData:
  PG_PASSWORD: "<PG_PASSWORD>" # ⚠️ Must match postgres password in Helm values
---
## Part 4: ClickHouse Connection Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-ch-backup-config
  namespace: <SOURCE_NAMESPACE>
data:
  CLICKHOUSE_HOST: "swanlab-self-hosted-clickhouse" # ⚠️ [Source Cluster] CH Service name
  CLICKHOUSE_DB: "app" # ⚠️ Database name to back up
  CLICKHOUSE_BACKUP_PATH: "clickhouse-backup" # ⚠️ Backup sub-path within S3 transit bucket (must match import side)
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-ch-backup-secret
  namespace: <SOURCE_NAMESPACE>
type: Opaque
stringData:
  CLICKHOUSE_USER: "<CLICKHOUSE_USER>" # ⚠️ Must match CH username in Helm values
  CLICKHOUSE_PASSWORD: "<CLICKHOUSE_PASSWORD>" # ⚠️ Must match CH password in Helm values
```

:::

::: tip
If the source cluster uses built-in MinIO storage, you also need to create `swanlab-cloud-s3-config` / `swanlab-cloud-s3-secret`. See the [Export S3 Data](#_4-export-s3-data-optional) section for details.
:::

```bash [Source Cluster]
kubectl apply -f config-export.yaml
```

#### Target Cluster Configuration (config-import.yaml)

The target cluster requires the following configurations:

- `swanlab-backup-storage-config` / `swanlab-backup-storage-secret` — S3 transit bucket credentials
- `swanlab-pg-backup-config` / `swanlab-pg-backup-secret` — Target PG connection info
- `swanlab-ch-backup-config` / `swanlab-ch-backup-secret` — Target CH connection info
- `swanlab-redis-backup-config` — Redis object name

::: details config-import.yaml

```yaml
## ============================================================
## Target Cluster: Migration Configuration (import only)
## ============================================================
## Only DB data (PG/CH/Redis) needs to be imported from the transit bucket
## MinIO data goes directly to the cloud, does not pass through the transit bucket, no import config needed
## ============================================================

apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-backup-storage-config
  namespace: <TARGET_NAMESPACE> # ⚠️ Required: [Target Cluster] K8s namespace
data:
  S3_ENDPOINT: "<BACKUP_S3_ENDPOINT>" # ⚠️ Backup transit bucket endpoint, no bucket prefix
  S3_REGION: "<BACKUP_S3_REGION>" # ⚠️ Backup transit bucket region
  S3_BUCKET: "<BACKUP_S3_BUCKET>" # ⚠️ Backup transit bucket name
  S3_PATH_PREFIX: "<S3_PATH_PREFIX>" # ⚠️ Path prefix within the bucket
  S3_FORCE_PATH_STYLE: "false"
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-backup-storage-secret
  namespace: <TARGET_NAMESPACE>
type: Opaque
stringData:
  S3_AK: "<BACKUP_S3_AK>" # ⚠️ Backup transit bucket AK
  S3_SK: "<BACKUP_S3_SK>" # ⚠️ Backup transit bucket SK
---
## Redis object name config (import only downloads and writes to disk, does not connect to Redis)
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-redis-backup-config
  namespace: <TARGET_NAMESPACE>
data:
  REDIS_BACKUP_OBJECT: "redis-restore.rdb" # ⚠️ Must match export side
---
## PostgreSQL connection config (import needs to connect to target cluster PG to execute pg_restore)
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-pg-backup-config
  namespace: <TARGET_NAMESPACE>
data:
  ## Choose one migration target:
  ##   A) Built-in PG: fill in K8s Service name, e.g., swanlab-self-hosted-postgres
  ##   B) Cloud RDS: fill in RDS domain, e.g., rm-xxx.pg.rds.aliyuncs.com
  ##      ⚠️ When choosing B, add the import Job's cluster egress IP to the RDS whitelist
  PG_HOST: "<PG_HOST>" # ⚠️ K8s Service name or cloud RDS domain
  PG_PORT: "5432" # ⚠️ PG port (use actual port for cloud RDS)
  PG_USER: "<PG_USER>" # ⚠️ PG username
  PG_DB: "app" # ⚠️ PG database name
  PG_BACKUP_OBJECT: "postgres-restore.dump" # ⚠️ Must match export side
  PGSSLMODE: "prefer" # ⚠️ Cloud RDS usually requires "require" or higher
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-pg-backup-secret
  namespace: <TARGET_NAMESPACE>
type: Opaque
stringData:
  PG_PASSWORD: "<PG_PASSWORD>" # ⚠️ Must match postgres password in target cluster Helm values
---
## ClickHouse connection config (Job connects to target CH via client to execute RESTORE)
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-ch-backup-config
  namespace: <TARGET_NAMESPACE>
data:
  CLICKHOUSE_HOST: "<CLICKHOUSE_HOST>" # ⚠️ [Target Cluster] CH Service name
  CLICKHOUSE_DB: "app" # ⚠️ Database name to restore (must match export side)
  CLICKHOUSE_BACKUP_PATH: "clickhouse-backup" # ⚠️ Must match export side
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-ch-backup-secret
  namespace: <TARGET_NAMESPACE>
type: Opaque
stringData:
  CLICKHOUSE_USER: "<CLICKHOUSE_USER>" # ⚠️ Target cluster CH username
  CLICKHOUSE_PASSWORD: "<CLICKHOUSE_PASSWORD>" # ⚠️ Target cluster CH password
```

:::

```bash [Target Cluster]
kubectl apply -f config-import.yaml
```

Also modify the following fields in the export/import Job YAML:

- `namespace`: Corresponding K8s namespace
- `claimName`: Corresponding PVC name (required for Redis import)
- `nodeSelector`: Needs to specify a node for local deployment scenarios

### 2. Stop Services

- Operation location: <span style="color: red"><strong>Source Cluster, Target Cluster</strong></span>

Be sure to stop services in order.

::: code-group

```bash [1. Stop Gateway]
# Cut off all external traffic
kubectl scale deploy/swanlab-self-hosted --replicas=0 -n <your_namespace>
```

```bash [2. Stop Application Layer]
# Stop backend core service
kubectl scale deploy/swanlab-self-hosted-server --replicas=0 -n <your_namespace>
# Stop backend metrics OLAP service
kubectl scale deploy/swanlab-self-hosted-house --replicas=0 -n <your_namespace>
```

```bash [3. Wait for Vector to Drain]
# Wait for the buffer to consume (watch logs for no new writes, then Ctrl+C)
kubectl logs -f swanlab-self-hosted-vector-0 -n <your_namespace> --tail=20
kubectl logs -f swanlab-self-hosted-vector-1 -n <your_namespace> --tail=20

```

:::

:::tip
For the **target cluster's** Redis database, since it needs to be restored using an RDB snapshot, you need to separately stop the target cluster's Redis service:

`kubectl scale deploy/swanlab-self-hosted-redis --replicas=0 -n <your_namespace>`
:::

### 3. Export DB Data

- Operation location: <span style="color: red"><strong>Source Cluster</strong></span>

Each database migration is encapsulated as an independent Job and can be executed in parallel.

::: info Export Notes

- **PostgreSQL**: Job connects directly to PG Service as a client, uses `pg_dump -Fc` (custom format) to export, then uploads to S3 transit bucket via `rclone`.
- **ClickHouse**: Job connects directly to CH Service as a client, uses native `BACKUP DATABASE ... TO S3()` command. The CH server guarantees parts consistency.
- **Redis**: Job connects directly to Redis Service as a client, uses `redis-cli --rdb` to export RDB, then uploads via `rclone`.

:::

::: details export-postgres

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-postgres
  namespace: <SOURCE_NAMESPACE> # ⚠️ Required: [Source Cluster] K8s namespace
  labels:
    swanlab: postgres
spec:
  backoffLimit: 1
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      ## ⚠️ /tmp uses emptyDir + sizeLimit to prevent dump from filling the node root disk and triggering node-level eviction.
      ##    Exceeding sizeLimit only kills this Pod, not affecting neighbors.
      volumes:
        - name: tmp
          emptyDir:
            sizeLimit: 20Gi # ⚠️ ~10-15 GB dump; adjust based on actual DB size, do not exceed node allocatable
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

              echo "[1/5] Configuring rclone S3..."
              export RCLONE_CONFIG_S3_TYPE=s3
              export RCLONE_CONFIG_S3_PROVIDER=Other
              export RCLONE_CONFIG_S3_ENDPOINT="${S3_ENDPOINT}"
              export RCLONE_CONFIG_S3_REGION="${S3_REGION}"
              export RCLONE_CONFIG_S3_ACCESS_KEY_ID="${S3_AK}"
              export RCLONE_CONFIG_S3_SECRET_ACCESS_KEY="${S3_SK}"
              export RCLONE_CONFIG_S3_NO_CHECK_BUCKET=true
              export RCLONE_CONFIG_S3_FORCE_PATH_STYLE="${S3_FORCE_PATH_STYLE:-false}"

              echo "[2/5] Connection check..."
              pg_isready -h "${PG_HOST}" -U "${PG_USER}" -d "${PG_DB}"

              DUMP=/tmp/${PG_BACKUP_OBJECT}
              echo "[3/5] pg_dump exporting (custom format)..."
              (while true; do sleep 5; [ -f "${DUMP}" ] && echo "  Exporting... $(du -h "${DUMP}" | cut -f1)"; done) &
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
              echo "  Export complete: $(du -h "${DUMP}" | cut -f1)"

              ls -lh "${DUMP}"
              pg_restore -l "${DUMP}" | sed -n '1,50p'

              PREFIX="${S3_PATH_PREFIX%/}"
              TARGET="s3:${S3_BUCKET}/${PREFIX}"

              echo "[4/5] Uploading dump to ${TARGET}/..."
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

              echo "[5/5] Post-upload listing:"
              rclone lsf "${TARGET}/" --s3-no-check-bucket
              echo "--- PostgreSQL dump export + upload complete ---"
```

:::

::: details export-clickhouse

```yaml
## ============================================================
## ClickHouse Native BACKUP Export Job
## ============================================================
## How it works: Job acts as a pure client, connects to the running CH Service
##       in the source cluster, and writes data to the S3 transit bucket
##       using the native BACKUP DATABASE command.
##
## ✅ No need to scale down CH Deployment, no PVC mounting.
##    BACKUP executes server-side, CH guarantees parts consistency.
##
## ⚠️ Prerequisites:
##   - Source cluster CH Service is reachable from the Job (port 9000)
##   - CH user has BACKUP permission
##   - Source/target CH versions are consistent
## ============================================================
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-clickhouse
  namespace: <SOURCE_NAMESPACE> # ⚠️ Required: [Source Cluster] K8s namespace
  labels:
    swanlab: clickhouse
spec:
  backoffLimit: 0 # ⚠️ TB-scale backup failure should not auto-retry to avoid data state confusion
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      # # ===== Local deployment scenario: specify node =====
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

              echo "[1/5] Connecting to source ClickHouse..."
              "${CH_CLIENT[@]}" --query "SELECT version(), uptime()"
              echo "  Tables in ${CLICKHOUSE_DB}:"
              "${CH_CLIENT[@]}" --query "SHOW TABLES FROM ${CLICKHOUSE_DB}" || true

              echo "[2/5] Estimating data size..."
              "${CH_CLIENT[@]}" --query "
                SELECT
                  formatReadableSize(sum(bytes_on_disk)) AS disk_size,
                  formatReadableQuantity(sum(rows)) AS total_rows,
                  count() AS parts
                FROM system.parts
                WHERE database='${CLICKHOUSE_DB}' AND active
              "

              ## Construct URL dynamically based on S3_FORCE_PATH_STYLE
              if [ "${S3_FORCE_PATH_STYLE:-false}" = "true" ]; then
                S3_URL="https://${S3_ENDPOINT}/${S3_BUCKET}"
              else
                S3_URL="https://${S3_BUCKET}.${S3_ENDPOINT}"
              fi
              PREFIX="${S3_PATH_PREFIX%/}"
              BACKUP_PATH="${PREFIX:+${PREFIX}/}${CLICKHOUSE_BACKUP_PATH}"

              echo "[3/5] Initiating BACKUP DATABASE ${CLICKHOUSE_DB} → s3://${S3_BUCKET}/${BACKUP_PATH}..."
              ## Submit in ASYNC mode, poll system.backups to track progress
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
              echo "  Backup task submitted, id=${BACKUP_ID}"

              echo "[4/5] Polling backup progress..."
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
                    echo "--- ClickHouse native backup complete ---"
                    echo "Backup location: s3://${S3_BUCKET}/${BACKUP_PATH}"
                    echo "Total files: ${NUM_FILES}, compressed: ${COMPRESSED_H}, uncompressed: ${UNCOMPRESSED_H}"
                    break
                    ;;
                  BACKUP_FAILED)
                    echo "ERROR: Backup failed: ${ERROR}"
                    exit 1
                    ;;
                esac
                sleep 10
              done

              echo "[5/5] Source row count (for import-side verification)..."
              "${CH_CLIENT[@]}" --query "
                SELECT
                  '${CLICKHOUSE_DB}' AS database,
                  sum(rows) AS total_rows,
                  formatReadableQuantity(sum(rows)) AS rows_readable
                FROM system.parts
                WHERE database='${CLICKHOUSE_DB}' AND active
              "
              echo "--- Export complete, please record the above row count for import-side verification ---"
```

:::

::: details export-redis

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-export-redis
  namespace: <SOURCE_NAMESPACE> # ⚠️ Required: [Source Cluster] K8s namespace
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

              : "${REDIS_HOST:?REDIS_HOST not set}"
              : "${REDIS_PORT:?REDIS_PORT not set}"
              : "${REDIS_BACKUP_OBJECT:?REDIS_BACKUP_OBJECT not set}"

              echo "[1/5] Configuring rclone S3..."
              export RCLONE_CONFIG_S3_TYPE=s3
              export RCLONE_CONFIG_S3_PROVIDER=Other
              export RCLONE_CONFIG_S3_ENDPOINT="${S3_ENDPOINT}"
              export RCLONE_CONFIG_S3_REGION="${S3_REGION}"
              export RCLONE_CONFIG_S3_ACCESS_KEY_ID="${S3_AK}"
              export RCLONE_CONFIG_S3_SECRET_ACCESS_KEY="${S3_SK}"
              export RCLONE_CONFIG_S3_NO_CHECK_BUCKET=true
              export RCLONE_CONFIG_S3_FORCE_PATH_STYLE="${S3_FORCE_PATH_STYLE:-false}"

              echo "[2/5] Connection check..."
              redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" PING

              DUMP=/tmp/${REDIS_BACKUP_OBJECT}
              echo "[3/5] redis-cli --rdb exporting..."
              (while true; do sleep 5; [ -f "${DUMP}" ] && echo "  Exporting... $(du -h "${DUMP}" | cut -f1)"; done) &
              PROGRESS_PID=$!
              redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" --rdb "${DUMP}"
              kill $PROGRESS_PID 2>/dev/null || true
              echo "  Export complete: $(du -h "${DUMP}" | cut -f1)"

              ls -lh "${DUMP}"

              PREFIX="${S3_PATH_PREFIX%/}"
              TARGET="s3:${S3_BUCKET}/${PREFIX}"

              echo "[4/5] Uploading RDB to ${TARGET}/..."
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

              echo "[5/5] Post-upload listing:"
              rclone lsf "${TARGET}/" --s3-no-check-bucket
              echo "--- Redis RDB export + upload complete ---"
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

#### Scenario 1: Source Cluster Already Uses S3

If S3 endpoints are already configured, you only need to configure the same S3 endpoint settings in the target cluster's `values.yaml`. See [External S3 Integration Configuration](./configuration.md#external-s3-integrationintegrations-s3) for details.

#### Scenario 2: Source Cluster Uses MinIO with PVC

MinIO data needs to be synced to cloud object storage via `rclone sync`. This Job connects directly to the MinIO Service.

::: info Prerequisites

- `swanlab-cloud-s3-config` `MINIO_ENDPOINT` points to the active MinIO's in-cluster Service address
- `swanlab-cloud-s3-secret` `MINIO_AK/SK` are the active MinIO's root credentials
- Business writes have been stopped, ensuring a stable set of objects for reading
  :::

First create the S3 migration-specific configuration:

::: details config-s3-export.yaml

```yaml
## MinIO → Cloud S3 migration specific configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-cloud-s3-config
  namespace: <SOURCE_NAMESPACE>
data:
  ## MinIO local source (direct Service connection, use short name in same namespace)
  MINIO_ENDPOINT: "http://swanlab-self-hosted-s3:9000" # ⚠️ Active MinIO Service
  MINIO_PUBLIC_BUCKET: "swanlab-public" # MinIO public bucket name
  MINIO_PRIVATE_BUCKET: "swanlab-private" # MinIO private bucket name
  ## Cloud S3 target (supports bucket/path format or separate buckets)
  CLOUD_S3_ENDPOINT: "<CLOUD_S3_ENDPOINT>" # ⚠️ Cloud S3 endpoint
  CLOUD_S3_REGION: "<CLOUD_S3_REGION>" # ⚠️ Cloud S3 region
  CLOUD_S3_PUBLIC_DEST: "<CLOUD_S3_PUBLIC_DEST>" # ⚠️ Cloud public bucket/path
  CLOUD_S3_PRIVATE_DEST: "<CLOUD_S3_PRIVATE_DEST>" # ⚠️ Cloud private bucket/path
---
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-cloud-s3-secret
  namespace: <SOURCE_NAMESPACE>
type: Opaque
stringData:
  MINIO_AK: "<MINIO_AK>" # ⚠️ Source cluster MinIO AK
  MINIO_SK: "<MINIO_SK>" # ⚠️ Source cluster MinIO SK
  CLOUD_S3_AK: "<CLOUD_S3_AK>" # ⚠️ Cloud S3 AK
  CLOUD_S3_SK: "<CLOUD_S3_SK>" # ⚠️ Cloud S3 SK
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
  namespace: <SOURCE_NAMESPACE> # ⚠️ Required: [Source Cluster] K8s namespace
  labels:
    swanlab: minio
spec:
  backoffLimit: 0 # ⚠️ TB-scale migration failure should not auto-retry
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: rclone-worker
          image: repo.swanlab.cn/public/s3-migrator:bookworm-slim
          imagePullPolicy: IfNotPresent
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

              echo "[1/4] Probing active MinIO (${MINIO_ENDPOINT})..."
              for i in $(seq 1 30); do
                if curl -sf -o /dev/null --max-time 2 "${MINIO_ENDPOINT}/minio/health/live"; then
                  echo "  MinIO reachable"
                  break
                fi
                [ "$i" -eq 30 ] && { echo "ERROR: MinIO unreachable"; exit 1; }
                echo "  probing minio... ($i/30)"
                sleep 3
              done

              echo "[2/4] Configuring rclone S3..."
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

              echo "[3/4] Syncing public..."
              rclone sync localminio:${MINIO_PUBLIC_BUCKET} clouds3:${CLOUD_S3_PUBLIC_DEST} \
                --transfers 16 --checkers 8 --buffer-size 128M \
                --multi-thread-streams 4 --s3-upload-concurrency 4 \
                --retries 3 --retries-sleep 5s --progress -vv

              echo "[4/4] Syncing private..."
              rclone sync localminio:${MINIO_PRIVATE_BUCKET} clouds3:${CLOUD_S3_PRIVATE_DEST} \
                --transfers 16 --checkers 8 --buffer-size 128M \
                --multi-thread-streams 4 --s3-upload-concurrency 4 \
                --retries 3 --retries-sleep 5s --progress -vv

              echo "=== S3 Export Complete ==="
```

:::

```bash
kubectl apply -f export-s3.yaml -n <your_namespace>

# View rclone logs
kubectl logs -f job/swanlab-migrate-s3-export -n <your_namespace>
```

### 5. Import DB Data

- Operation location: <span style="color: red"><strong>Target Cluster</strong></span>

Similar to export, each database has an independent import Job that can be executed in parallel.

::: info Import Architecture Notes

- **PostgreSQL**: Job downloads the dump file from the S3 transit bucket, then connects to the target cluster PG Service via `pg_restore` to restore. No PVC mounting, no need to scale down PG.
- **ClickHouse**: Job connects to the target cluster CH Service, uses native `RESTORE DATABASE ... FROM S3()` command. Before restoring, it automatically executes `DROP DATABASE IF EXISTS` and recreates an empty database (irreversible operation).
- **Redis**: Job downloads RDB from S3 and writes it to the target Redis PVC at `/data/dump.rdb`. After import completes, manually scale back the Redis Deployment, which will automatically load the RDB on startup.

:::

::: warning ClickHouse Restore Notes

- CH's `RESTORE` has no equivalent to `pg_restore --clean`. `allow_non_empty_tables=1` performs a merge, not a replace, which would double the row count. Therefore, the script executes `DROP DATABASE IF EXISTS` before RESTORE.
- If any single table in the target database exceeds 50GB, CH blocks DROP by default. You must first set `<max_table_size_to_drop>0</max_table_size_to_drop>` in the CH config to unlock.
  :::

::: details import-postgres

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-postgres
  namespace: <TARGET_NAMESPACE> # ⚠️ Required: [Target Cluster] K8s namespace
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
              echo "[1/4] Configuring rclone S3..."
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

              echo "[2/4] Downloading from ${TARGET}/..."
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

              echo "[3/4] Connection check..."
              ls -lh "${DUMP}"
              pg_isready -h "${PG_HOST}" -p "${PG_PORT:-5432}" -U "${PG_USER}" -d "${PG_DB}"

              echo "[4/4] pg_restore restoring..."
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
              echo "--- PostgreSQL restore complete ---"
```

:::

::: details import-clickhouse

```yaml
## ============================================================
## ClickHouse Native RESTORE Import Job
## ============================================================
## ⚠️ Database cleanup behavior (irreversible):
##   - Before RESTORE, executes DROP DATABASE IF EXISTS and recreates empty database
##   - If any single table in target DB exceeds 50GB, unlock DROP protection first
## ============================================================
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-clickhouse
  namespace: <TARGET_NAMESPACE> # ⚠️ Required: [Target Cluster] K8s namespace
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

              echo "[1/5] Connecting to target ClickHouse and cleaning target database..."
              "${CH_CLIENT[@]}" --query "SELECT version(), uptime()"

              ## Cleanup: DROP + CREATE empty database
              EXISTING_TABLES=$("${CH_CLIENT[@]}" --query "SHOW TABLES FROM ${CLICKHOUSE_DB}" 2>/dev/null || true)
              if [ -n "${EXISTING_TABLES}" ]; then
                echo "  Target database ${CLICKHOUSE_DB} is not empty, executing DROP..."
                "${CH_CLIENT[@]}" --query "DROP DATABASE IF EXISTS ${CLICKHOUSE_DB}"
              fi
              "${CH_CLIENT[@]}" --query "CREATE DATABASE IF NOT EXISTS ${CLICKHOUSE_DB}"
              echo "  Target database ${CLICKHOUSE_DB} is ready (empty)"

              ## Construct URL dynamically based on S3_FORCE_PATH_STYLE
              if [ "${S3_FORCE_PATH_STYLE:-false}" = "true" ]; then
                S3_URL="https://${S3_ENDPOINT}/${S3_BUCKET}"
              else
                S3_URL="https://${S3_BUCKET}.${S3_ENDPOINT}"
              fi
              PREFIX="${S3_PATH_PREFIX%/}"
              BACKUP_PATH="${PREFIX:+${PREFIX}/}${CLICKHOUSE_BACKUP_PATH}"

              echo "[2/5] Initiating RESTORE DATABASE ${CLICKHOUSE_DB} ← s3://${S3_BUCKET}/${BACKUP_PATH}..."
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
              echo "  Restore task submitted, id=${RESTORE_ID}"

              echo "[3/5] Polling restore progress..."
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
                    echo "--- ClickHouse native restore complete ---"
                    break
                    ;;
                  RESTORE_FAILED)
                    echo "ERROR: Restore failed: ${ERROR}"
                    exit 1
                    ;;
                esac
                sleep 10
              done

              echo "[4/5] Post-restore table list and sizes..."
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

              echo "[5/5] Post-restore row count (compare with export side)..."
              "${CH_CLIENT[@]}" --query "
                SELECT
                  '${CLICKHOUSE_DB}' AS database,
                  sum(rows) AS total_rows,
                  formatReadableQuantity(sum(rows)) AS rows_readable
                FROM system.parts
                WHERE database='${CLICKHOUSE_DB}' AND active
              "
              echo "--- Import complete, please compare with export-side row count ---"
```

:::

::: details import-redis

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: swanlab-import-redis
  namespace: <TARGET_NAMESPACE> # ⚠️ Required: [Target Cluster] K8s namespace
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
            claimName: swanlab-redis-pvc # ⚠️ Required: [Target Cluster] Redis PVC name
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
              echo "[1/4] Configuring rclone S3..."
              export RCLONE_CONFIG_S3_TYPE=s3
              export RCLONE_CONFIG_S3_PROVIDER=Other
              export RCLONE_CONFIG_S3_ENDPOINT="${S3_ENDPOINT}"
              export RCLONE_CONFIG_S3_REGION="${S3_REGION}"
              export RCLONE_CONFIG_S3_ACCESS_KEY_ID="${S3_AK}"
              export RCLONE_CONFIG_S3_SECRET_ACCESS_KEY="${S3_SK}"
              export RCLONE_CONFIG_S3_NO_CHECK_BUCKET=true
              export RCLONE_CONFIG_S3_FORCE_PATH_STYLE="${S3_FORCE_PATH_STYLE:-false}"

              : "${REDIS_BACKUP_OBJECT:?REDIS_BACKUP_OBJECT not set}"
              PREFIX="${S3_PATH_PREFIX%/}"
              TARGET="s3:${S3_BUCKET}/${PREFIX}"
              DUMP=/tmp/${REDIS_BACKUP_OBJECT}

              echo "[2/4] Downloading from ${TARGET}/..."
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

              echo "[3/4] Writing to /data/dump.rdb..."
              ls -lh "${DUMP}"
              cp -f "${DUMP}" /data/dump.rdb
              # Clean up AOF remnants to ensure Redis loads from dump.rdb on startup
              rm -f /data/appendonly.aof* 2>/dev/null || true
              rm -rf /data/appendonlydir 2>/dev/null || true
              ls -lh /data

              echo "[4/4] --- Redis restore complete ---"
              echo "⚠️ After this Job completes, scale back the Redis Deployment. Redis will automatically load dump.rdb on startup."
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
# Restore database services (mainly Redis) (replicas must be 1)
kubectl scale deployment swanlab-self-hosted-redis --replicas=1 -n <your_namespace>

# Confirm databases are ready
kubectl get pods -n <your_namespace> -w
```

```bash [2. Restore Application Layer]
# Restore replicas first, then scale up as needed
kubectl scale deploy/swanlab-self-hosted-house deploy/swanlab-self-hosted-server --replicas=1 -n <your_namespace>
```

```bash [4. Restore Gateway]
# Restore gateway
kubectl scale deploy/swanlab-self-hosted --replicas=2 -n <your_namespace>
```

:::

After restoration, you can observe pod health status and verify data recovery through the online service.

## 🧹 Job Cleanup

Jobs on both original and target clusters are **automatically cleaned up after 24 hours** (`ttlSecondsAfterFinished: 86400`). For manual cleanup:

```bash
# Source cluster
kubectl delete job swanlab-export-postgres swanlab-export-clickhouse swanlab-export-redis swanlab-migrate-s3-export -n <SOURCE_NAMESPACE>

# Target cluster
kubectl delete job swanlab-import-postgres swanlab-import-clickhouse swanlab-import-redis -n <TARGET_NAMESPACE>
```
