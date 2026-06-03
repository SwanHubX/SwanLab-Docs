# Custom Value Configuration

> This document explains the meaning and configuration of each field in `values.yaml`.

## Global Configuration (`global`)

| Field | Type | Default | Description |
|------|------|--------|------|
| `global.imagePullSecrets` | list | `[]` | Image pull credentials (private repository authentication) |
| `global.clusterDomain` | string | `cluster.local` | Kubernetes cluster domain |
| `global.podAntiAffinityPreset` | string | `soft` | Pod anti-affinity strategy: `soft` (try to spread) / `hard` (force spread) / `none` (no setting) |
| `global.settings.loginHost` | string | `""` | Login host address. After modification, the API Key login address displayed in the frontend application will change accordingly (does not affect the actual backend service address) |

### Pod Anti-Affinity

You can control Pod distribution strategy by setting `global.podAntiAffinityPreset` to improve disaster recovery capabilities:

::: details Pod Anti-Affinity Configuration Example

```yaml
global:
  podAntiAffinityPreset: "soft" # soft, hard, or none
```

:::

- `soft` (default): All Pods will try to be evenly distributed across Nodes
- `hard`: Ensures Pods of the same service are not scheduled on the same Node
- `none`: Disables Pod anti-affinity

### Login Domain (loginHost)

By default, the **login host** displayed on the `<Your Host>/space/~/quick-start` page automatically uses the domain name `<Your Host>` you are currently using to access the frontend. If you need to modify this value, you can specify it to your desired domain name by configuring `global.settings.loginHost`.

> **Note**: This setting does not affect the actual backend service address. You need to configure the corresponding forwarding rules yourself.

## Gateway (`gateway`)

| Field | Type | Default | Description |
|------|------|--------|------|
| `gateway.replicas` | int | `2` | Gateway replica count |
| `gateway.image.repository` | string | `repo.swanlab.cn/public/traefik` | Traefik gateway image address |
| `gateway.image.tag` | string | `3.6` | Traefik image tag |
| `gateway.identifyImage.repository` | string | `repo.swanlab.cn/public/swanlab-helper/identify` | Gateway authentication auxiliary image address |
| `gateway.identifyImage.tag` | string | `v1.2` | Authentication auxiliary image tag |
| `gateway.service.type` | string | `ClusterIP` | Service type |
| `gateway.service.ports.web` | int | `80` | Non-secure entry port (accessible from outside the cluster) |
| `gateway.service.ports.internal` | int | `8080` | Internal entry port (only accessible within the cluster) |
| `gateway.service.ports.traefik` | int | `8081` | Traefik dashboard port |
| `gateway.service.ports.metrics` | int | `9100` | Prometheus metrics collection port |
| `gateway.customNodeSelector` | object | `{}` | Node selector, e.g., `{ swanlab: "true" }` |

### Configuring Application Access Entrypoint

The domain name of the application service within the cluster is the release name you deployed. For example, assuming your `cluster domain` is `cluster.local` and your deployment command is:

```bash
# Assuming the default release_name is swanlab-self-hosted
helm install swanlab-self-hosted swanlab/self-hosted -n <your_namespace>
```

- The domain name of the application within the `<your_namespace>` namespace is `swanlab-self-hosted` i.e. (`<release_name>`)
- The domain name of the application within the `kubernetes` cluster is: `swanlab-self-hosted.<your_namespace>.svc.cluster.local`

You can write your load balancing strategy based on the above information. It is generally recommended to prioritize using **dedicated domain names (Host-based)** to configure access policies to avoid routing conflicts caused by complex or changing path rules.

**Based on the principle of architectural decoupling**, `swanlab-self-hosted` does not have a built-in Ingress controller. You need to configure the external access entrypoint on the cluster's load balancer (or Ingress), which is also responsible for **TLS termination (HTTPS offloading)**.

**Regarding security policies**, the application trusts all `X-Forwarded-*` request headers by default. If you need stricter header validation or forwarding control, be sure to implement it uniformly at the load balancing layer — this may affect the effectiveness of internal S3 signatures. If you use an external object storage service, you don't need to worry about this.

## Metrics Buffer Queue (`vector`)

| Field | Type | Default | Description |
|------|------|--------|------|
| `vector.replicas` | int | `2` | Vector replica count |
| `vector.image.repository` | string | `repo.swanlab.cn/public/vector` | Vector image address |
| `vector.image.tag` | string | `0.51.1-debian` | Vector image tag |
| `vector.sinks.bufferMaxSize` | int | `10737418240` | Maximum buffer size (bytes), **must not exceed 1/3 of `persistence.storageSize`** |
| `vector.persistence.storageClass` | string | `""` | StorageClass (leave empty to use cluster default) |
| `vector.persistence.storageSize` | string | `60Gi` | Storage volume size, **recommended at least 60Gi**, ensure ≥ 3x `bufferMaxSize` |

> ⚠️ Vector's PVC names are not modifiable by default (`data-swanlab-self-hosted-vector-0` / `data-swanlab-self-hosted-vector-1`).

## Helper Container (`helper`)

| Field | Type | Default | Description |
|------|------|--------|------|
| `helper.image.repository` | string | `repo.swanlab.cn/public/busybox` | Busybox image address (used for health checks of various components) |
| `helper.image.tag` | string | `1.37.0` | Busybox image tag |

## Application Services (`service`)

### SwanLab-Server (Backend Service)

| Field | Type | Default | Description |
|------|------|--------|------|
| `service.server.replicas` | int | `2` | Replica count |
| `service.server.image.repository` | string | `repo.swanlab.cn/self-hosted/swanlab-server` | Image address |
| `service.server.image.tag` | string | `""` | Image tag, **set to empty string** to auto-sync the version specified by the Chart |

### SwanLab-House (Backend Experiment OLAP Service)

| Field | Type | Default | Description |
|------|------|--------|------|
| `service.house.replicas` | int | `2` | Replica count |
| `service.house.image.repository` | string | `repo.swanlab.cn/self-hosted/swanlab-house` | Image address |
| `service.house.image.tag` | string | `""` | Image tag, **set to empty string** to auto-sync the version specified by the Chart |
| `service.house.persistence.storageClass` | string | `""` | StorageClass |
| `service.house.persistence.storageSize` | string | `10Gi` | Storage volume size |

> **Storage Note**: `swanlab-house` is deployed as a `StatefulSet` and requires a mounted storage volume. Unlike base services, `existingClaim` is **not supported** here.
> `swanlab-house` stores some metric intermediate products in the storage volume. Generally, you do not need to care about the data in this storage volume.

### SwanLab-Cloud (Frontend Charts)

| Field | Type | Default | Description |
|------|------|--------|------|
| `service.cloud.replicas` | int | `1` | Replica count |
| `service.cloud.image.repository` | string | `repo.swanlab.cn/self-hosted/swanlab-cloud` | Image address |
| `service.cloud.image.tag` | string | `""` | Image tag, **set to empty string** to auto-sync the version specified by the Chart |

### SwanLab-Next (Frontend UI)

| Field | Type | Default | Description |
|------|------|--------|------|
| `service.next.replicas` | int | `2` | Replica count |
| `service.next.image.repository` | string | `repo.swanlab.cn/self-hosted/swanlab-next` | Image address |
| `service.next.image.tag` | string | `""` | Image tag, **set to empty string** to auto-sync the version specified by the Chart |

> **Application Image Tag Note**: The `tag` of the four application images under `service` (server / house / cloud / next) should all be set to **empty strings** rather than `latest`. The Chart will automatically inject the correct version number during rendering.

### Common Fields (Supported by all services)

| Field | Type | Default | Description |
|------|------|--------|------|
| `*.customLabels` | object | `{}` | Custom Service labels |
| `*.customAnnotations` | object | `{}` | Custom Service annotations |
| `*.customPodLabels` | object | `{}` | Custom Pod labels |
| `*.customPodAnnotations` | object | `{}` | Custom Pod annotations |
| `*.customTolerations` | list | `[]` | Custom Tolerations |
| `*.customNodeSelector` | object | `{}` | Node selector, JSON format, e.g., `{ swanlab: "true" }` |
| `*.resources` | object | `{}` | Resource limits (requests/limits), e.g., `{ requests: { cpu: "500m", memory: "512Mi" } }` |


> Application performance is a complex calculation metric that typically also depends on resource limits. It is recommended to configure CPU and memory usage reasonably through the `resources` field.

## Built-in Base Services (`dependencies`)

When `integrations.<service>.enabled` is `false`, the Chart will automatically deploy the following components within the cluster.

### Storage Resource Configuration Recommendations

If you use built-in single-instance base services, it is recommended to declare your own `storage-class` to support data persistence.

Before customizing the storage class configuration, please ensure:
1. The corresponding base service resource does **not** have `integrations` enabled
2. Your `storage-class` or `claim` exists in the cluster

**Configuration Method (using PostgreSQL as an example):**

1. **Auto-create Storage Volume**: Configure `storageClass` and `storageSize` under `dependencies.postgres.persistence`
2. **Use Existing Storage Volume**: Specify an existing PVC via `dependencies.postgres.persistence.existingClaim` (recommended practice, ensuring storage resources are managed by you)

> The storage configuration method for other base services (Redis, ClickHouse, MinIO) is the same.

### PostgreSQL

| Field | Type | Default | Description |
|------|------|--------|------|
| `dependencies.postgres.image.repository` | string | `repo.swanlab.cn/self-hosted/postgres` | Image address |
| `dependencies.postgres.image.tag` | string | `16.1` | Image tag, recommended 16.x and above |
| `dependencies.postgres.username` | string | `""` | Database username |
| `dependencies.postgres.password` | string | `""` | Database password |
| `dependencies.postgres.persistence.existingClaim` | string | `""` | Use an existing PVC name (leave empty to auto-create) |
| `dependencies.postgres.persistence.storageClass` | string | `""` | StorageClass |
| `dependencies.postgres.persistence.storageSize` | string | `10Gi` | Storage volume size |

### Redis

| Field | Type | Default | Description |
|------|------|--------|------|
| `dependencies.redis.image.repository` | string | `repo.swanlab.cn/self-hosted/redis-stack` | Image address |
| `dependencies.redis.image.tag` | string | `7.4.0-v8` | Image tag |
| `dependencies.redis.persistence.existingClaim` | string | `""` | Use an existing PVC name (leave empty to auto-create) |
| `dependencies.redis.persistence.storageClass` | string | `""` | StorageClass |
| `dependencies.redis.persistence.storageSize` | string | `10Gi` | Storage volume size |

### ClickHouse

| Field | Type | Default | Description |
|------|------|--------|------|
| `dependencies.clickhouse.image.repository` | string | `repo.swanlab.cn/self-hosted/clickhouse-server` | Image address |
| `dependencies.clickhouse.image.tag` | string | `24.3` | Image tag |
| `dependencies.clickhouse.username` | string | `""` | Database username (leave empty if using existingSecret) |
| `dependencies.clickhouse.password` | string | `""` | Database password (leave empty if using existingSecret) |
| `dependencies.clickhouse.persistence.existingClaim` | string | `""` | Use an existing PVC name (leave empty to auto-create) |
| `dependencies.clickhouse.persistence.storageClass` | string | `""` | StorageClass |
| `dependencies.clickhouse.persistence.storageSize` | string | `20Gi` | Storage volume size |

### MinIO (Built-in S3 Object Storage)

> If external S3 is already integrated, this can be ignored.

| Field | Type | Default | Description |
|------|------|--------|------|
| `dependencies.s3.image.repository` | string | `repo.swanlab.cn/self-hosted/minio/minio` | MinIO image address |
| `dependencies.s3.image.tag` | string | `RELEASE.2025-09-07T16-13-09Z` | MinIO image tag |
| `dependencies.s3.mcImage.repository` | string | `repo.swanlab.cn/self-hosted/minio/mc` | MinIO client image address |
| `dependencies.s3.mcImage.tag` | string | `RELEASE.2025-08-13T08-35-41Z` | MinIO client image tag |
| `dependencies.s3.accessKey` | string | `""` | Access Key |
| `dependencies.s3.secretKey` | string | `""` | Secret Key (leave empty to auto-generate) |
| `dependencies.s3.persistence.existingClaim` | string | `""` | Use an existing PVC (leave empty to auto-create) |
| `dependencies.s3.persistence.storageClass` | string | `""` | StorageClass |
| `dependencies.s3.persistence.storageSize` | string | `20Gi` | Storage volume size |

## External Base Service Integration (`integrations`)

The `integrations` section is used to connect to existing external base services (databases, caches, object storage, etc.), replacing the built-in single-instance deployment of the Chart.

:::warning
If you enable any integrated base service resource (e.g., set `integrations.postgres.enabled` to `true`), the corresponding single-instance service deployed by `swanlab-self-hosted` in `dependencies` will be destroyed.
:::

### [Recommended] External Object Storage S3 (`integrations.s3`)

Used to connect to external S3 compatible object storage (such as Alibaba Cloud OSS, Tencent Cloud COS, AWS S3, etc.), requiring **mandatory AWS S3 protocol compatibility**.

:::warning
If your cloud object storage distinguishes S3 protocol endpoint access, please pay special attention to filling in the S3 endpoint
:::
![s3-Config](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260602111133909.png)

| Field | Type | Default | Description |
|------|------|--------|------|
| `integrations.s3.enabled` | bool | `true` | Enable external S3 integration (when enabled, `dependencies.s3` will not be deployed) |
| `integrations.s3.existingSecret` | string | `swanlab-secret-s3` | K8s Secret name storing AK/SK |

**Secret Data Structure (`integrations.s3.existingSecret`):**

| `.data.<keys>` | Description |
| --- | --- |
| `accessKey` | Object storage access key |
| `secretKey` | Object storage secret key |

#### Public Bucket Configuration (`integrations.s3.public`)

| Field | Type | Default | Description |
|------|------|--------|------|
| `public.ssl` | bool | `true` | Enable SSL |
| `public.endpoint` | string | `""` | S3 endpoint, **without bucket prefix**, e.g., `oss-cn-beijing.aliyuncs.com` |
| `public.region` | string | `""` | S3 region, e.g., `cn-beijing` |
| `public.port` | int | `443` | Port number |
| `public.domain` | string | `""` | Public bucket URL, **must include `https://` prefix**, e.g., `https://<bucket_name>.oss-cn-beijing.aliyuncs.com` |
| `public.pathStyle` | bool | `false` | Path access method, usually set to `false` for public cloud object storage |
| `public.bucket` | string | `""` | Bucket name |

> 📎 Special note: Major cloud providers no longer recommend using pathStyle=True path naming. The default is False. For the difference, please refer to: [Virtual hosting of general purpose buckets - AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/VirtualHosting.html)

#### Private Bucket Configuration (`integrations.s3.private`)

| Field | Type | Default | Description |
|------|------|--------|------|
| `private.ssl` | bool | `true` | Enable SSL |
| `private.endpoint` | string | `""` | S3 endpoint, **without bucket prefix**, e.g., `oss-cn-beijing.aliyuncs.com` |
| `private.region` | string | `""` | S3 region, e.g., `cn-beijing` |
| `private.port` | int | `443` | Port number |
| `private.pathStyle` | bool | `false` | Path access method, usually set to `false` for public cloud object storage |
| `private.bucket` | string | `""` | Bucket name |

::: details External S3 Object Storage Integration Configuration Example

```yaml
integrations:
  s3:
    enabled: true
    public:
      ssl: true
      endpoint: "xxx.s3.com"
      region: "cn-beijing"
      pathStyle: false
      port: 443
      domain: "https://xxx.xxxx.s3.com"
      bucket: "swanlab-public"
    private:
      ssl: true
      endpoint: "xxx.s3.com"
      region: "cn-beijing"
      pathStyle: false
      port: 443
      bucket: "swanlab-private"
    existingSecret: integration-s3
```

:::

:::warning
- The permission for publicBucket is **public read, private write**. The permission for privateBucket is **private read-write**
- When you choose a custom object storage service, please ensure your object storage service can be accessed directly from outside (via IP or domain name)
- Your object storage secret key must have write permissions and S3 signing permissions for both **publicBucket** and **privateBucket**
:::



### [Not Recommended] External PostgreSQL (`integrations.postgres`)

Connect to external PostgreSQL (self-built cnpg cluster or cloud provider RDS).

| Field | Type | Default | Description |
|------|------|--------|------|
| `integrations.postgres.enabled` | bool | `false` | Enable external PostgreSQL, built-in single instance will be removed |
| `integrations.postgres.host` | string | `""` | Database host address |
| `integrations.postgres.port` | int | `5432` | Database port |
| `integrations.postgres.database` | string | `""` | Database name |
| `integrations.postgres.existingSecret` | string | `""` | K8s Secret name storing credentials |

**Secret Data Structure (`integrations.postgres.existingSecret`):**

| `.data.<keys>` | Description |
| --- | --- |
| `username` | Read-write username |
| `password` | Read-write user password |
| `primaryUrl` | Read-write database connection string, format: `postgresql://{username}:${password}@postgres:5432/app?schema=public` |
| `replicaUrl` | Read-only database connection string, generally used for load balancing. If a read-only user/cluster is not configured, the read-write connection string can be used instead |

::: details External PostgreSQL Integration Configuration Example

```yaml
integrations:
  postgres:
    enabled: true
    host: "example.postgres"
    port: 5432
    database: "app"
    existingSecret: integration-postgres
```

:::

> Please ensure the above configuration corresponds with the information in the Secret. For detailed key data structure descriptions, please refer to [values.yaml](https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml).

### [Not Recommended] External Redis (`integrations.redis`)

Connect to external Redis (self-built cluster or cloud provider Redis service).

| Field | Type | Default | Description |
|------|------|--------|------|
| `integrations.redis.enabled` | bool | `false` | Enable external Redis, built-in single instance will be removed |
| `integrations.redis.host` | string | `""` | Redis host address |
| `integrations.redis.port` | int | `6379` | Redis port |
| `integrations.redis.database` | string | `"0"` | Database number |
| `integrations.redis.existingSecret` | string | `""` | K8s Secret name storing credentials |

**Secret Data Structure (`integrations.redis.existingSecret`):**

| `.data.<keys>` | Description |
| --- | --- |
| `url` | Database connection string, format: `redis://{username}:${password}@redis:6379` |

::: details External Redis Integration Configuration Example

```yaml
integrations:
  redis:
    enabled: true
    host: "example.redis"
    port: 6379
    database: "0"
    existingSecret: integration-redis
```

:::

> Please ensure the above configuration corresponds with the information in the Secret.

### [Not Recommended] External ClickHouse (`integrations.clickhouse`)

Connect to external ClickHouse (self-built cluster or cloud provider service).

| Field | Type | Default | Description |
|------|------|--------|------|
| `integrations.clickhouse.enabled` | bool | `false` | Enable external ClickHouse, built-in single instance will be removed |
| `integrations.clickhouse.host` | string | `""` | ClickHouse host address |
| `integrations.clickhouse.httpPort` | int | `8123` | HTTP protocol port |
| `integrations.clickhouse.tcpPort` | int | `9000` | TCP protocol port |
| `integrations.clickhouse.database` | string | `""` | Database name |
| `integrations.clickhouse.existingSecret` | string | `""` | K8s Secret name storing credentials |

**Secret Data Structure (`integrations.clickhouse.existingSecret`):**

| `.data.<keys>` | Description |
| --- | --- |
| `username` | Read-write username |
| `password` | Read-write user password |

::: details External ClickHouse Integration Configuration Example

```yaml
integrations:
  clickhouse:
    enabled: true
    host: "example.clickhouse"
    httpPort: 8123
    tcpPort: 9000
    database: "app"
    existingSecret: integration-clickhouse
```

:::

> Please ensure the above configuration corresponds with the information in the Secret.
