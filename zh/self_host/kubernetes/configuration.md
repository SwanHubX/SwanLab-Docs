# 自定义 Value 配置说明

> 本文档说明 `values.yaml` 中各字段的含义与配置方式。

## 全局配置（`global`）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `global.imagePullSecrets` | list | `[]` | 镜像拉取凭证（私有仓库认证） |
| `global.clusterDomain` | string | `cluster.local` | Kubernetes 集群域名 |
| `global.podAntiAffinityPreset` | string | `soft` | Pod 反亲和策略：`soft`（尽量分散）/ `hard`（强制分散）/ `none`（不设置） |
| `global.settings.loginHost` | string | `""` | 登录主机地址，修改后前端应用中显示的 API Key 登录地址会随之变化（不影响实际后端服务地址） |

### Pod 反亲和性

您可以通过设置 `global.podAntiAffinityPreset` 来控制 Pod 的分布策略，以提升容灾能力：

::: details Pod 反亲和性配置示例

```yaml
global:
  podAntiAffinityPreset: "soft" # soft, hard, or none
```

:::

- `soft`（默认）：所有 Pod 将尽量均匀分布在各个 Node 上
- `hard`：确保同一服务的 Pod 不会分布在同一 Node 上
- `none`：禁用 Pod 反亲和性

### 登录域名（loginHost）

默认情况下，`<Your Host>/space/~/quick-start` 页面中显示的 **login host** 会自动使用您当前访问前端的域名 `<Your Host>`。如果您需要修改此值，可以通过配置 `global.settings.loginHost` 指定为您期望的域名。

> **注意**：此设置不会影响实际的后端服务地址，您需要自己配置对应的转发规则。

## 网关（`gateway`）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gateway.replicas` | int | `2` | 网关副本数 |
| `gateway.image.repository` | string | `repo.swanlab.cn/public/traefik` | Traefik 网关镜像地址 |
| `gateway.image.tag` | string | `3.6` | Traefik 镜像标签 |
| `gateway.identifyImage.repository` | string | `repo.swanlab.cn/public/swanlab-helper/identify` | 网关鉴权辅助镜像地址 |
| `gateway.identifyImage.tag` | string | `v1.2` | 鉴权辅助镜像标签 |
| `gateway.service.type` | string | `ClusterIP` | Service 类型 |
| `gateway.service.ports.web` | int | `80` | 非安全入口端口（可从集群外部访问） |
| `gateway.service.ports.internal` | int | `8080` | 内部入口端口（仅集群内部访问） |
| `gateway.service.ports.traefik` | int | `8081` | Traefik 仪表盘端口 |
| `gateway.service.ports.metrics` | int | `9100` | Prometheus 指标采集端口 |
| `gateway.customNodeSelector` | object | `{}` | 节点选择器，例如 `{ swanlab: "true" }` |

### 配置应用访问入口

应用服务在集群中的域名为您部署的 release 名称。例如，假设您的 `cluster domain` 为 `cluster.local`，假设您的部署命令为：

```bash
# 假设默认 release_name 为 swanlab-self-hosted
helm install swanlab-self-hosted swanlab/self-hosted -n <your_namespace>
```

- 应用在 `<your_namespace>` 命名空间下的域名为 `swanlab-self-hosted` 即 (`<release_name>`)
- 应用在 `kubernetes` 集群下的域名为：`swanlab-self-hosted.<your_namespace>.svc.cluster.local`

您可以基于如上信息编写负载均衡策略。通常建议您优先使用 **独立域名（Host-based）** 来配置访问策略，以规避因路径规则复杂或变更导致的路由冲突。

**基于架构解耦原则**，`swanlab-self-hosted` 不内置 Ingress 控制器。您需要在集群的负载均衡器（或 Ingress）上配置外部访问入口，并由其负责 **TLS 终止（HTTPS 卸载）**。

**在安全策略方面**，应用默认信任所有的 `X-Forwarded-*` 请求头。如果您需要更严格的头部校验或转发控制，请务必在负载均衡层统一实施——这有可能影响到内部 S3 的签名效果，如果您使用外部对象存储服务，则不需要有这方面担忧。

## 指标缓冲队列（`vector`）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vector.replicas` | int | `2` | Vector 副本数 |
| `vector.image.repository` | string | `repo.swanlab.cn/public/vector` | Vector 镜像地址 |
| `vector.image.tag` | string | `0.51.1-debian` | Vector 镜像标签 |
| `vector.sinks.bufferMaxSize` | int | `10737418240` | 缓冲区最大大小（字节），**不得超过 `persistence.storageSize` 的 1/3** |
| `vector.persistence.storageClass` | string | `""` | StorageClass（留空使用集群默认） |
| `vector.persistence.storageSize` | string | `60Gi` | 存储卷大小，**建议至少 60Gi**，确保 ≥ `bufferMaxSize` 的 3 倍 |

> ⚠️ Vector 的 PVC 名称默认不可修改（`data-swanlab-self-hosted-vector-0` / `data-swanlab-self-hosted-vector-1`）。

## 辅助容器（`helper`）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `helper.image.repository` | string | `repo.swanlab.cn/public/busybox` | Busybox 镜像地址（用于各组件健康检查等） |
| `helper.image.tag` | string | `1.37.0` | Busybox 镜像标签 |

## 应用服务（`service`）

### SwanLab-Server（后端服务）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `service.server.replicas` | int | `2` | 副本数 |
| `service.server.image.repository` | string | `repo.swanlab.cn/self-hosted/swanlab-server` | 镜像地址 |
| `service.server.image.tag` | string | `""` | 镜像标签，**置为空字符串**以自动同步 Chart 指定的版本号 |

### SwanLab-House（后端实验 OLAP 服务）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `service.house.replicas` | int | `2` | 副本数 |
| `service.house.image.repository` | string | `repo.swanlab.cn/self-hosted/swanlab-house` | 镜像地址 |
| `service.house.image.tag` | string | `""` | 镜像标签，**置为空字符串**以自动同步 Chart 指定的版本号 |
| `service.house.persistence.storageClass` | string | `""` | StorageClass |
| `service.house.persistence.storageSize` | string | `10Gi` | 存储卷大小 |

> **存储说明**：`swanlab-house` 以 `StatefulSet` 部署，需要挂载存储卷。与基础服务不同，此处**不支持**配置 `existingClaim`。
> `swanlab-house` 会在存储卷下存储一些指标中间产物，一般情况下您不需要关心此存储卷中的数据。

### SwanLab-Cloud（前端图表）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `service.cloud.replicas` | int | `1` | 副本数 |
| `service.cloud.image.repository` | string | `repo.swanlab.cn/self-hosted/swanlab-cloud` | 镜像地址 |
| `service.cloud.image.tag` | string | `""` | 镜像标签，**置为空字符串**以自动同步 Chart 指定的版本号 |

### SwanLab-Next（前端UI）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `service.next.replicas` | int | `2` | 副本数 |
| `service.next.image.repository` | string | `repo.swanlab.cn/self-hosted/swanlab-next` | 镜像地址 |
| `service.next.image.tag` | string | `""` | 镜像标签，**置为空字符串**以自动同步 Chart 指定的版本号 |

> **应用镜像标签说明**：`service` 下的四个应用镜像（server / house / cloud / next）的 `tag` 均应设置为**空字符串**而非 `latest`，Chart 会在渲染时自动注入正确的版本号。

### 通用字段（所有 service 均支持）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `*.customLabels` | object | `{}` | 自定义 Service 标签 |
| `*.customAnnotations` | object | `{}` | 自定义 Service 注解 |
| `*.customPodLabels` | object | `{}` | 自定义 Pod 标签 |
| `*.customPodAnnotations` | object | `{}` | 自定义 Pod 注解 |
| `*.customTolerations` | list | `[]` | 自定义容忍度（Toleration） |
| `*.customNodeSelector` | object | `{}` | 节点选择器，JSON 格式，例如 `{ swanlab: "true" }` |
| `*.resources` | object | `{}` | 资源限制（requests/limits），例如 `{ requests: { cpu: "500m", memory: "512Mi" } }` |


> 应用性能是一个复杂的计算指标，通常还取决于资源限制。建议同时通过 `resources` 字段合理配置 CPU 和内存用量。

## 内置基础服务（`dependencies`）

当 `integrations.<service>.enabled` 为 `false` 时，Chart 会在集群内自动部署以下组件。

### 存储资源配置建议

如果您使用内置的单实例基础服务，建议您自行声明 `storage-class` 以支持数据持久化。

在进行自定义存储类配置之前，请确保：
1. 对应的基础服务资源**没有**开启 `integrations`
2. 您的 `storage-class` 或 `claim` 已存在于集群中

**配置方式（以 PostgreSQL 为例）：**

1. **自动创建存储卷**：配置 `dependencies.postgres.persistence` 下的 `storageClass` 和 `storageSize`
2. **使用已有存储卷**：通过 `dependencies.postgres.persistence.existingClaim` 指定已存在的 PVC（推荐做法，确保存储资源由您自己管理）

> 其他基础服务（Redis、ClickHouse、MinIO）的存储配置方式相同。

### PostgreSQL

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dependencies.postgres.image.repository` | string | `repo.swanlab.cn/self-hosted/postgres` | 镜像地址 |
| `dependencies.postgres.image.tag` | string | `16.1` | 镜像标签，建议 16.x 及以上 |
| `dependencies.postgres.username` | string | `""` | 数据库用户名 |
| `dependencies.postgres.password` | string | `""` | 数据库密码 |
| `dependencies.postgres.persistence.existingClaim` | string | `""` | 使用已有的 PVC 名称（留空则自动创建） |
| `dependencies.postgres.persistence.storageClass` | string | `""` | StorageClass |
| `dependencies.postgres.persistence.storageSize` | string | `10Gi` | 存储卷大小 |

### Redis

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dependencies.redis.image.repository` | string | `repo.swanlab.cn/self-hosted/redis-stack` | 镜像地址 |
| `dependencies.redis.image.tag` | string | `7.4.0-v8` | 镜像标签 |
| `dependencies.redis.persistence.existingClaim` | string | `""` | 使用已有的 PVC 名称（留空则自动创建） |
| `dependencies.redis.persistence.storageClass` | string | `""` | StorageClass |
| `dependencies.redis.persistence.storageSize` | string | `10Gi` | 存储卷大小 |

### ClickHouse

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dependencies.clickhouse.image.repository` | string | `repo.swanlab.cn/self-hosted/clickhouse-server` | 镜像地址 |
| `dependencies.clickhouse.image.tag` | string | `24.3` | 镜像标签 |
| `dependencies.clickhouse.username` | string | `""` | 数据库用户名（如使用 existingSecret 则留空） |
| `dependencies.clickhouse.password` | string | `""` | 数据库密码（如使用 existingSecret 则留空） |
| `dependencies.clickhouse.persistence.existingClaim` | string | `""` | 使用已有的 PVC 名称（留空则自动创建） |
| `dependencies.clickhouse.persistence.storageClass` | string | `""` | StorageClass |
| `dependencies.clickhouse.persistence.storageSize` | string | `20Gi` | 存储卷大小 |

### MinIO（内置 S3 对象存储）

> 如已集成外部 S3，可忽略此项。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dependencies.s3.image.repository` | string | `repo.swanlab.cn/self-hosted/minio/minio` | MinIO 镜像地址 |
| `dependencies.s3.image.tag` | string | `RELEASE.2025-09-07T16-13-09Z` | MinIO 镜像标签 |
| `dependencies.s3.mcImage.repository` | string | `repo.swanlab.cn/self-hosted/minio/mc` | MinIO 客户端镜像地址 |
| `dependencies.s3.mcImage.tag` | string | `RELEASE.2025-08-13T08-35-41Z` | MinIO 客户端镜像标签 |
| `dependencies.s3.accessKey` | string | `""` | Access Key |
| `dependencies.s3.secretKey` | string | `""` | Secret Key（留空将自动生成） |
| `dependencies.s3.persistence.existingClaim` | string | `""` | 使用已有的 PVC（留空则自动创建） |
| `dependencies.s3.persistence.storageClass` | string | `""` | StorageClass |
| `dependencies.s3.persistence.storageSize` | string | `20Gi` | 存储卷大小 |

## 外部基础服务集成（`integrations`）

`integrations` 部分用于接入外部已有的基础服务（数据库、缓存、对象存储等），替代 Chart 内置的单实例部署。

:::warning
如果您将任一集成基础服务资源开启（例如设置 `integrations.postgres.enabled` 为 `true`），`swanlab-self-hosted` 在 `denpendencies` 中部署的对应单实例服务将被销毁。
:::

### 【建议】外部对象存储 S3（`integrations.s3`）

用于接入外部 S3 兼容对象存储（如阿里云 OSS、腾讯云 COS、AWS S3 等），要求 **必须兼容 AWS S3 协议**。

:::warning
如果您所在的云对象存储对 S3 协议的 endpoint 访问做了区分，需要请特别注意应该填写 S3 endpoint
:::
![s3-Config](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260602111133909.png)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `integrations.s3.enabled` | bool | `true` | 启用后支持集成外部 S3（启用后 `dependencies.s3` 不会部署） |
| `integrations.s3.existingSecret` | string | `swanlab-secret-s3` | 存储 AK/SK 的 K8s Secret 名称 |

**Secret 数据结构（`integrations.s3.existingSecret`）：**

| `.data.<keys>` | 说明 |
| --- | --- |
| `accessKey` | 对象存储访问密钥 |
| `secretKey` | 对象存储密钥 |

#### Public 桶配置（`integrations.s3.public`）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `public.ssl` | bool | `true` | 是否启用 SSL |
| `public.endpoint` | string | `""` | S3 接入点，**不带 bucket 前缀**，例如 `oss-cn-beijing.aliyuncs.com` |
| `public.region` | string | `""` | S3 地域，例如 `cn-beijing` |
| `public.port` | int | `443` | 端口号 |
| `public.domain` | string | `""` | Public 桶 URL，**需携带 `https://` 前缀**，例如 `https://<bucket_name>.oss-cn-beijing.aliyuncs.com` |
| `public.pathStyle` | bool | `false` | 路径访问方式，公有云对象存储通常设为 `false` |
| `public.bucket` | string | `""` | 桶名称 |

> 📎 特别说明：主流云厂商基本不再推荐使用 pathStyle=True 的路径命名方式，默认均为 False，区别可参考阅读:  [Virtual hosting of general purpose buckets-AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/VirtualHosting.html)

#### Private 桶配置（`integrations.s3.private`）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `private.ssl` | bool | `true` | 是否启用 SSL |
| `private.endpoint` | string | `""` | S3 接入点，**不带 bucket 前缀**，例如 `oss-cn-beijing.aliyuncs.com` |
| `private.region` | string | `""` | S3 地域，例如 `cn-beijing` |
| `private.port` | int | `443` | 端口号 |
| `private.pathStyle` | bool | `false` | 路径访问方式，公有云对象存储通常设为 `false` |
| `private.bucket` | string | `""` | 桶名称 |

::: details 外部集成 S3 对象存储配置示例

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
- publicBucket 的权限为**公有读私有写**，privateBucket 的权限为**私有读写**
- 当您选择自定义对象存储服务时，请保证您的对象存储服务可以直接通过外部访问（通过 IP 或域名）
- 您的对象存储密钥必须对 **publicBucket** 和 **privateBucket** 同时有写权限和 S3 签名权限
:::



### 【不推荐】外部 PostgreSQL（`integrations.postgres`）

接入外部 PostgreSQL（自建 cnpg 集群或云厂商 RDS）。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `integrations.postgres.enabled` | bool | `false` | 启用后使用外部 PostgreSQL，内置单实例将被移除 |
| `integrations.postgres.host` | string | `""` | 数据库主机地址 |
| `integrations.postgres.port` | int | `5432` | 数据库端口 |
| `integrations.postgres.database` | string | `""` | 数据库名称 |
| `integrations.postgres.existingSecret` | string | `""` | 存储凭据的 K8s Secret 名称 |

**Secret 数据结构（`integrations.postgres.existingSecret`）：**

| `.data.<keys>` | 说明 |
| --- | --- |
| `username` | 可读可写用户名称 |
| `password` | 可读可写用户密码 |
| `primaryUrl` | 可读可写数据库连接串，格式：`postgresql://{username}:${password}@postgres:5432/app?schema=public` |
| `replicaUrl` | 只读数据库连接串，一般用于负载均衡。如果未配置只读用户/集群，可使用可读可写连接串代替 |

::: details 外部集成 PostgreSQL 配置示例

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

> 请保证上述配置与 Secret 中能对应上。详细密钥数据结构说明请参阅 [values.yaml](https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml)。

### 【不推荐】外部 Redis（`integrations.redis`）

接入外部 Redis（自建集群或云厂商 Redis 服务）。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `integrations.redis.enabled` | bool | `false` | 启用后使用外部 Redis，内置单实例将被移除 |
| `integrations.redis.host` | string | `""` | Redis 主机地址 |
| `integrations.redis.port` | int | `6379` | Redis 端口 |
| `integrations.redis.database` | string | `"0"` | 数据库编号 |
| `integrations.redis.existingSecret` | string | `""` | 存储凭据的 K8s Secret 名称 |

**Secret 数据结构（`integrations.redis.existingSecret`）：**

| `.data.<keys>` | 说明 |
| --- | --- |
| `url` | 数据库连接串，格式：`redis://{username}:${password}@redis:6379` |

::: details 外部集成 Redis 配置示例

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

> 请保证上述配置与 Secret 中能对应上。

### 【不推荐】外部 ClickHouse（`integrations.clickhouse`）

接入外部 ClickHouse（自建集群或云厂商服务）。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `integrations.clickhouse.enabled` | bool | `false` | 启用后使用外部 ClickHouse，内置单实例将被移除 |
| `integrations.clickhouse.host` | string | `""` | ClickHouse 主机地址 |
| `integrations.clickhouse.httpPort` | int | `8123` | HTTP 协议端口 |
| `integrations.clickhouse.tcpPort` | int | `9000` | TCP 协议端口 |
| `integrations.clickhouse.database` | string | `""` | 数据库名称 |
| `integrations.clickhouse.existingSecret` | string | `""` | 存储凭据的 K8s Secret 名称 |

**Secret 数据结构（`integrations.clickhouse.existingSecret`）：**

| `.data.<keys>` | 说明 |
| --- | --- |
| `username` | 可读可写用户名称 |
| `password` | 可读可写用户密码 |

::: details 外部集成 ClickHouse 配置示例

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

> 请保证上述配置与 Secret 中能对应上。

