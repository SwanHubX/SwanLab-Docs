# 自定义配置

> 本文档说明 `values.yaml` 中各字段的含义与配置方式。完整配置示例请参阅 [Kubernetes 部署指引](https://rcnpx636fedp.feishu.cn/wiki/KNHrwP796iscZOkMGldcSuPKnNb)。

## 全局配置（`global`）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `global.imagePullSecrets` | list | `[]` | 镜像拉取凭证（私有仓库认证） |
| `global.clusterDomain` | string | `cluster.local` | Kubernetes 集群域名 |
| `global.podAntiAffinityPreset` | string | `soft` | Pod 反亲和策略：`soft`（尽量分散）/ `hard`（强制分散）/ `none`（不设置） |
| `global.settings.loginHost` | string | `""` | 登录主机地址，修改后前端应用中显示的 API Key 登录地址会随之变化（不影响实际后端服务地址） |

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

### Server（后端核心服务）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `service.server.replicas` | int | `2` | 副本数 |
| `service.server.image.repository` | string | `repo.swanlab.cn/self-hosted/swanlab-server` | 镜像地址 |
| `service.server.image.tag` | string | `""` | 镜像标签，**置为空字符串**以自动同步 Chart 指定的版本号 |

### House（后端指标 OLAP 服务）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `service.house.replicas` | int | `2` | 副本数 |
| `service.house.image.repository` | string | `repo.swanlab.cn/self-hosted/swanlab-house` | 镜像地址 |
| `service.house.image.tag` | string | `""` | 镜像标签，**置为空字符串**以自动同步 Chart 指定的版本号 |
| `service.house.persistence.storageClass` | string | `""` | StorageClass |
| `service.house.persistence.storageSize` | string | `10Gi` | 存储卷大小 |

### Cloud（前端服务 - 旧版）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `service.cloud.replicas` | int | `1` | 副本数 |
| `service.cloud.image.repository` | string | `repo.swanlab.cn/self-hosted/swanlab-cloud` | 镜像地址 |
| `service.cloud.image.tag` | string | `""` | 镜像标签，**置为空字符串**以自动同步 Chart 指定的版本号 |

### Next（前端服务 - 新版）

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
| `*.customTolerations` | list | `[]` | 自定义容忍（Toleration） |
| `*.customNodeSelector` | object | `{}` | 节点选择器，JSON 格式，例如 `{ swanlab: "true" }` |
| `*.resources` | object | `{}` | 资源限制（requests/limits），例如 `{ requests: { cpu: "500m", memory: "512Mi" } }` |

## 内置基础服务（`dependencies`）

当 `integrations.<service>.enabled` 为 `false` 时，Chart 会在集群内自动部署以下组件。

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

## 外部 S3 集成（`integrations.s3`）

用于接入外部 S3 兼容对象存储（如阿里云 OSS、腾讯云 COS、AWS S3 等），要求 **必须兼容 AWS S3 协议**。

:::warning
如果您所在的云对象存储对 S3 协议的 endpoint 访问做了区分，需要请特别注意应该填写 S3 endpoint
:::
![s3-Config](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260602111133909.png)

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `integrations.s3.enabled` | bool | `true` | 启用后支持集成外部 S3（启用后 `dependencies.s3` 不会部署） |
| `integrations.s3.existingSecret` | string | `swanlab-secret-s3` | 存储 AK/SK 的 K8s Secret 名称 |

### Public 桶配置（`integrations.s3.public`）

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


### Private 桶配置（`integrations.s3.private`）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `private.ssl` | bool | `true` | 是否启用 SSL |
| `private.endpoint` | string | `""` | S3 接入点，**不带 bucket 前缀** |
| `private.region` | string | `""` | S3 地域 |
| `private.port` | int | `443` | 端口号 |
| `private.pathStyle` | bool | `true` | 路径访问方式 |
| `private.bucket` | string | `""` | 桶名称 |

> **建议**：Public 桶和 Private 桶分离使用。如云厂商对 S3 endpoint 访问做了区分，请特别注意填写正确的 `endpoint`。