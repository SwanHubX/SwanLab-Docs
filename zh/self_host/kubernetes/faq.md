# 常见问题


## 【权限要求】部署服务是否需要较高部署权限（如部署 CRD 或 Controller）？
- 不需要。

## 【数据迁移】迁移数据时能否保证原服务不停机？
- ❌ **不能**

迁移过程中必须停机原服务。如果不停止原服务会出现数据 gap。  
此时可考虑使用[swanlab sync](/api/cli-swanlab-sync.md)将数据上传至新服务。

## 【镜像类】SwanLab 私有化版用到了哪些镜像？
### SwanLab 应用服务镜像
> ⚠️ 注意： `value.yaml` 应用镜像的 tag **默认设置为空字符串**，可以从 template 中自动同步最新的版本号作为镜像标签，一般无需修改。特殊热更新补丁版本镜像需要手动填充 tag。

| 组件 | 镜像地址 | values.yaml 配置路径 | 说明 |
|------|----------|---------------------|------|
| swanlab-server | `repo.swanlab.cn/self-hosted/swanlab-server:<APP_VERSION>` | `service.server.image` | 后端核心服务 |
| swanlab-house | `repo.swanlab.cn/self-hosted/swanlab-house:<APP_VERSION>` | `service.house.image` | 后端实验指标OLAP服务 |
| swanlab-cloud | `repo.swanlab.cn/self-hosted/swanlab-cloud:<APP_VERSION>` | `service.cloud.image` | 前端实验图表渲染组件 |
| swanlab-next | `repo.swanlab.cn/self-hosted/swanlab-next:<APP_VERSION>` | `service.next.image` | 前端UI |

### SwanLab 基础设施镜像
> ⚠️注意：当某一项存储组件 选择[自定义基础服务资源](/self_host/kubernetes/deploy.md#_3-1-自定义基础服务资源)时，以下对应镜像可忽略（使用自建的外部服务）。

::: warning
SwanLab 私有化版本服务的数据库采用单实例模式，未来在架构上会有变更。**为保证架构与测试行为的一致性**，除 **S3 对象存储** 外，我们 **暂不推荐使用云数据库**进行接入，推荐使用 **云硬盘SSD** 作为对应基础服务的 PVC 存储资源的 storageClass 。
:::

| 组件 | 镜像地址 | values.yaml 配置路径 | 说明 |
|------|----------|---------------------|------|
| Traefik | `repo.swanlab.cn/public/traefik:3.6` | `service.gateway.image` | 反向代理 / 网关入口 |
| Identify Helper | `repo.swanlab.cn/public/swanlab-helper/identify:v1.2` | `service.gateway.identifyImage` | 网关鉴权辅助镜像 |
| Busybox | `repo.swanlab.cn/public/busybox:1.37.0` | `service.helper` | 部署辅助初始化容器 |
| Vector | `repo.swanlab.cn/public/vector:0.51.1-debian` | — | 实验指标采集缓冲队列 |
| PostgreSQL | `repo.swanlab.cn/self-hosted/postgres:16.1` | `dependencies.postgres.image` | PostgreSQL关系型数据库（用户、项目、实验元数据） |
| Redis | `repo.swanlab.cn/self-hosted/redis-stack:7.4.0-v8` | `dependencies.redis.image` | 缓存与会话存储 |
| ClickHouse | `repo.swanlab.cn/self-hosted/clickhouse-server:24.3` | `dependencies.clickhouse.image` | 实验指标与日志列数据库 |
| MinIO | `repo.swanlab.cn/self-hosted/minio/minio:RELEASE.2025-09-07T16-13-09Z` | `dependencies.s3.image` | S3 兼容对象存储（实验媒体资源与导出日志文件） |
| MinIO MC | `repo.swanlab.cn/self-hosted/minio/mc:RELEASE.2025-08-13T08-35-41Z` | — | MinIO 客户端工具（初始化 bucket 等） |



## 【镜像类】集群无法连接外网，如何下载、更新镜像？

- 您可**提前在公网环境**中，手动从 SwanLab 公共镜像仓库（即 `repo.swanlab.cn`）拉取全部所需镜像 (`docker pull`)，并上传至内网私有镜像仓库(`docker push`)。


## 【高可用】如何保障服务高可用与数据安全性？
根据数据库配置，主要针对两种情况进行分别设置：

「**推荐**」针对使用本地数据库的情况：

- 部署过程中，每一个 PVC 申请对应一块**独立云SSD硬盘**，支持无感扩容。
- 由云硬盘本身做持久化存储，配置以天为单位的 **快照策略**，TTL 过期时间建议设置 2~7 天，保证每日数据可靠性。

「**暂不推荐**」针对外接云数据库的情况：
- 可由 IaaS 公有云本身的数据库主从同步进行保障，可联系各公有云厂商的云数据库产品技术支持、或自建集群的 DBA 进行相关对接配置。


## 【对象存储】实验图片上传失败/CSV和日志无法下载/头像显示异常？
此类问题与 `S3对象存储` 配置问题强相关，可以在 `swanlab-house` 对应的 pod 中定位到对应的服务报错日志，推荐排查顺序:

###  `value.yaml` 配置校验
- 首先校验一下 `integrations.s3` 中的配置是否正确，详见 [外部 S3 集成配置](/self_host/kubernetes/configuration.md#外部-s3-集成-integrations-s3)


### 存储桶跨域规则配置
- 以阿里云 OSS 对象存储为例，配置示例为：

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260602112952339.png" width="80%"/>

- **来源**：建议放开到您公司内网域名的最顶级域名，如：您的内网域名为 ：`domain.com`，那么可以将来源设置为 `*.domain.com`
- **允许 Methods**: GET, POST, PUT, HEAD
- **允许 Headers**：填写 * 通配符

### public 桶ACL配置
- SwanLab 的默认用户头像使用「**彩色 SVG**」，如果无法正确显示，一般是 public 桶的公共读被关闭，以 阿里云 OSS 为例，可以在如下设置中开启
- 「**权限控制**」 -> 「**阻止公共访问**」，将按钮关闭
<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260602120512627.png" width="80%"/>

- 「**权限控制**」 ->  「**读写权限**」，开启公有读的 Bucket ACL
<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260602121636204.png" width="80%"/>