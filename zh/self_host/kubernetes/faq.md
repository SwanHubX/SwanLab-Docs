# 常见问题

> 本文档记录了 SwanLab K8s 私有化版本部署过程中的常见问题。

## 【权限要求】部署服务是否需要较高部署权限（如部署 CRD 或 Controller）？
- 不需要。

## 【数据迁移】迁移数据时能否保证原服务不停机？
- ❌ **不能**

迁移过程中必须停机原服务。如果不停止原服务会出现数据 gap。  
此时可考虑使用[swanlab sync](/api/cli-swanlab-sync.md)将数据上传至新服务。


## 【副本数】SwanLab 私有化服务推荐的服务副本数如何设置？

以下是根据线上运维经验给出的推荐副本配置最佳实践，在 `values.yaml` 中通过修改对应服务的 `replicas` 字段即可调整：

::: warning
当前 `swanlab-self-hosted` 部署方案中，`postgres` 和 `clickhouse` 均采用**单副本方案**。数据库主从复制涉及较大的架构变动，在当前私有化部署版本中**暂不支持**，请勿调整数据库相关的服务副本数量。
:::


| 服务名 | 副本数量 | 说明 |
|--------|---------|------|
| clickhouse | 1 | 【不可修改】列数据库，负责实验指标存储 |
| postgres | 1 | 【不可修改】关系型数据库，负责元数据和关系记录 |
| redis | 1 | 【不可修改】内存数据库，缓存会话数据 |
| vector | 2 | 【不可修改】clickhouse 指标写入缓冲队列 |
| traefik | 2 | 【按需修改】主网关，分发服务流量 |
| swanlab-server | ≥ 3 | 【按需修改】SwanLab 核心服务，根据服务负载动态调整 |
| swanlab-house | ≥ 3 | 【按需修改】SwanLab 指标分析服务，根据服务负载动态调整 |
| swanlab-next | 2 | 【按需修改】SwanLab 前端框架 |
| swanlab-cloud | 1 | 【按需修改】SwanLab 前端实验页面 |


## 【节点指定】如何将 SwanLab 私有化服务 Pod 调度到指定节点？

在 `values.yaml` 中，所有服务均支持通过 `customNodeSelector` 字段指定节点选择器，Kubernetes 只会将 Pod 调度到满足对应标签的节点上。

**给节点打标签**：

```bash
kubectl label nodes <node-name> swanlab=true
```

**示例**：将 SwanLab Server 调度到带有 `swanlab=true` 标签的节点：

```yaml
service:
  server:
    customNodeSelector: { "swanlab": "true" }
```

网关同样支持：

```yaml
gateway:
  customNodeSelector: { "swanlab": "true" }
```

如需在存在污点（Taint）的节点上运行，可配合 `customTolerations` 一起使用：

```yaml
service:
  server:
    customNodeSelector: { "swanlab": "true" }
    customTolerations:
      - key: "dedicated"
        operator: "Equal"
        value: "swanlab"
        effect: "NoSchedule"
```

::: tip
`customNodeSelector` 与 `customTolerations` 为所有服务的通用字段，包括应用服务（`gateway`、`vector`、`service.server`、`service.house`、`service.cloud`、`service.next`）和基础服务（`dependencies.postgres`、`dependencies.redis`、`dependencies.clickhouse`、`dependencies.s3`），按需为各服务单独配置即可。
:::


## 【资源限制】如何限制 SwanLab 服务的 CPU 和 内存用量？

在 `values.yaml` 中，所有应用服务均支持通过 `resources` 字段设置 CPU 和内存的 Requests / Limits，格式与 Kubernetes 原生 `resources` 一致。

**示例**：限制 SwanLab Server 的资源用量：

```yaml
service:
  server:
    resources:
      requests:
        cpu: "2"
        memory: "2Gi"
      limits:
        cpu: "4"
        memory: "4Gi"
```

各服务均可按需配置，未设置时默认不限制。基础服务（`dependencies.postgres`、`dependencies.redis`、`dependencies.clickhouse`、`dependencies.s3`）同样支持 `resources` 字段。


## 【镜像类】集群无法连接外网，如何下载、更新镜像？

- 您可**提前在公网环境**中，手动从 SwanLab 公共镜像仓库（即 `repo.swanlab.cn`）拉取全部所需镜像 (`docker pull`)，并上传至内网私有镜像仓库(`docker push`)。


## 【高可用】如何保障服务高可用与数据安全性？
根据数据库配置，主要针对两种情况进行分别设置：

「✅ **推荐**」针对使用本地数据库的情况：

- 部署过程中，每一个 PVC 申请对应一块**独立云SSD硬盘**，支持无感扩容。
- 由云硬盘本身做持久化存储，配置以天为单位的 **快照策略**，TTL 过期时间建议设置 2~7 天，保证每日数据可靠性。

「⚠️ **暂不推荐**」针对外接云数据库的情况：
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
- **返回 Vary:Origin**

### public 桶ACL配置
- SwanLab 的默认用户头像使用「**彩色 SVG**」，如果无法正确显示，一般是 public 桶的公共读被关闭，以 阿里云 OSS 为例，可以在如下设置中开启
- 「**权限控制**」 -> 「**阻止公共访问**」，将按钮关闭
<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260602120512627.png" width="80%"/>

- 「**权限控制**」 ->  「**读写权限**」，开启公有读的 Bucket ACL
<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260602121636204.png" width="80%"/>


