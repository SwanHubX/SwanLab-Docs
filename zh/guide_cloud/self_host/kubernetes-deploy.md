# 使用Kubernetes进行部署

> 如需要从Docker版本迁移至Kubernetes版本，请参考[此文档](/guide_cloud/self_host/migration-docker-kubernetes.md)。  
> Kubernetes版本支持的SwanLab Python SDK版本为 >= 0.7.4

如果你想要使用 [Kubernetes](https://kubernetes.io/) 进行 SwanLab 私有化部署，请按照下面的流程进行安装。

![swanlab kubernetes logo](./kubernetes/logo.png)

---

[[toc]]

<br>

---

**资源和概念：**

- [SwanHubX/charts](https://github.com/SwanHubX/charts/tree/main/charts/self-hosted)：SwanLab的Kubernetes Helm Chart仓库
- `self-hosted`：部署好的SwanLab Kubernetes集群



## 先决条件

使用 Kubernetes 部署 SwanLab 私有化版本，请确保您的 Kubernetes 和相关基础设施满足如下要求：

| 软件/基础设施 | 版本/配置要求 | 必要性说明 |
| --- | --- | --- |
| kubernetes | v1.24 及以上 | 官方测试验证覆盖了 v1.24+ 版本。为确保 API 兼容性与系统稳定性，不建议在低于此版本的集群中部署。 |
| helm | version>=3 | SwanLab 采用 Helm v3 标准构建 Chart 包，不兼容 Helm v2（Tiller 模式）。 |
| RBAC 权限 | Namespace Admin | 部署账户需具备当前命名空间下的**写权限**。核心资源包括：`Deployment, StatefulSet, Service, PVC, Secret, ConfigMap`等。 |
| 网络访问 (Egress) | *.swanlab.cn | 集群节点需具备访问公网的能力（或配置 NAT 网关）：<br>1. `repo.swanlab.cn`：用于拉取应用镜像。  <br>2. `api.swanlab.cn`：用于 License 在线激活与校验。 |

## 1. 快速开始

您可以通过[helm](https://helm.sh/)安装SwanLab私有化服务K8S版。

首先建立本地仓库映射：

```bash
helm repo add swanlab https://helm.swanlab.cn
```

`swanlab`这个仓库将包含SwanLab官方开源的所有Charts，你可以使用如下命令安装SwanLab私有化服务：

```bash
helm install swanlab-self-hosted swanlab/self-hosted
```

:::warning 提示
您可以在[此处](https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml)查看`self-hosted`的所有可配置项，于此同时我们建议您保存好您的`values.yaml`
:::

通过安装 `swanlab/self-hosted`（下面简称`self-hosted`），即可在k8s上安装SwanLab私有化部署版应用，安装结果会在终端打印：

```bash
Release "swanlab-self-hosted" has been upgraded. Happy Helming!
NAME: swanlab-self-hosted
LAST DEPLOYED: Sat Dec 13 17:52:05 2025
NAMESPACE: self-hosted
STATUS: deployed
REVISION: 6
TEST SUITE: None
NOTES:
Thank you for installing self-hosted!

Get the application URL by running these commands:

1. Access via kube-proxy:
   Run the following command to forward your local port 8080 to the service:
     kubectl port-forward --namespace self-hosted svc/swanlab-self-hosted 8080:80

   Then, you can access the service via:
     http://127.0.0.1:8080

2. Expose Service Externally:
   SwanLab self-hosted is not exposed to the public internet.
   If you wish to expose this service, you need to configure a LoadBalancer manually or use an Ingress Controller.
   
   Please refer to the official documentation for configuration details:
   https://docs.swanlab.cn/guide_cloud/self_host/kubernetes-deploy.html
```

如上所示，`self-hosted`默认无法直接通过外部网络访问，您可以通过`port-forward`功能在本地访问此服务。如果您希望**开启外部访问（通过IP或域名）**，请参考[3.6 配置应用访问入口](/guide_cloud/self_host/kubernetes-deploy.md#_3-6-配置应用访问入口)。

下面是一个在本机访问的例子，打开终端并执行：

```bash
kubectl port-forward --namespace self-hosted svc/self-hosted 8080:80
```

然后你可以在浏览器中访问：`http://127.0.0.1:8080`，即可看到SwanLab的页面：

![](./docker-deploy/create-account.png)

现在，你需要激活你的主账号。激活需要1个License，个人使用可以免费在[SwanLab官网](https://swanlab.cn)申请一个，位置在 「设置」-「账户与许可证」。

![](./docker-deploy/apply-license.png)

拿到License后，回到激活页面，填写用户名、密码、确认密码和License，点击激活即可完成创建。

![](./docker-deploy/quick-start.png)


## 2. 资源清单

为了您更好理解SwanLab的服务状态本部分将列出SwanLab服务运行包含的所有部署资源和对应特征——`self-hosted`大致包含两种资源：基础服务资源以及应用服务资源。

### 2.1 基础服务资源

基础服务资源指的是数据库、对象存储等SwanLab应用依赖的必要资源，他们包括：

1. **PostgreSQL单实例**：存储SwanLab核心数据
2. **redis单实例**：存储服务cache
3. **clickhouse单实例**：存储实验日志资源
4. **minio单实例**：存储媒体资源
5. **traefik单实例**：网关和应用入口

### 2.2 应用服务资源

应用服务资源指的是SwanLab核心的业务资源——这些服务的镜像会跟随`self-hosted`版本更新变动——他们包括：

1. **SwanLab-Server**：SwanLab核心后端服务
2. **SwanLab-House**：SwanLab指标计算与分析服务
3. **SwanLab-Cloud**：SwanLab前端展示组件
4. **SwanLab-Next**：SwanLab前端展示组件
5. **Traefik-Proxy**：基于Traefik封装的网关组件

通常情况下，您可以随意改动这些应用服务资源的副本数量，所有的可配置字段都可以通过如下命令获取：

```bash
helm show values swanlab/self-hosted
```


## 3. 配置自定义资源

您可以在[此处](https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml)查看self-hosted的所有可配置项。在本部分将说明一些常用的、SwanLab官方推荐的配置实践。


### 3.1 自定义基础服务资源

如您所见，`self-hosted`部署的所有基础服务都为单实例，如果您在寻求企业级稳定性，这并不能满足需求。因此`self-hosted`支持外挂基础服务资源链接——你可以通过`integrations`部分配置他们。接下来分别讲述如何使用各种基础服务资源。

我们在[values.yaml](https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml)中撰写了详细的注释和密钥数据结构说明。需要注意的是，如果您将任一集成基础服务资源开启（例如设置`integrations.postgres.enabled`为`true`），`self-hosted`部署的单实例服务将被销毁。


#### 3.1.1 Postgres

如果您希望使用自己部署的cnpg集群或者使用云厂商的服务，您只需要：

1. 将`integrations.postgres.enabled`设置为`true`
2. 设置一个Secret，通过`integrations.postgres.existingSecret`传入此密钥名称，密钥的信息包括：

| `.data.<keys>` | 解释 |
| --- | --- |
| `username` | 可读可写用户名称 |
| `password` | 可读可写用户密码 |
| `primaryUrl` | 可读可写数据库连接串，格式类似于： `postgresql://{username}:${password}@postgres:5432/app?schema=public` |
| `replicaUrl` | 只读数据库连接串，一般用于负载均衡并且除了账号密码以外全部与primaryUrl相同。如果并没有配置只读用户/集群，可使用可读可写数据库连接串代替 |

3. 通过`values.yaml`设置`integrations.postgres`的其他配置

最终配置示例如下：
```yaml
integrations:
  postgres:
    enabled: true
    host: "example.postgres"
    port: 5432
    database: "app"
    existingSecret: integration-postgres
```

> 请保证上述配置与secret中能对应上


#### 3.1.2 Redis

如果您希望使用自己部署的redis集群或者使用云厂商的服务，您只需要：
1. 将`integrations.redis.enabled`设置为`true`
2. 设置一个Secret，通过`integrations.redis.existingSecret`传入此密钥名称，密钥的信息包括：

| `.data.<keys>` | 解释 |
| --- | --- |
| `url` | 数据库连接串，格式类似于： `redis://{username}:${password}@redis:6379` |

3. 通过`values.yaml`设置`integrations.redis`的其他配置

最终配置示例如下：

```yaml
integrations:
  redis:
    enabled: true
    host: "example.redis"
    port: 6379
    database: "0"
    existingSecret: integration-redis
```

> 请保证上述配置与secret中能对应上


#### 3.1.3 Clickhouse

如果您希望使用自己部署的clickhouse集群或者使用云厂商的服务，您只需要：
1. 将`integrations.clickhouse.enabled`设置为`true`
2. 设置一个Secret，通过`integrations.clickhouse.existingSecret`传入此密钥名称，密钥的信息包括：

| `.data.<keys>` | 解释 |
| --- | --- |
| `username` | 可读可写用户名称 |
| `password` | 可读可写用户密码 |

3. 通过`values.yaml`设置`integrations.clickhouse`的其他配置

最终配置示例如下：

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

> 请保证上述配置与secret中能对应上

    
#### 3.1.4 对象存储

如果您希望使用自己部署的minio集群或者使用云厂商的服务（必须兼容S3协议），您只需要：
1. 将`integrations.s3.enabled`设置为`true`
2. 设置一个Secret，通过`integrations.s3.existingSecret`传入此密钥的名称，密钥信息包括：

| `.data.<keys>` | 解释 |
| --- | --- |
| `accessKey` | 对象存储访问密钥 |
| `secretKey` | 对象存储密钥 |

3. 通过`values.yaml`设置`integrations.s3`的其他配置
最终配置示例如下：
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

> 请保证上述配置与secret中能对应上

:::warning
- publicBucket的权限为**公有读私有写**，privateBucket的权限为**私有读写**
- 当您选择自定义对象存储服务时，请保证您的对象存储服务在可以直接通过外部访问（通过IP或域名）
- 您的对象存储密钥必须对 **publicBucket** 和 **privateBucket** 同时有写权限和S3签名权限

:::


### 3.2 自定义存储资源

如果您希望使用`self-hosted`部署的单实例基础服务，那么我们建议您自己声明`storage-class`以支持数据持久化。

在进行自定义基础资源的存储类之前，请确保：
1. 这个基础服务资源并没有开启`integrations`
2. 确保您的`storage-class`或者`claim`存在于集群中


#### 3.2.1 基础服务资源的存储类

> 基础服务资源的定义，请见本文档 2.1

你可以通过`dependencies`部分配置基础服务资源。以`postgres`为例：

1. 如果您希望`self-hosted`生成存储卷，可以通过配置`dependencies.postgres.persistence`下的`storageClass`和`storageSize`配置存储卷类型和大小
2. 如果您已经有存储卷，可以通过`dependencies.postgres.persistence.existingClaim`配置一个已经存在的存储卷

通常，配置`dependencies.postgres.persistence.existingClaim`是一个比较推荐的做法，这将确保存储资源由您自己管理。


#### 3.2.2 应用服务资源的存储类

> 应用服务资源的定义，请见本文档 2.2

由于目前技术限制，`swanlab-house`以`StatefulSet`部署，因此您需要为它挂载存储卷。与配置基础服务资源类似，您需要配置`service.house.persistence`下的字段。需要注意的是，这里并不允许配置`existingClaim`。

:::warning
`swanlab-house`会在存储卷下存储一些指标中间产物，一般情况下您不需要关心此存储卷中的数据。
我们会在未来移除这一设计。
:::


### 3.3 增加应用副本以提高服务质量

我们为`service`字段下的所有服务提供了`replicas`接口，您可以自由更改其数量，根据SwanLab的运维经验，绝大多数场景下：
1. `server`服务的副本数量为3
2. `house`服务的副本数量为3
3. `next`服务的副本数量为2
4. `cloud`服务的副本数量为1
当然，应用性能是一个复杂的计算指标，通常它还取决于资源限制，我们也提供了`resources`等接口允许您配置应用的资源用量。

### 3.4 更改Pod反亲和性

您可以通过设置`global.podAntiAffinityPreset`来设置Pod反亲和性以提升容灾能力：

```yaml
global:
  # Kubernetes Pod Affinity/Anti-Affinity settings
  # We use Topology Spread Constraints to achieve pod distribution across nodes.
  podAntiAffinityPreset: "soft" # soft, hard, or none
```
默认情况为`soft`，这意味着所有Pod将均匀分布在各个Node上。您可以将其设置为`hard`以确保同一服务Pod不会分布在同一Node上，或者设置为`none`以禁用Pod反亲和性。


### 3.5 定义声明、标签等元数据

对于任一服务，我们定义了如下接口以方便您调度SwanLab应用容器：

1. `resources`：限定服务所需CPU和RAM资源
2. `customLabels`：自定义应用标签
3. `customAnnotations`：自定义应用注解
4. `customTolerations`：自定义容忍度
5. `customNodeSelector`：自定义节点选择器

您可以通过这些资源自由管理和调度SwanLab应用。

### 3.6 配置应用访问入口

应用服务在集群中的域名为您部署的release名称，举个例子，假设您的`cluster domain`为`cluster.local`，部署命令为：

```bash
helm install swanlab-self-hosted swanlab/self-hosted -n self-hosted
```

- 应用在`self-hosted`命名空间下的域名为`swanlab-self-hosted`
- 应用在`kubernetes`集群下的域名为：`swanlab-self-hosted.self-hosted.svc.cluster.local`

您可以基于如上信息编写负载均衡策略。通常建议您优先使用 **独立域名（Host-based）** 来配置访问策略，以规避因路径规则复杂或变更导致的路由冲突。

**基于架构解耦原则**，Self-hosted 不内置 Ingress 控制器。您需要在集群的负载均衡器（或 Ingress）上配置外部访问入口，并由其负责 **TLS 终止（HTTPS 卸载）**。

**在安全策略方面**，应用默认信任所有的 `X-Forwarded-*` 请求头。如果您需要更严格的头部校验或转发控制，请务必在负载均衡层统一实施——这有可能影响到内部S3的签名效果，如果您使用外部对象存储服务，则不需要有这方面担忧。


### 3.7 更改swanlab.login显示的域名

默认情况下，`<Your Host>/space/~/quick-start` 页面中显示的 **login host** 会自动使用您当前访问前端的域名 `<Your Host>`。

如果您需要修改此值，可以通过配置 `global.settings.loginHost` 将其指定为您期望的域名。**请注意**，此设置不会影响实际的后端服务地址，您需要自己配置对应的转发规则。

### 3.8 更新、回滚服务

#### 更新

我们会更新helm chart版本，这将包含一些新功能和错误修复，您可以选择在适当的时机更新您部署的服务。

:::warning 
在更新前，请您确保已备份对应的PVC数据！
:::

您可以运行如下命令升级您的本地repo，其中应该包含swanlab repo：

```bash
# 更新仓库
helm repo update

# 列出所有版本
helm search repo swanlab/self-hosted --versions
```

然后选择您所需的版本号完成更新：

```bash
helm upgrade self-hosted swanlab-charts/self-hosted \
  --version x.x.x \
  -f my-values.yaml \
  --namespace xxx
```

#### 回滚

如果在更新后发现服务无法启动（如 CrashLoopBackOff），请通过以下命令立即回滚至上一个稳定版本：

```bash
helm rollback self-hosted x.x.x -n xxx
```

### 3.9 接入Prometheus

SwanLab的应用服务暂不支持接入`Prometheus`，此功能正在开发中，敬请期待！

### 4. FAQ

#### 部署服务是否需要较高部署权限（如部署 CRD 或 Controller）？
不需要。

#### 迁移数据时能否保证原服务不停机？
迁移过程中必须停机原服务。如果不停止原服务会出现数据 gap。  
此时可考虑使用[swanlab sync](/api/cli-swanlab-sync.md)将数据上传至新服务。

#### 集群无法连接外网，如何下载、更新镜像？
您可手动下载 swanlab 系列服务镜像，并上传至内网镜像仓库，包括：

- `service.gateway.image`：swanlab 网关服务，整个应用服务入口  
- `service.gateway.identifyImage`：swanlab 网关辅助镜像  
- `service.helper`：帮助部署 swanlab 服务，本质为 busybox  
- `service.server.image`：swanlab 后端服务，镜像标签为 Chart 的 AppVersion  
- `service.house.image`：swanlab 后端服务，镜像标签为 Chart 的 AppVersion  
- `service.house.logImage`：swanlab-house 辅助镜像  
- `service.house.fbImage`：swanlab-house 辅助镜像  
- `service.cloud.image`：swanlab 前端服务，镜像标签为 Chart 的 AppVersion  
- `service.next.image`：swanlab 前端服务，镜像标签为 Chart 的 AppVersion  
- `dependencies.postgres.image`：Postgres 数据库镜像（选择[自定义基础服务资源](/guide_cloud/self_host/kubernetes-deploy.md#_3-1-自定义基础服务资源)可忽略此项）  
- `dependencies.redis.image`：Redis 数据库镜像（选择[自定义基础服务资源](/guide_cloud/self_host/kubernetes-deploy.md#_3-1-自定义基础服务资源)可忽略此项）  
- `dependencies.s3.image`：MinIO 对象存储镜像（选择[自定义基础服务资源](/guide_cloud/self_host/kubernetes-deploy.md#_3-1-自定义基础服务资源)可忽略此项）  
- `dependencies.clickhouse.image`：ClickHouse 数据库镜像（选择[自定义基础服务资源](/guide_cloud/self_host/kubernetes-deploy.md#_3-1-自定义基础服务资源)可忽略此项）

#### SwanLab 用到的所有存储组件及用途有哪些？
swanlab所有存储组件与用途如下：

- `service.house.persistence`  
  辅助 swanlab 数据写入服务实现高可用，未来版本将删除。[迁移服务](/guide_cloud/self_host/migration-docker-kubernetes.md)时无需迁移该卷数据。

- `dependencies.postgres.persistence`  
  用于 Postgres 数据库持久化（选择[自定义基础服务资源](/guide_cloud/self_host/kubernetes-deploy.md#_3-1-自定义基础服务资源)可忽略此项）。

- `dependencies.redis.persistence`  
  用于 Redis 数据库持久化（选择[自定义基础服务资源](/guide_cloud/self_host/kubernetes-deploy.md#_3-1-自定义基础服务资源)可忽略此项）。

- `dependencies.s3.persistence`  
  用于对象存储服务持久化（选择[自定义基础服务资源](/guide_cloud/self_host/kubernetes-deploy.md#_3-1-自定义基础服务资源)可忽略此项）。

- `dependencies.clickhouse.persistence`  
  用于 ClickHouse 数据库持久化（选择[自定义基础服务资源](/guide_cloud/self_host/kubernetes-deploy.md#_3-1-自定义基础服务资源)可忽略此项）。
