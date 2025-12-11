# 使用Kubernetes进行部署

如果你想要使用 [Kubernetes](https://kubernetes.io/) 进行 SwanLab 私有化部署，请按照下面的流程进行安装。

![](./kebunetes/logo.png)

---

[[toc]]

## 先决条件

使用 Kubernetes 部署 SwanLab 私有化版本，请确保您的 Kubernetes 和相关基础设施满足如下要求：

| 软件/基础设施 | 要求 | 解释 |
| --- | --- | --- |
| kubernetes | version>=1.24 | SwanLab仅在此版本以上做过测试，较低版本并不保证使用 |
| helm | version>=3 | SwanLab使用helm3编写swanlab的kubernetes软件包 |
| NAT白名单 | 允许集群访问swanlab.cn根域名和子域名 | 集群需要通过`repo.swanlab.cn`拉取镜像，并且商业版需要通过`api.swanlab.cn`完成License校验 |

## 1. 快速开始

您可以通过[helm](https://helm.sh/)安装SwanLab私有化服务K8S版。

首先建立本地仓库映射：

```bash
helm repo add swanlab https://helm.swanlab.cn
```

swanlab这个仓库将包含SwanLab官方开源的所有Charts，你可以使用如下命令安装SwanLab私有化服务：

```bash
helm install swanlab-self-hosted swanlab/self-hosted
```

通过安装 `swanlab/self-hosted`（下面简称`self-hosted`)，即可在k8s上安装SwanLab私有化部署版应用。

> 您可以在[此处](https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml)查看self-hosted的所有可配置项。


## 2. 资源清单

为了您更好理解SwanLab的服务状态本部分将列出SwanLab服务运行包含的所有部署资源和对应特征——`self-hosted`大致包含两种资源：基础服务资源以及应用服务资源。

### 2.1 基础服务资源

基础服务资源指的是数据库、对象存储等SwanLab应用依赖的必要资源，他们包括：

1. **PostgreSQL单实例**：存储SwanLab核心数据
2. **redis单实例**：存储服务cache
3. **clickhouse单实例**：存储实验日志资源
4. **minio单实例**：存储媒体资源

### 2.2 应用服务资源

应用服务资源指的是SwanLab核心的业务资源——这些服务的镜像会跟随self-hosted版本更新变动——他们包括：

1. **Swanlab-Server**：SwanLab核心后端服务
2. **SwanLab-House**：SwanLab日志分析服务
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


#### 3.1.1 自定义Postgress

如果您希望使用自己部署的cnpg集群或者使用云厂商的服务，您只需要：

1. 将`integrations.postgres.enabled`设置为`true`
2. 设置一个Secret，通过`integrations.postgres.existingSecret`传入此密钥名称，密钥的信息包括：

| `.data.<keys>` | 解释 |
| --- | --- |
| `database` | 使用的数据库名称，我们推荐设置为app |
| `username` | 可读可写用户名称 |
| `password` | 可读可写用户密码 |
| `primaryUrl` | 可读可写数据库连接串，格式类似于： `postgresql://{username}:${password}@postgres:5432/app?schema=public` |
| `replicaUrl` | 只读数据库连接串，一般用于负载均衡并且除了账号密码以外全部与primaryUrl相同。如果并没有配置只读用户/集群，可使用可读可写数据库连接串代替 |


#### 3.1.2 自定义Redis

如果您希望使用自己部署的redis集群或者使用云厂商的服务，您只需要：
1. 将`integrations.redis.enabled`设置为`true`
2. 设置一个Secret，通过`integrations.redis.existingSecret`传入此密钥名称，密钥的信息包括：

| `.data.<keys>` | 解释 |
| --- | --- |
| `url` | 数据库连接串，格式类似于： `redis://{username}:${password}@redis:6379` |


#### 3.1.3 自定义Clickhouse

如果您希望使用自己部署的clickhouse集群或者使用云厂商的服务，您只需要：
1. 将`integrations.clickhouse.enabled`设置为`true`
2. 设置一个Secret，通过`integrations.clickhouse.existingSecret`传入此密钥名称，密钥的信息包括：

| `.data.<keys>` | 解释 |
| --- | --- |
| `database` | 使用的数据库名称，我们推荐设置为app |
| `username` | 可读可写用户名称 |
| `password` | 可读可写用户密码 |
| `host` | clickhouse服务地址 |
| `httpPort` | Clickhouse http服务端口，一般为9000 |
| `tcpPort` | Clickhouse tcp服务端口，一般为8123 |


#### 3.1.4 自定义对象存储

如果您希望使用自己部署的minio集群或者使用云厂商的服务，您只需要：
1. 将`integrations.s3.enabled`设置为`true`
2. 设置一个Secret，通过`integrations.s3.existingSecret`传入此密钥的名称，密钥信息包括：

| `.data.<keys>` | 解释 |
| --- | --- |
| `accessKey` | 对象存储访问密钥 |
| `secretKey` | 对象存储密钥 |
| `endpoint` | 对象存储地址 |
| `privateBucket` | 私有桶名称，我们推荐设置为swanlab-private |
| `publicBucket` | 公有桶名称，我们推荐设置为swanlab-public |
| `region` | 对象存储地域，如果您使用自己部署的minio等服务，可能没有此字段，设置为local即可 |

3. 确保密钥中配置的`privateBucket`和`publicBucket`已经在对象存储服务中存在

:::warning
如果您打算在集群中配置自己的minio或者其他对象存储服务，您应该保证此服务对公网可访问——因为SwanLab前端服务也需要访问此服务，并且self-hosted默认不会配置第三方服务的负载均衡策略。
:::


### 3.2 自定义存储资源

如果您希望使用self-hosted部署的单实例基础服务，那么我们建议您自己声明storage-class以支持数据持久化，因为self-hosted默认使用local-storage声明PVC。

在进行自定义基础资源的存储类之前，请确保：
1. 这个基础服务资源并没有开启`integrations`
2. 确保您的storage-class或者claim存在于集群中


#### 3.2.1 自定义基础服务资源的存储类

你可以通过`dependencies`部分配置基础服务资源。以postgres为例：

1. 如果您希望self-hosted生成存储卷，可以通过配置`dependencies.postgres.persistence`下的`storageClass`和`storageSize`配置存储卷类型和大小
2. 如果您已经有存储卷，可以通过`dependencies.postgres.persistence.existingClaim`配置一个已经存在的存储卷

通常，配置`dependencies.postgres.persistence.existingClaim`是一个比较推荐的做法，这将确保存储资源由您自己管理。


#### 3.2.2 自定义应用服务资源的存储类

由于目前技术限制，`swanlab-house`以StatefulSet部署，因此您需要为它挂载存储卷。与配置基础服务资源类似，您需要配置`service.house.persistence`下的字段。需要注意的是，这里并不允许配置`existingClaim`。

:::warning
`swanlab-house`会在存储卷下存储一些指标中间产物，一般情况下您不需要关心此存储卷中的数据。
我们会在未来移除这一设计。
:::


### 3.3 增加应用副本以提高服务质量

我们为`services`字段下的所有服务提供了`replicas`接口，您可以自由更改其数量，根据SwanLab的运维经验，绝大多数场景下：
1. `server`服务的副本数量为3
2. `house`服务的副本数量为3
3. `next`服务的副本数量为2
4. `cloud`服务的副本数量为1
当然，应用性能是一个复杂的计算指标，通常它还取决于资源限制，我们也提供了`resources`等接口允许您配置应用的资源用量。


### 3.4 定义声明、标签等元数据

对于任一服务，我们定义了如下接口以方便您调度SwanLab应用容器：
1. `customLabels`：自定义应用标签
2. `customAnnotations`：自定义应用注解
3. `customTolerations`：自定义容忍度
4. `customNodeSelector`：自定义节点选择器
您可以通过这些资源自由管理和调度SwanLab应用。


### 3.5 配置自定义负载均衡、域名并提供TLS服务

`self-hosted`本身不提供ingress，在k8s中您需要使用外部负载均衡访问。首先，确保您已更改应用服务类型为NodePort或ClusterIP；此外，为了避免不必要的意外，通常我们推荐您在负载均衡器侧完成TLS终止，并要求负载均器传递`X-Forwarded-*`相关的请求头。

最后，除了在您自己的负载均衡器上配置对应流量转发规则以外，请配置traefik信任上游服务设置的`X-Forwarded-*`请求头，参考[此文档](https://doc.traefik.io/traefik/reference/install-configuration/entrypoints/#opt-forwardedHeaders-trustedIPs)：

```yaml
ingress:
  traefik:
    ports:
      web:
        forwardedHeaders:
          trustedIPs: [] # 在此设置您上游负载均衡器内网IP
```


### 3.6 更改swanlab.login显示的域名

默认情况下，`<Your Host>/space/~/quick-start` 页面中显示的 login host 会自动使用您当前访问前端的域名 `<Your Host>`。

如果您需要修改此值，可以通过配置 `env.apiHost` 将其指定为您期望的域名。

需要注意的是，这一配置仅仅是显示上的更改而不会作用到实际的路由转发规则上。此外，这一配置与ingress.host存在冲突，后者将会配置严格的域名转发规则以导致客户端无法访问env.apiHost——在这种情况下，我们建议您在self-hosted服务上层部署负载均衡器接管流量转发规则和实现TLS终止，详见更改应用服务类型。


### 3.7 接入Prometheus

SwanLab的应用服务暂不支持接入`Prometheus`，此功能正在开发中，敬请期待！