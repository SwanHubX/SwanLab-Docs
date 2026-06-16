# 使用Kubernetes进行部署



> 如需要从Docker版本迁移至Kubernetes版本，请参考[此文档](/self_host/docker/migration-docker-kubernetes.md)。  
> Kubernetes版本支持的SwanLab Python SDK版本为 >= 0.7.4

如果你想要使用 [Kubernetes](https://kubernetes.io/) 进行 SwanLab 私有化部署，请按照下面的流程进行安装。

![swanlab kubernetes logo](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/zh/guide_cloud/self_host/kubernetes/logo.png)

[[toc]]

<br>

**资源和概念：**

- [SwanHubX/charts - 私有化服务发布地址](https://github.com/SwanHubX/charts/releases)：SwanLab的Kubernetes Helm Chart仓库
- `swanlab-self-hosted`: 在集群中部署的 SwanLab 私有化服务的**默认 RELEASE 名称** 
- `<your_namespace>`: 在集群中部署的 SwanLab 私有化服务的命名空间，请替换为部署使用的命名空间

::: info
**当前APP_VERSION: v2.8.1**
:::



## 🧱 先决条件

使用 Kubernetes 部署 SwanLab 私有化版本，请确保您的 Kubernetes 和相关基础设施满足如下要求：

| 软件/基础设施 | 版本/配置要求 | 必要性说明 |
| --- | --- | --- |
| kubernetes | v1.24 及以上 | 官方测试验证覆盖了 v1.24+ 版本。为确保 API 兼容性与系统稳定性，不建议在低于此版本的集群中部署。 |
| helm | version>=3.9 | SwanLab Chart 包依赖于 Helm v3.9+ 的新特性，与早期版本不兼容，也不兼容 Helm v2（Tiller 模式）。 |
| RBAC 权限 | Namespace Admin | 账户需具备部署 **SwanLab私有化服务对应命名空间下的写权限**。核心资源包括：`Deployment, StatefulSet, Service, PVC, Secret, ConfigMap`等。 |
| 网络访问 (Egress) | *.swanlab.cn | 集群节点需具备访问公网的能力（或配置 NAT 网关）：<br>1. `repo.swanlab.cn`：用于拉取应用镜像。  <br>2. `api.swanlab.cn`：用于 License 在线激活与校验。 |
| 对象存储 | 兼容 AWS S3 协议 | SwanLab 上报的媒体资源等文件默认保存在对象存储中，为节约存储成本，推荐**外部集成对象存储**，确保兼容 S3 API |

## 🧾 资源清单
### 应用服务镜像
> ⚠️ 注意： `value.yaml` 应用镜像的 tag **默认设置为空字符串**，可以从 template 中自动同步最新的版本号作为镜像标签，一般无需修改。特殊热更新补丁版本镜像需要手动填充 tag。

| 组件 | 镜像地址 | values.yaml 配置路径 | 说明 |
|------|----------|---------------------|------|
| swanlab-server | `repo.swanlab.cn/self-hosted/swanlab-server:<APP_VERSION>` | `service.server.image` | 后端核心服务 |
| swanlab-house | `repo.swanlab.cn/self-hosted/swanlab-house:<APP_VERSION>` | `service.house.image` | 后端实验指标OLAP服务 |
| swanlab-cloud | `repo.swanlab.cn/self-hosted/swanlab-cloud:<APP_VERSION>` | `service.cloud.image` | 前端实验图表渲染组件 |
| swanlab-next | `repo.swanlab.cn/self-hosted/swanlab-next:<APP_VERSION>` | `service.next.image` | 前端UI |

### 基础设施镜像
> ⚠️注意：当某一项存储组件 选择[自定义基础服务资源](/self_host/kubernetes/deploy.md#_3-1-自定义基础服务资源)时，以下对应镜像可忽略（使用自建的外部服务）。

::: warning
SwanLab 私有化版本服务的**数据库采用单实例模式**，未来在架构上会有变更。**为保证架构与测试行为的一致性**，除 **S3 对象存储** 外，我们 **暂不推荐使用云数据库**进行接入，推荐使用 **云硬盘SSD** 作为对应基础服务的 PVC 存储资源的 storageClass 。
:::

| 组件 | 镜像地址 | values.yaml 配置路径 | 说明 |
|------|----------|---------------------|------|
| Traefik | `repo.swanlab.cn/public/traefik:3.6` | `service.gateway.` | 反向代理 / 网关入口 |
| Identify | `repo.swanlab.cn/public/swanlab-helper/identify:v1.2` | `service.gateway.identifyImage` | 网关鉴权辅助镜像 |
| Busybox | `repo.swanlab.cn/public/busybox:1.37.0` | `helper.image` | 部署辅助初始化容器 |
| Vector | `repo.swanlab.cn/public/vector:0.51.1-debian` | `vector.image` | 实验指标采集缓冲队列 |
| PostgreSQL | `repo.swanlab.cn/self-hosted/postgres:16.1` | `dependencies.postgres.image` | PostgreSQL关系型数据库（用户、项目、实验元数据） |
| Redis | `repo.swanlab.cn/self-hosted/redis-stack:7.4.0-v8` | `dependencies.redis.image` | 缓存与会话存储 |
| ClickHouse | `repo.swanlab.cn/self-hosted/clickhouse-server:24.3` | `dependencies.clickhouse.image` | 实验指标与日志列数据库 |
| MinIO | `repo.swanlab.cn/self-hosted/minio/minio:RELEASE.2025-09-07T16-13-09Z` | `dependencies.s3.image` | S3 兼容对象存储（实验媒体资源与导出日志文件） |
| MinIO MC | `repo.swanlab.cn/self-hosted/minio/mc:RELEASE.2025-08-13T08-35-41Z` | `dependencies.s3.mcImage` | MinIO 客户端工具（初始化 bucket 等） |



## 🪜 安装指引
::: info
本指引按照 SwanLab K8s 私有化服务的最佳实践进行安装，推荐 **【外部集成 S3 对象存储】 + 【集群数据库通过 PVC 挂载云硬盘】** 的方案。如您有特殊集成需求，可参考 [自定义 value 配置](/self_host/kubernetes/configuration.md) 进行修改
:::

### 1. 创建 S3 Secret
推荐使用已有的在线对象存储服务（必须兼容AWS S3协议），请确保您已创建好**具有可写权限的 AK/SK**，参考下列 yaml: 
::: details swanlab-self-hosted-secret.yaml 模板
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-self-hosted-secret-s3
  namespace: <your_namespace> # 请替换为实际使用的命名空间
type: Opaque
stringData:
  accessKey: xxxx
  secretKey: xxxx

```
:::

```bash
# 在集群中创建 secret
kubectl apply -f swanlab-self-hosted-secret.yaml

# 查看 secret
kubectl get secret swanlab-self-hosted-secret-s3 -n <your_namespace>
```

### 2. 创建 PVC

SwanLab 私有化服务中，主要依赖以下基础服务的存储资源：

| 基础资源 | PVC 命名 | 存储大小推荐 |
|---------|----------|-------------|
| Redis | `swanlab-redis-pvc` | ≥ 50G |
| PostgreSQL | `swanlab-postgres-pvc` | ≥ 100G |
| ClickHouse | `swanlab-clickhouse-pvc` | ≥ 1000G |
| Vector | `data-swanlab-self-hosted-vector-0` / `data-swanlab-self-hosted-vector-1` | 每个副本各自 ≥ 60G， 两块独立存储|
| 「**可选**」MinIO | `swanlab-minio-pvc` / `data-swanlab-self-hosted-vector-1` | ≥ 1000G，建议优先使用在线对象存储 |

::: warning
- `storageClassName` 应以**您集群中挂载的云硬盘类型为准**（例如：腾讯云默认的云硬盘 `cbs`），要求**支持动态扩容与快照策略**
- Vector 部署为 `StatefulSet`，PVC名称默认**不可修改**
- 务必确保 `postgresr/redis/clickhouse` 相关的 PVC 均已处于 **Bound** 状态后再执行后续安装步骤
:::

::: details swanlab-self-hosted-pvc.yaml 模板

```yaml
# ----------------------------
# PersistentVolumeClaim for Redis
# ----------------------------
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: swanlab-redis-pvc
  namespace: <your_namespace> # 替换为实际命名空间
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: ""      # 替换为您集群中实际的 storageClass
  volumeMode: Filesystem
---
# ----------------------------
# PersistentVolumeClaim for PostgreSQL
# ----------------------------
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: swanlab-postgres-pvc
  namespace: <your_namespace> # 替换为实际命名空间
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: ""      # 替换为您集群中实际的 storageClass
  volumeMode: Filesystem
---
# ----------------------------
# PersistentVolumeClaim for ClickHouse
# ----------------------------
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: swanlab-clickhouse-pvc
  namespace: <your_namespace> # 替换为实际命名空间
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1000Gi
  storageClassName: ""      # 替换为您集群中实际的 storageClass
  volumeMode: Filesystem
---
# ----------------------------
# PersistentVolumeClaim for Vector-0
# ----------------------------
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-swanlab-self-hosted-vector-0
  namespace: <your_namespace> # 替换为实际命名空间
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 60Gi
  storageClassName: ""      # 替换为您集群中实际的 storageClass
  volumeMode: Filesystem
---
# ----------------------------
# PersistentVolumeClaim for Vector-1
# ----------------------------
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-swanlab-self-hosted-vector-1
  namespace: <your_namespace> # 替换为实际命名空间
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 60Gi
  storageClassName: ""      # 替换为您集群中实际的 storageClass
  volumeMode: Filesystem
```

:::

```bash
# 创建 PVC
kubectl apply -f swanlab-self-hosted-pvc.yaml

# 验证 PVC 状态（务必确保除 vector 外全部 Bound）
kubectl get pvc -n <your_namespace>
```



### 3. value 填写

完整的 `value.yaml` 配置项可参考 [自定义 Value 配置说明](/self_host/kubernetes/configuration.md) 文档。

您可以在 [values.yaml 模板](https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml) 查看 `swanlab-self-hosted` 使用的原始 value 模板，也可以通过下列指令查看：

```bash
helm show values swanlab/self-hosted
```

我们 **强烈建议您保存好部署使用的 `values.yaml`，以便后续正常升级**，以下是相关的 `value.yaml` 修改说明: 

#### 3.1 PVC 绑定

`dependencies` 字段下各组件的 `existingClaim` 需与第 2 步已创建的 PVC 名称对齐：

| 组件 | values.yaml 路径 | 对应 PVC 名称 |
|------|-----------------|--------------|
| PostgreSQL | `dependencies.postgres.persistence.existingClaim` | `swanlab-postgres-pvc` |
| Redis | `dependencies.redis.persistence.existingClaim` | `swanlab-redis-pvc` |
| ClickHouse | `dependencies.clickhouse.persistence.existingClaim` | `swanlab-clickhouse-pvc` |

修改示例: 

::: details dependencies PVC 修改示例

::: code-group

```yaml [PostgreSQL]
dependencies:
  postgres: # PostgreSQL 数据库配置
    fullnameOverride: ""

    image: # 镜像配置
      # 完整镜像仓库路径（例如 ghcr.io/cloudnative-pg/postgresql）
      repository: repo.swanlab.cn/self-hosted/postgres
      # 镜像标签/版本。建议 SwanLab 使用 16.x 及以上版本
      tag: "16.1"
      # Kubernetes 镜像拉取策略（Always、IfNotPresent、Never）
      pullPolicy: "IfNotPresent"

    username: ""
    password: ""

    persistence: # 持久化存储配置
      # 用于动态供应的 Kubernetes StorageClass
      storageClass: "disk-cloud-auto"
      # 数据库存储卷大小
      storageSize: "10Gi"

      # 使用已有的 PVC（留空则自动创建新的 pvc）
      existingClaim: "swanlab-postgres-pvc" # ⚠️： 通常仅修改这一个名称，对齐创建的 postgres PVC 名称

    customLabels: {}
    customAnnotations: {}
    customPodLabels: {}
    customPodAnnotations: {}
    customTolerations: []
    customNodeSelector: {}
    resources: {}
```

```yaml [Redis]
  redis: # Redis 配置
    fullnameOverride: ""

    image: # 镜像配置
      # 完整镜像仓库路径（例如 redis/redis-stack）
      repository: repo.swanlab.cn/self-hosted/redis-stack
      # 镜像标签/版本
      tag: "7.4.0-v8"
      # Kubernetes 镜像拉取策略（Always、IfNotPresent、Never）
      pullPolicy: "IfNotPresent"

    persistence: # 持久化存储配置
      storageClass: "disk-cloud-auto"
      storageSize: "10Gi"

      # 使用已有的 PVC（留空则自动创建新的 pvc）
      existingClaim: "swanlab-redis-pvc" # ⚠️： 通常仅修改这一个名称，对齐创建的 redis PVC 名称

    customLabels: {}
    customAnnotations: {}
    customPodLabels: {}
    customPodAnnotations: {}
    customTolerations: []
    customNodeSelector: {}
    resources: {}
```

```yaml [ClickHouse]
  clickhouse: # ClickHouse 配置
    fullnameOverride: ""

    image: # 镜像配置
      # 完整镜像仓库路径（例如 clickhouse/clickhouse-server）
      repository: repo.swanlab.cn/self-hosted/clickhouse-server
      # 镜像标签/版本
      tag: "24.3"
      pullPolicy: "IfNotPresent"

    # 认证凭据
    # 如果使用 existingSecret 则留空
    username: ""
    password: ""

    persistence: # 持久化存储配置
      storageClass: "disk-cloud-auto"
      storageSize: "20Gi"

      # 使用已有的 PVC（留空则自动创建新的 pvc）
      existingClaim: "swanlab-clickhouse-pvc" # ⚠️： 通常仅修改这一个名称，对齐创建的 clickhouse PVC 名称

    customLabels: {}
    customAnnotations: {}
    customPodLabels: {}
    customPodAnnotations: {}
    customTolerations: []
    customNodeSelector: {}
    resources: {}
```

:::



#### 3.2 应用镜像标签
`service` 下的四个应用镜像（server / house / cloud / next）的 `tag` 应设置为**空字符串**，Chart 会在渲染时自动注入正确的版本号。

#### 3.3 Vector 存储
`vector.persistence.storageClass` 和 `vector.persistence.storageSize`，与创建 vector PVC 时的大小保持一致，默认size 需修改为 `60Gi`。


#### 3.4 外部 S3 集成配置

`integrations.s3` 字段需要根据您使用的对象存储服务手动填写，推荐 public 桶和 private 桶分离。如您的云厂商对 S3 协议的 endpoint 访问做了区分，请特别注意应填写 S3 endpoint。详细的字段说明与配置示例，请参阅 [外部 S3 集成](/self_host/kubernetes/configuration.md#外部对象存储-s3-integrations-s3) 章节。

### 4. 添加 helm 仓库
您可以通过[helm](https://helm.sh/)安装SwanLab私有化服务K8S版。

首先建立本地仓库映射：

```bash
helm repo add swanlab https://helm.swanlab.cn
# 更新仓库
helm repo update
# 列出所有 chart 版本
helm search repo swanlab/self-hosted --versions

```

### 5. 执行安装
您可以根据集群网络环境选择以下两种安装方式之一。

#### 选项一：Helm 仓库安装

如果您的**集群节点可以直接访问 Helm 仓库**（即集群节点可以直接访问 `github.com`），可以参考如下命令执行安装：
> ⚠️ 注意：https://helm.swanlab.cn 的 chart 包在 [GitHub Release](https://github.com/SwanHubX/charts/releases )做版本 tag 索引 ， **请提前确认网络连通性!**

```bash
# 建议先使用 --dry-run 验证模板兼容性
helm install swanlab-self-hosted swanlab/self-hosted \
  -f <your_own_values.yaml> \
  --namespace <your_namespace> \
  --dry-run
```
- 确认无报错后，去掉 `--dry-run` 选项执行安装


#### 选项二：本地 Chart 包安装

如果您的**集群节点无法直接访问 Helm 仓库**（即集群节点无法直接访问 `github.com`），可以通过 OCI 方式拉取 chart 包到本地后执行安装：

> 如遇到 401 认证失败问题，可以通过 `helm registry logout xxx.com` 的形式清除本机此前存在的 helm 登录态。

```bash
# 拉取 chart 包到本地
helm pull oci://swanlab-registry.cn-hangzhou.cr.aliyuncs.com/chart/self-hosted --version <latest_chart_version>
# 解压 chart 包，预期只有一个 self-hosted/ 文件夹
tar -zxvf self-hosted-<latest_chart_version>.tgz
```

然后使用本地 chart 包更新验证：
```bash
# 建议先使用 --dry-run 验证模板兼容性
helm install swanlab-self-hosted ./self-hosted/ \
  -f <your_own_values.yaml> \
  --namespace <your_namespace> \
  --dry-run
```
- 确认无报错后，去掉 `--dry-run` 选项执行安装

通过安装 `swanlab/self-hosted`，即可在k8s上安装SwanLab私有化部署版应用，安装结果会在终端打印类似如下信息：

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
   https://docs.swanlab.cn/self_host/kubernetes/deploy.html
```

如上所示，`swanlab-self-hosted` 私有化服务默认无法直接通过外部网络访问，您可以通过`port-forward`功能在本地访问此服务。
如果您希望**开启外部访问（通过IP或域名）**，请参考 [配置应用访问入口](/self_host/kubernetes/configuration.md#配置应用访问入口)。

下面是一个在本机访问的例子，打开终端并执行：

```bash
kubectl port-forward --namespace self-hosted svc/swanlab-self-hosted 8080:80
```

然后你可以在浏览器中访问：`http://127.0.0.1:8080`，即可看到SwanLab的页面：

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/zh/guide_cloud/self_host/docker-deploy/create-account.png)

## 📖 License 激活
### 个人版激活
现在，你需要激活你的主账号。激活需要1个License，个人使用可以免费在[SwanLab官网](https://swanlab.cn)申请一个，位置在 「设置」-「账户与许可证」。

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/zh/guide_cloud/self_host/docker-deploy/apply-license.png)

拿到License后，回到激活页面，填写用户名、密码、确认密码和License，点击激活即可完成创建。

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/zh/guide_cloud/self_host/docker-deploy/quick-start.png)

### 企业版激活
如您需要测试企业版能力，请联系 [contact@swanlab.cn](mailto:contact@swanlab.cn)，我们会将测试 License 发送到该email中。


## ⚙️ 验证测试
部署完成后，可通过下面 Python 代码用于验证标量与媒体的指标上报

::: details 指标上报测试
```python
import swanlab
import random
import numpy as np  # 添加 NumPy 导入

swanlab.login(
    api_key="xxxxx",
    host="https://xxxxx" 
)

# 创建一个SwanLab项目
swanlab.init(
    # 设置项目名
    project="my-first-project",
    experiment_name="my-first-experiment",
    # 设置超参数
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10
    }
)

# 模拟一次训练
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    swanlab.log({
        "step_time": acc,
        "speed": loss
    })

# 生成随机噪声图像（64x64 RGB，随机像素值）
random_noise = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
img = swanlab.Image(random_noise, caption="Random Noise")
swanlab.log({
    "image": img
})

# [可选] 完成训练，这在notebook环境中是必要的
swanlab.finish()

```
:::

预期效果如下图所示⬇️：

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260603151219153.png"  style="width:60%"/>

可以检查如下功能：
- ⬜ 查看指标是否能够正常上报
- ⬜ 查看图像是否正常上传并显示
- ⬜ 查看 metric 的 csv 下载是否都顺利
- ⬜ 查看用户头像是否能够正常显示

##  🧱 额外说明

您可以在[此处](https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml)查看 `swanlab-self-hosted` 的所有可配置项。

详细的字段说明与配置实践，请参阅 [自定义 Value 配置说明](/self_host/kubernetes/configuration) 文档，其中涵盖：

- **全局配置**：Pod 反亲和性、登录域名等
- **应用服务**：副本数量、资源限制、标签与注解等
- **网关配置**：应用访问入口、端口等
- **内置基础服务**：PostgreSQL / Redis / ClickHouse / MinIO 的存储资源配置
- **外部服务集成**：接入外部 PostgreSQL、Redis、ClickHouse、S3 对象存储

### 更新与回滚

如需更新 SwanLab 版本或在更新失败后进行回滚，请参考[更新与回滚](/self_host/kubernetes/upgrade)文档。

### Prometheus 可观测接入指引

文档待更新，敬请期待。

