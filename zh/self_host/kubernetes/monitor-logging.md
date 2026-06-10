# 监控与日志配置指南

> 本文档介绍了利用 `Prometheus + Grafana` 监测 SwanLab 线上应用的配置方法。

## ☀️ 架构概述

SwanLab 私有化部署采用微服务架构，各应用服务按照职责拆分并独立运行，整体监控链路如下：

1. **Prometheus** 定期抓取 SwanLab 各个服务暴露的 `/metrics` 接口。
2. **Grafana** 从 Prometheus 读取数据，并渲染 SwanLab 的监控仪表盘和告警面板。
3. **「可选」Alertmanager** 或您已有的告警系统在 Prometheus 告警规则触发时发送通知。

## 🧱 前置条件

- 已通过 Helm 安装 SwanLab 私有化服务（参考 [Kubernetes 部署指南](./deploy.md)）
- 应用默认 `release_name` 为 `swanlab-self-hosted`，安装命名空间为 `<your_namespace>`（请根据实际情况替换）
- 具备访问相关 Kubernetes 资源的权限


下表为 SwanLab 后端服务目前支持访问 metrics 信息的应用和对应接口配置、路由：

| 服务名称 | 服务说明 | 端口 | 路由 |
|---------|---------|------|------|
| SwanLab-Server | 后端核心业务服务 | 3000 | /metrics |
| SwanLab-House | 实验指标OLAP服务 | 3000 | /api/house/metrics |


在实际配置 Prometheus 抓取任务前，建议先验证各自服务的 Prometheus Metrics 接口是否正常。

- **验证 SwanLab-Server**

```bash
kubectl exec -n <your_namespace> -c server "$(
  kubectl get pod -n <your_namespace> \
    -l app.kubernetes.io/instance=swanlab-self-hosted,app.kubernetes.io/service=server \
    -o jsonpath='{.items[0].metadata.name}'
)" -- wget -qO- http://127.0.0.1:3000/metrics
```

- **验证 SwanLab-House**

```bash
kubectl exec -n <your_namespace> -c house "$(
  kubectl get pod -n <your_namespace> \
    -l app.kubernetes.io/instance=swanlab-self-hosted,app.kubernetes.io/service=house \
    -o jsonpath='{.items[0].metadata.name}'
)" -- wget -qO- http://127.0.0.1:3000/api/house/metrics
```
其中：
- `app.kubernetes.io/instance=<release_name>` 中，`<release_name>` 使用的是默认的 RELEASE 名称，默认为 `swanlab-self-hosted` ，请按照实际部署情况替换
- `<your_namespace>` 替换为您实际部署使用的集群命名空间



## 📊 集成监控服务

根据您的环境选择合适的配置方式：

- **场景一：集群中【没有】 Prometheus** — 在 SwanLab 私有化服务的命名空间内独立部署 Prometheus + Grafana，并配置可观测指标
- **场景二：集群中【已有】 Prometheus** — 将 SwanLab 的可观测指标接入现有的 Prometheus 监控体系


### 1. 场景一：集群中没有 Prometheus 监控

此场景适用于没有现成 Prometheus 监控的集群，需要在 SwanLab 的命名空间内独立部署一套完整的 `Prometheus + Grafana` 监控栈。

如集群中已有成熟的 `Prometheus` 可观测服务，可以跳过本步骤，查看 [配置 Prometheus 抓取任务](#_2-场景二-集群中已有-prometheus)。



#### 1.1 创建 swanlab-monitor-PVC

为 Prometheus 和 Grafana 创建持久化存储，示例中以各自 `20Gi` 的存储空间大小进行申请，可根据集群实际使用情况进行申请

::: details swanlab-monitor-pvc.yaml 配置示例
```yaml
# ============================================================
# Prometheus + Grafana PVC 配置
# ============================================================
# 用于为 Prometheus 和 Grafana 预创建持久化存储
# 必须在安装 kube-prometheus-stack 之前创建并确保 Bound
#
# 使用方式：
#   kubectl apply -f swanlab-monitor-pvc.yaml
#
# 验证状态：
#   kubectl get pvc -n <namespace>
#
# 注意：
#   - Prometheus PVC 名称必须与 Operator（StatefulSet）自动生成的名称完全一致：
#     prometheus-<CR名>-db-prometheus-<CR名>-<序号>
#     其中 CR名 = <release>-kube-prome-prometheus（release 名与 chart 名拼接后截断 26 字符）
#     release 为 swanlab-monitor 时即：
#     prometheus-swanlab-monitor-kube-prome-prometheus-db-prometheus-swanlab-monitor-kube-prome-prometheus-0
#     名称不一致时 Operator 会另行自动创建新 PVC，预创建的 PVC 将被闲置
#     安装后请用 kubectl get pvc -n <namespace> 核对 Prometheus Pod 实际挂载的 PVC
#   - Grafana PVC 名称在 values 中通过 existingClaim 引用
# ============================================================

# Prometheus 数据存储
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  # PVC 名称必须与 Operator 自动生成的名称完全一致（格式说明见文件头注释）
  name: prometheus-swanlab-monitor-kube-prome-prometheus-db-prometheus-swanlab-monitor-kube-prome-prometheus-0
  namespace: <your_namespace> # TODO: 替换为实际的命名空间
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi    # TODO: 按需动态扩容
  storageClassName: <your_storageClassName> # TODO: 替换为实际的存储类名称
  volumeMode: Filesystem
---
# Grafana 数据存储
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  # Grafana PVC 名称在 values 中通过 existingClaim 引用
  name: swanlab-monitor-grafana-pvc
  namespace: <your_namespace>  # TODO: 替换为实际的命名空间
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi # TODO: 按需动态扩容
  storageClassName: <your_storageClassName> # TODO: 替换为实际的存储类名称
  volumeMode: Filesystem
```
:::

使用如下指令申请监控所需的 PVC 存储资源
```bash
kubectl apply -f swanlab-monitor-pvc.yaml

# 验证 PVC 状态（务必确保全部 Bound）
kubectl get pvc -n <your_namespace>
```

#### 1.2 配置 swanlab-monitor-value

编写独立的 `swanlab-monitor-value.yaml`  文件，包含 Prometheus scrape job 配置、PVC 引用：

::: details swanlab-monitor-value.yaml 配置示例
```yaml
# ============================================================
# kube-prometheus-stack values — 使用阿里云 ACR 镜像
# ============================================================
# 适用于在 SwanLab 命名空间内独立部署 Prometheus + Grafana
# 通过 Pod Annotation 自动发现 SwanLab 服务
#
# 使用方式：
#   helm install swanlab-monitor prometheus-community/kube-prometheus-stack \
#     -n <namespace> -f swanlab-monitor-value.yaml
#
# 前置条件：
#   1. 已创建 PVC（见 swanlab-monitor-pvc.yaml）
#   2. 已添加 Helm 仓库：
#      helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
# ============================================================

# ---------- Grafana ----------
# Grafana 用于可视化 Prometheus 数据，提供 SwanLab 监控仪表盘
grafana:
  persistence:
    enabled: true
    # 使用预创建的 PVC（必须在安装前创建并 Bound）
    existingClaim: swanlab-monitor-grafana-pvc
  # Grafana 管理员默认密码（默认用户名为 admin）
  # ⚠️ 安全提示：此为示例默认密码，安装完成后请立即登录 Grafana 修改，
  # 或改用 admin.existingSecret 引用 Kubernetes Secret 管理凭据
  adminPassword: "swanlab-monitor@default"

  # Grafana 主镜像（阿里云 ACR）
  image:
    registry: repo.swanlab.cn
    repository: public/grafana
    tag: "13.0.1-security-01"

  # initChownData 容器镜像（用于初始化数据目录权限）
  initChownData:
    image:
      registry: repo.swanlab.cn
      repository: public/busybox
      tag: "1.38.0"

  # sidecar 容器镜像（用于自动加载 ConfigMap 中的 dashboard 和 datasource）
  sidecar:
    image:
      registry: repo.swanlab.cn
      repository: public/k8s-sidecar
      tag: "2.7.3"

  replicas: 1

# ---------- Prometheus ----------
# Prometheus 用于采集和存储 SwanLab 的 metrics 数据
prometheus:
  prometheusSpec:
    # 允许选择所有 ServiceMonitor（不限于 Helm 管理的）
    serviceMonitorSelectorNilUsesHelmValues: false

    # 持久化存储配置
    # Operator 按 volumeClaimTemplate 为 StatefulSet 生成 PVC，名称格式为：
    #   prometheus-<CR名>-db-prometheus-<CR名>-<序号>（CR名 = <release>-kube-prome-prometheus）
    # 本 release（swanlab-monitor）对应：
    #   prometheus-swanlab-monitor-kube-prome-prometheus-db-prometheus-swanlab-monitor-kube-prome-prometheus-0
    # 该 PVC 已在 swanlab-monitor-pvc.yaml 中按此名称预创建，安装时将直接复用
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: <your_storageClassName> # TODO: 替换为实际的存储类名称，必须与 swanlab-monitor-pvc.yaml 中的 storageClassName 保持一致
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 20Gi # 必须与 swanlab-monitor-pvc.yaml 中预创建的容量保持一致

    replicas: 1

    # Prometheus 主镜像（阿里云 ACR）
    image:
      registry: repo.swanlab.cn
      repository: public/prometheus
      tag: "v3.12.0-distroless"

    # 自定义抓取配置 —— SwanLab 专属 scrape job
    # 通过 Pod Annotation 自动发现 SwanLab 服务
    additionalScrapeConfigs:
      - job_name: "swanlab"
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          # 只抓取标记为 prometheus.io/scrape: "swanlab" 的 Pod
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: swanlab
          # 从注解读取 metrics_path
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          # 从注解读取端口
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            target_label: __address__
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
          # 保留常用 Kubernetes 标签，方便 Grafana 查询和分组
          - source_labels: [__meta_kubernetes_namespace]
            target_label: namespace
          - source_labels: [__meta_kubernetes_pod_name]
            target_label: pod
          - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_instance]
            target_label: release
          - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_service]
            target_label: service

# ---------- Alertmanager ----------
# Alertmanager 用于处理 Prometheus 告警规则并发送通知
alertmanager:
  replicas: 1
  alertmanagerSpec:
    image:
      registry: repo.swanlab.cn
      repository: public/alertmanager
      tag: "v0.32.2"

# ---------- Prometheus Operator ----------
# Prometheus Operator 用于管理 Prometheus 和 Alertmanager 实例
prometheusOperator:
  replicas: 1
  image:
    registry: repo.swanlab.cn
    repository: public/prometheus-operator
    tag: "v0.91.0"
  # admissionWebhook 用于验证 PrometheusRule 和 ServiceMonitor 配置
  admissionWebhook:
    image:
      registry: repo.swanlab.cn
      repository: public/kube-webhook-certgen
      tag: "1.8.3"
    patch:
      image:
        registry: repo.swanlab.cn
        repository: public/kube-webhook-certgen
        tag: "1.8.3"
  # prometheusConfigReloader 用于在 ConfigMap 变更时自动重载 Prometheus 配置
  prometheusConfigReloader:
    image:
      registry: repo.swanlab.cn
      repository: public/prometheus-config-reloader
      tag: "v0.91.0"

# ---------- Kube State Metrics ----------
# kube-state-metrics 用于从 Kubernetes API 导出集群资源指标
kube-state-metrics:
  replicas: 1
  image:
    registry: repo.swanlab.cn
    repository: public/kube-state-metrics
    tag: "v2.19.0"

# ---------- Node Exporter (DaemonSet) ----------
# node-exporter 用于采集节点级别的硬件和操作系统指标
# 注意：chart 默认 distroless: true，会自动追加 -distroless 后缀
#       因此 tag 只需写 "v1.11.1"，不要带 -distroless
prometheus-node-exporter:
  # 禁用 hostNetwork 避免端口冲突（默认为 true）
  hostNetwork: false
  hostPort:
    enabled: false
  # 容忍所有 taint，确保 DaemonSet 在所有节点上运行（包括控制面）
  tolerations:
    - effect: NoSchedule
      operator: Exists
    - effect: NoExecute
      operator: Exists
    - effect: PreferNoSchedule
      operator: Exists
  image:
    registry: repo.swanlab.cn
    repository: public/node-exporter
    tag: "v1.11.1"

```
:::

:::warning ⚠️ 安全提示
配置示例中的 Grafana 管理员密码 `swanlab-monitor@default` 仅为默认示例值。安装完成后，请**立即登录 Grafana 修改管理员密码**；生产环境建议改用 `grafana.admin.existingSecret` 引用 Kubernetes Secret 管理凭据，避免密码明文出现在 values 文件中。
:::

#### 1.3 安装 Prometheus + Grafana

类似 [升级与回滚](./upgrade.md) 章节，区分集群能否访问 `github.com`，可以选择不同的安装方式


- **集群可以访问 github**

首先添加 prometheus-community 的仓库的 `kube-prometheus-stack` chart:

```bash
# 添加 Helm 仓库
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# 使用 helm repo 在线安装
helm install swanlab-monitor prometheus-community/kube-prometheus-stack \
  -n <your_namespace> \
  -f swanlab-monitor-value.yaml
```

- **集群无法访问 github**

通过 `oci` 的方式将 `kube-prometheus-stack` 的 chart 包拉取到本地，再执行安装
```bash
# 1. 拉取 chart 包到本地
helm pull oci://swanlab-registry.cn-hangzhou.cr.aliyuncs.com/chart/monitoring/kube-prometheus-stack \
  --version 86.2.1
# 2. 解压
tar -zxvf kube-prometheus-stack-86.2.1.tgz
# 3. 使用本地 chart 安装
helm install swanlab-monitor ./kube-prometheus-stack \
  -n <your_namespace> \
  -f swanlab-monitor-value.yaml
```

等待所有 `pods` 和 `deployments` 正常

```bash
# deployments
kubectl get deployments -n <your_namespace> | grep monitor

# pods
kubectl get pods -n <your_namespace> | grep monitor
```



### 2. 场景二：集群中已有 Prometheus

此场景适用于已有成熟 Prometheus 监控的集群，只需将 SwanLab 接入现有的 Prometheus 抓取任务。本节提供三种接入方式，选择其一即可：

| 方式 | 说明 | 适用条件 | 需要修改 Pod 注解 |
|------|------|----------|------------------|
| [2.1 方式一：Pod Annotation（推荐）](#_2-1-方式一-pod-annotation-推荐) | 单个 job，通过注解动态发现服务 | 需要 [第 3 步](#_3-配置-pod-注解并更新服务) 配置注解 | ✅ 是 |
| [2.2 方式二：Label-Based 双 Job](#_2-2-方式二-label-based-双-job) | 两个独立 job，通过 Pod 标签发现 | SwanLab Helm Chart 默认生成的标签即可 | ❌ 否 |
| [2.3 方式三：Prometheus Operator](#_2-3-方式三-prometheus-operator-servicemonitor-podmonitor) | 通过 ServiceMonitor/PodMonitor 声明式接入 | 集群已部署 Prometheus Operator | ❌ 否 |

#### 2.1 方式一：Pod Annotation（推荐）

在现有的 `prometheus.yaml` 的 `scrape_configs` 中添加一个 SwanLab 专属 job，通过 Pod 注解动态发现服务。此方式使用单个 job 统一采集 Server 和 House，后续新增服务也只需添加注解、无需修改 Prometheus 配置。

::: details prometheus.yaml - Pod Annotation scrape job 配置示例
```yaml
scrape_configs:
  - job_name: "swanlab"
    kubernetes_sd_configs:
      - role: pod

    relabel_configs:
      # 只抓取标记为 prometheus.io/scrape: "swanlab" 的 Pod
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: swanlab

      # 将 Pod Annotation 中的 prometheus.io/path 作为 metrics_path
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)

      # 将 Pod Annotation 中的 prometheus.io/port 作为抓取端口
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        target_label: __address__
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2

      # 保留常用 Kubernetes 标签，方便 Grafana 查询和分组
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace

      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod

      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_instance]
        target_label: release

      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_service]
        target_label: service
```
:::

采用方式一时，需要继续完成 [第 3 步：配置 Pod 注解](#_3-配置-pod-注解并更新服务) 才能生效。

#### 2.2 方式二：Label-Based 双 Job

在 `prometheus.yaml` 的 `scrape_configs` 中添加两个独立的 job，分别采集 Server 和 House。此方式基于 SwanLab Helm Chart 自动为 Pod 生成的 `app.kubernetes.io/instance` 和 `app.kubernetes.io/service` 标签进行发现，**无需修改 Pod 注解**，部署后即可接入。

::: details prometheus.yaml - Label-Based 双 Job 配置示例
```yaml
scrape_configs:
  - job_name: "swanlab-server"
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - <your_namespace>    # TODO: 替换为 SwanLab 部署的命名空间
    relabel_configs:
      # 只抓取 release=<release_name> 且 service=server 的 Pod
      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_instance]
        action: keep
        regex: swanlab-self-hosted    # TODO: 替换为实际的 release name

      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_service]
        action: keep
        regex: server

      # 强制使用 Metrics 端口
      - source_labels: [__address__]
        action: replace
        target_label: __address__
        regex: ([^:]+)(?::\d+)?
        replacement: $1:3000

      - target_label: __metrics_path__
        replacement: /metrics

      # 保留常用 Kubernetes 标签
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace

      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod

      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_service]
        target_label: service

  - job_name: "swanlab-house"
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - <your_namespace>    # TODO: 替换为 SwanLab 部署的命名空间
    relabel_configs:
      # 只抓取 release=<release_name> 且 service=house 的 Pod
      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_instance]
        action: keep
        regex: swanlab-self-hosted    # TODO: 替换为实际的 release name

      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_service]
        action: keep
        regex: house

      # 强制使用 Metrics 端口
      - source_labels: [__address__]
        action: replace
        target_label: __address__
        regex: ([^:]+)(?::\d+)?
        replacement: $1:3000

      # House 的 Metrics 路径与 Server 不同
      - target_label: __metrics_path__
        replacement: /api/house/metrics

      # 保留常用 Kubernetes 标签
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace

      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod

      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_service]
        target_label: service
```
:::

::: tip 与方式一的区别
- **无需修改 Pod 注解**：利用 SwanLab Helm Chart 默认生成的 `app.kubernetes.io/*` 标签发现，部署后即可接入。
- **限定命名空间**：通过 `namespaces.names` 缩小 SD 范围，对权限收紧的集群更友好。
- **独立 Job**：Server 和 House 各自一个 job，便于在 Prometheus 的 Targets 面板中独立监控和排查。
- 导入 Grafana 仪表盘时，模板变量 `$job` 的值为 `swanlab-server` 或 `swanlab-house`（而非 `swanlab`）。
:::

#### 2.3 方式三：Prometheus Operator（ServiceMonitor / PodMonitor）

如果集群中的 Prometheus 由 [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator) 管理（例如 kube-prometheus-stack、Bitnami 等 Helm Chart），可以通过声明式的 `ServiceMonitor` 或 `PodMonitor` 资源接入 SwanLab，无需手动编辑 `prometheus.yaml`。

::: details ServiceMonitor 配置示例
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: swanlab-server
  namespace: <your_namespace>    # TODO: 替换为 SwanLab 部署的命名空间
  labels:
    release: kube-prometheus-stack    # TODO: 匹配 Prometheus Operator 的 serviceMonitorSelector
spec:
  namespaceSelector:
    matchNames:
      - <your_namespace>
  selector:
    matchLabels:
      app.kubernetes.io/instance: swanlab-self-hosted    # TODO: 替换为实际的 release name
      app.kubernetes.io/service: server
  endpoints:
    - port: http
      path: /metrics
      interval: 30s
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: swanlab-house
  namespace: <your_namespace>    # TODO: 替换为 SwanLab 部署的命名空间
  labels:
    release: kube-prometheus-stack    # TODO: 匹配 Prometheus Operator 的 serviceMonitorSelector
spec:
  namespaceSelector:
    matchNames:
      - <your_namespace>
  selector:
    matchLabels:
      app.kubernetes.io/instance: swanlab-self-hosted    # TODO: 替换为实际的 release name
      app.kubernetes.io/service: house
  endpoints:
    - port: http
      path: /api/house/metrics
      interval: 30s
```
:::

::: warning
- `metadata.labels.release` 的值必须与 Prometheus Operator 的 `serviceMonitorSelector` 匹配，否则 Operator 不会拾取此 ServiceMonitor。请检查现有 Prometheus 的 Helm values 中 `prometheus.prometheusSpec.serviceMonitorSelector` 的设置。
- 场景一的 `swanlab-monitor-value.yaml` 已通过 `serviceMonitorSelectorNilUsesHelmValues: false` 设置允许选择所有 ServiceMonitor，因此如果在场景一中新增 ServiceMonitor 无需额外配置。
:::

### 3. 配置 Pod 注解并更新服务

> 此步骤仅适用于 [方式一（Pod Annotation）](#_2-1-方式一-pod-annotation-推荐)。若采用 [方式二（Label-Based）](#_2-2-方式二-label-based-双-job) 或 [方式三（ServiceMonitor）](#_2-3-方式三-prometheus-operator-servicemonitor-podmonitor)，可跳过此步，直接前往 [第 4 步](#_4-配置-ingress)。

在 SwanLab 私有化服务的 `swanlab-self-hosted-value.yaml` 中，分别找到`service.server.customPodAnnotations` 和`service.house.customPodAnnotations`，为 `SwanLab-Server` 和 `SwanLab-House` 添加相应的 Prometheus抓取注解: 

::: details swanlab-self-hosted-value.yaml - pod 注解配置示例
```yaml
....
service:
  server:
    ....
    customPodAnnotations:
      prometheus.io/scrape: "swanlab"    # SwanLab 专属抓取标识
      prometheus.io/port: "3000"         # Server Metrics 端口
      prometheus.io/path: "/metrics"     # Server Metrics 路径
...
  house:
    ... 
    customPodAnnotations:
      prometheus.io/scrape: "swanlab"    # SwanLab 专属抓取标识
      prometheus.io/port: "3000"         # House Metrics 端口
      prometheus.io/path: "/api/house/metrics"  # House Metrics 路径
    ...
```
:::

:::warning
⚠️ **注意**: `prometheus.io/port` 和 `prometheus.io/path` 为 SwanLab 服务内置要求，通常无法更改。
:::

仅修改上述两项 service 的 pod 注解，参考 [更新与回滚](./upgrade.md) 章节，执行 value 更新，使 SwanLab 私有化服务的 pod 注解生效
```bash
# 在线更新
helm upgrade swanlab-self-hosted swanlab/self-hosted \
  -f swanlab-self-hosted-value.yaml \
  -n <your_namespace>

# 或使用离线 chart 包更新，
helm upgrade swanlab-self-hosted ./self-hosted \
  -f swanlab-self-hosted-value.yaml \
  -n <your_namespace>
```


### 4. 配置 Ingress
同理，`swanlab-monitor` 服务不包括 ingress 的网关配置，您需要在集群的负载均衡器（或 Ingress）上配置外部访问入口，需要对 `swanlab-monitor-grafana` 该 SVC 对应 pod 的 **80 端口**配置访问入口。

如果您可以配置端口转发，可以通过如下指令配置端口转发，打开 `localhost:3000` 进行配置
```bash
kubectl port-forward svc/swanlab-monitor-grafana 3000:80 -n <your_namespace>
```

### 5. 导入 Grafana 仪表盘

SwanLab 官方提供了 Grafana 仪表盘模板，支持两种场景下的监控可视化。

#### 5.1 下载配置模板

- [swanlab-monitor-config-server.json](https://baidu.com) — Server 服务监控仪表盘
- [swanlab-monitor-config-house.json](https://baidu.com) — House 服务监控仪表盘




#### 5.2 导入步骤

1. 在 Grafana 中，添加对应的 `prometheus` 数据源，可以点击左侧的 `Connections` -> `Data Sources` -> 右上角 `Add new data source`，选择添加 prometheus 配置，设置 URL 为您的 prometheus 端点

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260609202045722.png"/>

以图中为例:
- `swanlab-monitor-kube-prome-prometheus` 为该 namespace 下的 prometheus 对应的 SVC，暴露端口为 `9090`
- `tenant-shaobo` 为安装该 `swanlab-self-hosted` 私有化服务的命名空间

那么可以配置 `Prometheus server URL` 为 `http://swanlab-monitor-kube-prome-prometheus.tenant-shaobo:9090/`

> 上述示例适用于按照 [场景一](#_1-场景一-集群中没有-prometheus-监控) 在与 `swanlab-self-hosted` 相同命名空间内安装 Prometheus 的情况。若您的 Prometheus 部署在其他命名空间或集群外，请将 URL 替换为实际可访问的 Prometheus 地址。


2. 在 Grafana 中，导航至 **Dashboards → New → Import**

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260609200709949.png"/>

3. 分别粘贴或上传 `swanlab-monitor-config-server.json` 和 `swanlab-monitor-config-house.json`

4. 选择对应的 **dataSource**, **namespace**, **job** 和 **service**。

仪表盘使用模板变量自动适配不同的 Prometheus 配置。导入后在 **顶部下拉菜单** 中选择：

| 变量 | 说明 | 示例值 |
|------|------|--------|
| `$datasource` | Prometheus 数据源 | 选择已配置的数据源 |
| `$namespace` | Kubernetes 命名空间 | `swanlab` |
| `$job` | Prometheus scrape job 名称 | `swanlab`、`swanlab-server` 或 `swanlab-house`（取决于接入方式） |
| `$service` | SwanLab 服务名称 | `server` 或 `house` |

> **说明**：模板变量会自动从 Prometheus 读取可用值，无需手动输入。
>
> **注意**：House 仪表盘中的「CPU 使用情况」「内存使用量」面板数据来自 kubelet/cAdvisor 指标（`container_cpu_usage_seconds_total`、`container_memory_working_set_bytes`），并非来自 SwanLab 服务的 Metrics 接口。场景一安装的 kube-prometheus-stack 默认会采集这些指标；场景二请确认现有 Prometheus 已抓取 cAdvisor 指标，否则这两个面板将无数据。

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260609201624323.png"/>


配置正常后可以看到相关的服务检测指标
- **SwanLab-Server**:
<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260609201132687.png"/>

- **SwanLab-House**:
<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260609201039152.png"/>



## 📝 日志采集

> 🚧 日志采集（如 `Loki + Promtail`、`ELK` 等方案）的配置指南正在编写中，敬请期待。
> 在此之前，您可以通过 `kubectl logs` 查看各服务 Pod 的运行日志，或通过公有云自带的集群 Pod 日志服务进行观测：
>
> ```bash
> kubectl logs -n <your_namespace> <pod_name> -c <container_name>
> ```

## ❓ 常见问题

### 为什么 Metrics 接口返回 404？

最有可能的原因是请求 Method 不对。请确保使用 `HTTP GET` 访问 metrics 接口。除此之外，请确保访问的服务、端口、路由都是正确的。

### Metrics 接口返回的指标分别代表什么？

Metrics 接口遵循 Prometheus 格式规范，通常会返回请求 QPS、请求延迟、请求错误率等信息，同时包含 Node.js、Go 等语言内部运行指标。由于指标数量庞大，很难完全列出所有指标及其含义。通常我们建议您通过 [前置条件](#前置条件) 中的验证 Metrics 接口，或者在 Prometheus 面板手动获取所有指标信息，然后借助其他工具（如大语言模型）查询对应指标的含义。

### Metrics 接口是否返回了 CPU、内存等指标？

Metrics 接口没有采集 CPU、内存等硬件指标。

首先，出于性能考虑，SwanLab 应用服务的 Metrics 接口主要暴露应用运行状态指标，不包含 CPU、内存等系统资源指标，采集 CPU 等资源信息可能会加重应用负担。另一方面，CPU、内存指标采集可能要求更高权限，这不符合 SwanLab 的私有化部署要求。最后，在云原生环境中，这类资源指标通常由 [cAdvisor](https://github.com/google/cadvisor)、[node-exporter](https://github.com/prometheus/node_exporter) 或云厂商监控组件统一采集，您可考虑部署对应组件以采集 CPU 等数据。

### 为什么 SwanLab 监控仪表盘中的面板无数据？

如果是 CPU、内存等面板无数据，正如上一问所述，您需要考虑部署对应的硬件监控组件。如果您确认已部署对应的组件，或者是请求延迟等面板无数据，建议的排查步骤为：

1. 在 Prometheus 面板上查询对应名称的指标是否存在；
2. 如果存在，则说明在 Grafana 面板上的指标查询配置存在错误，需要修改 Grafana 面板配置；
3. 如果不存在，说明 Prometheus 的抓取任务存在问题，需要排查对应任务。

### 是否支持监控 PostgreSQL、ClickHouse 等基础服务？

PostgreSQL、ClickHouse 有推出对应的 exporter（例如 [postgres_exporter](https://github.com/prometheus-community/postgres_exporter)），但是对部署权限要求较高。
未来更新中会考虑为 Grafana 面板集成相应的基础服务指标。
