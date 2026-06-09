# 监控与日志配置指南

> 本文档介绍了利用 `Prometheus + Grafana` 监测 SwanLab 线上应用的配置方法。

## 架构概述

SwanLab 私有化部署采用微服务架构，各应用服务按照职责拆分并独立运行。整体监控链路如下：

1. **Prometheus** 定期抓取 SwanLab 各个服务暴露的 `/metrics` 接口。
2. **Grafana** 从 Prometheus 读取数据，并渲染 SwanLab 的监控仪表盘和告警面板。
3. **（可选）Alertmanager** 或您已有的告警系统在 Prometheus 告警规则触发时发送通知。

## 前置条件

- 已通过 Helm 安装 SwanLab 私有化服务（参考 [Kubernetes 部署指南](./deploy.md)）
- 应用名称为 `swanlab-self-hosted`，安装命名空间为 `swanlab`（请根据实际情况替换）
- 具备访问相关 Kubernetes 资源的权限

## 接口配置信息概览

下表展示了 SwanLab 目前支持访问 metrics 信息的应用和对应接口配置、路由：

| 服务名称 | 服务说明 | 端口 | 路由 |
|---------|---------|------|------|
| SwanLab-Server | 核心业务服务 | 3000 | /metrics |
| SwanLab-House | 实验数据分析服务 | 3000 | /api/house/metrics |

## 验证 Metrics 接口

在配置 Prometheus 抓取任务前，建议先验证 Metrics 接口是否正常。

### 验证 SwanLab-Server

```bash
# 获取 Server Pod
kubectl get pods -n swanlab \
  -l app.kubernetes.io/instance=swanlab-self-hosted,app.kubernetes.io/service=server

# 访问 Metrics 接口
kubectl exec -n swanlab -c server "$(
  kubectl get pod -n swanlab \
    -l app.kubernetes.io/instance=swanlab-self-hosted,app.kubernetes.io/service=server \
    -o jsonpath='{.items[0].metadata.name}'
)" -- wget -qO- http://127.0.0.1:3000/metrics
```

### 验证 SwanLab-House

```bash
# 获取 House Pod
kubectl get pods -n swanlab \
  -l app.kubernetes.io/instance=swanlab-self-hosted,app.kubernetes.io/service=house

# 访问 Metrics 接口
kubectl exec -n swanlab -c house "$(
  kubectl get pod -n swanlab \
    -l app.kubernetes.io/instance=swanlab-self-hosted,app.kubernetes.io/service=house \
    -o jsonpath='{.items[0].metadata.name}'
)" -- wget -qO- http://127.0.0.1:3000/api/house/metrics
```

## 集成监控服务

根据您的环境选择合适的配置方式：

- **场景一：集群中没有 Prometheus** — 在 SwanLab 命名空间内独立部署 Prometheus + Grafana
- **场景二：集群中已有 Prometheus** — 将 SwanLab 接入现有的 Prometheus 监控体系

---

## 场景一：集群中没有 Prometheus

此场景适用于没有现成 Prometheus 监控的集群，需要在 SwanLab 命名空间内独立部署一套完整的监控栈。

### 第一步：配置 Pod 注解

在 SwanLab 的 Helm values 中为 Server 和 House 添加 Prometheus 抓取注解：

```yaml
# swanlab-values.yaml
service:
  server:
    customPodAnnotations:
      prometheus.io/scrape: "swanlab"    # SwanLab 专属抓取标识
      prometheus.io/port: "3000"         # Metrics 端口
      prometheus.io/path: "/metrics"     # Metrics 路径

  house:
    customPodAnnotations:
      prometheus.io/scrape: "swanlab"    # SwanLab 专属抓取标识
      prometheus.io/port: "3000"         # Metrics 端口
      prometheus.io/path: "/api/house/metrics"  # Metrics 路径
```

> **注意**：`prometheus.io/port` 和 `prometheus.io/path` 为 SwanLab 服务内置要求，通常无法更改。

### 第二步：创建 PVC

为 Prometheus 和 Grafana 创建持久化存储：

```yaml
# swanlab-monitor-pvc.yaml
---
# Prometheus 数据存储
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus-swanlab-monitor-kube-prome-prometheus-0
  namespace: swanlab  # 替换为实际命名空间
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: "your-storage-class"  # 替换为实际 StorageClass
  volumeMode: Filesystem
---
# Grafana 数据存储
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: swanlab-monitor-grafana-pvc
  namespace: swanlab  # 替换为实际命名空间
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: "your-storage-class"  # 替换为实际 StorageClass
  volumeMode: Filesystem
```

```bash
kubectl apply -f swanlab-monitor-pvc.yaml

# 验证 PVC 状态（务必确保全部 Bound）
kubectl get pvc -n swanlab
```

### 第三步：配置 values-monitor.yaml

编写独立的 values 文件，包含 Prometheus scrape job 配置、PVC 引用和 Grafana 密码：

```yaml
# swanlab-monitor-value.yaml
# kube-prometheus-stack values

# ---------- Grafana ----------
grafana:
  persistence:
    enabled: true
    existingClaim: swanlab-monitor-grafana-pvc
  adminPassword: "your-password"  # 替换为实际密码

# ---------- Prometheus ----------
prometheus:
  prometheusSpec:
    serviceMonitorSelectorNilUsesHelmValues: false
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: your-storage-class  # 替换为实际 StorageClass
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 20Gi

    # 自定义抓取配置 —— SwanLab 专属 scrape job
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
```

### 第四步：安装 Prometheus + Grafana

```bash
# 添加 Helm 仓库
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# 安装
helm install swanlab-monitor prometheus-community/kube-prometheus-stack \
  -n swanlab \
  -f swanlab-monitor-value.yaml

# 升级 SwanLab 使注解生效
helm upgrade swanlab-self-hosted swanlab/self-hosted \
  -f swanlab-values.yaml \
  -n swanlab
```

### 第五步：验证抓取

```bash
# 访问 Prometheus UI
kubectl port-forward -n swanlab \
  svc/swanlab-monitor-kube-prome-prometheus 9090:9090
# 浏览器打开 http://localhost:9090/targets
# 查看 swanlab job 状态是否为 UP
```


## 场景二：集群中已有 Prometheus

此场景适用于已有成熟 Prometheus 监控的集群，只需将 SwanLab 接入现有的 Prometheus 抓取任务。

### 第一步：配置 Pod 注解

在 SwanLab 的 Helm values 中添加 Prometheus 抓取注解：

```yaml
# swanlab-values.yaml
service:
  server:
    customPodAnnotations:
      prometheus.io/scrape: "swanlab"    # SwanLab 专属抓取标识
      prometheus.io/port: "3000"         # Metrics 端口
      prometheus.io/path: "/metrics"     # Metrics 路径

  house:
    customPodAnnotations:
      prometheus.io/scrape: "swanlab"    # SwanLab 专属抓取标识
      prometheus.io/port: "3000"         # Metrics 端口
      prometheus.io/path: "/api/house/metrics"  # Metrics 路径
```

### 第二步：升级 SwanLab 使注解生效

```bash
helm upgrade swanlab-self-hosted swanlab/self-hosted \
  -f swanlab-values.yaml \
  -n swanlab
```

### 第三步：配置 Prometheus 抓取任务

在现有的 `prometheus.yml` 中添加 SwanLab 专属 scrape job：

```yaml
# prometheus.yml
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

> **注意**：请特别确认 `namespace`、`__meta_kubernetes_pod_label_app_kubernetes_io_instance` 等参数的设置是否正确，这直接影响 Prometheus 是否能准确抓取对应服务。



## 导入 Grafana 仪表盘

SwanLab 官方提供了 Grafana 仪表盘模板，支持两种场景下的监控可视化。

### 下载模板

- [SwanLab-Server.json](./SwanLab-Server.json) — Server 服务监控仪表盘
- [SwanLab-House.json](./SwanLab-House.json) — House 服务监控仪表盘

### 导入步骤

1. 在 Grafana 中，导航至 **Dashboards → New → Import**
2. 分别粘贴或上传 `SwanLab-Server.json` 和 `SwanLab-House.json`
3. 选择对应的 **Prometheus 数据源**

### 配置模板变量

仪表盘使用模板变量自动适配不同的 Prometheus 配置。导入后在顶部下拉菜单中选择：

| 变量 | 说明 | 示例值 |
|------|------|--------|
| `$datasource` | Prometheus 数据源 | 选择已配置的数据源 |
| `$namespace` | Kubernetes 命名空间 | `swanlab` |
| `$job` | Prometheus scrape job 名称 | `swanlab` |
| `$service` | SwanLab 服务名称 | `server` 或 `house` |

> **说明**：模板变量会自动从 Prometheus 读取可用值，无需手动输入。

---

## 常见问题

### 为什么 Metrics 接口返回 404？

最有可能的原因是请求 Method 不对。请确保使用 `HTTP GET` 访问 metrics 接口。除此之外，请确保访问的服务、端口、路由都是正确的。

### Metrics 接口返回的指标分别代表什么？

Metrics 接口遵循 Prometheus 格式规范，通常会返回请求 QPS、请求延迟、请求错误率等信息，同时包含 Node.js、Go 等语言内部运行指标。由于指标数量庞大，很难完全列出所有指标及其含义。通常我们建议您通过 [访问 Metrics](#验证-metrics-接口) 或者在 Prometheus 面板手动获取所有指标信息，然后借助其他工具（如大语言模型）查询对应指标的含义。

### Metrics 接口是否返回了 CPU、内存等指标？

Metrics 接口没有采集 CPU、内存等硬件指标。

首先，出于性能考虑，SwanLab 应用服务的 Metrics 接口主要暴露应用运行状态指标，不包含 CPU、内存等系统资源指标，采集 CPU 等资源信息可能会加重应用负担。另一方面，CPU、内存指标采集可能要求更高权限，这不符合 SwanLab 的私有化部署要求。最后，在云原生环境中，这类资源指标通常由 [cAdvisor](https://github.com/google/cadvisor)、[node-exporter](https://github.com/prometheus/node_exporter) 或云厂商监控组件统一采集，您可考虑部署对应组件以采集 CPU 等数据。

### 为什么 SwanLab 监控仪表盘中的面板无数据？

如果是 CPU、内存等面板无数据，正如上一问所述，您需要考虑部署对应的硬件监控组件。如果您确认已部署对应的组件，或者是请求延迟等面板无数据，建议的排查步骤为：

1. 在 Prometheus 面板上查询对应名称的指标是否存在；
2. 如果存在，则说明在 Grafana 面板上的指标查询配置存在错误，需要修改 Grafana 面板配置；
3. 如果不存在，说明 Prometheus 的抓取任务存在问题，需要排查对应任务。

### 是否支持监控 PostgreSQL、ClickHouse 等基础服务？

PostgreSQL、ClickHouse 有推出对应的 exporter（例如 [postgres_exporter](https://github.com/prometheus-community/postgres_exporter)），但是都需要较高的部署权限。SwanLab 会在未来的更新中为 Grafana 面板集成相应的基础服务指标。
