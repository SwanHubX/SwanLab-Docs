# 监控与日志配置指南

> 本文档介绍了利用 `Prometheus + Grafana` 监测 SwanLab 线上应用的配置方法。

## ☀️ 架构概述

SwanLab 私有化部署采用微服务架构，各应用服务按照职责拆分并独立运行，整体监控链路如下：

1. **Prometheus** 定期抓取 SwanLab 各个服务暴露的 `/metrics` 接口。
2. **Grafana** 从 Prometheus 读取数据，并渲染 SwanLab 的监控仪表盘和告警面板。
3. **「可选」Alertmanager** 或您已有的告警系统在 Prometheus 告警规则触发时发送通知。

## 🪜 流程示意

## 🧱 前置条件

- 已通过 Helm 安装 SwanLab 私有化服务（参考 [Kubernetes 部署指南](./deploy.md)）
- 应用默认 `release_name` 为 `swanlab-self-hosted`，安装命名空间为 `<your_namespace>`（请根据实际情况替换）
- 具备访问相关 Kubernetes 资源的权限

下表为 SwanLab 后端服务目前支持访问 metrics 信息的应用和对应接口配置、路由：

| 服务名称       | 服务说明         | 端口 | 路由     |
| -------------- | ---------------- | ---- | -------- |
| SwanLab-Server | 后端核心业务服务 | 3000 | /metrics |
| SwanLab-House  | 实验指标OLAP服务 | 3000 | /metrics |

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
)" -- wget -qO- http://127.0.0.1:3000/metrics
```

其中：

- `app.kubernetes.io/instance=<release_name>` 中，`<release_name>` 使用的是默认的 RELEASE 名称，默认为 `swanlab-self-hosted` ，请按照实际部署情况替换
- `<your_namespace>` 替换为您实际部署使用的集群命名空间

## 📊 集成监控服务

待更新

### 配置仪表盘

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

Metrics 接口遵循 Prometheus 格式规范，通常会返回请求 QPS、请求延迟、请求错误率等信息，同时包含 Node.js、Go 等语言内部运行指标。由于指标数量庞大，很难完全列出所有指标及其含义。通常我们建议您通过 [前置条件](#🧱-前置条件) 中的验证 Metrics 接口，或者在 Prometheus 面板手动获取所有指标信息，然后借助其他工具（如大语言模型）查询对应指标的含义。

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
