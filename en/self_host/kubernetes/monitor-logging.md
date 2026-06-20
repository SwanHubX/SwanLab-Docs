# Monitor & Logging Configuration Guide

> This guide describes how to configure `Prometheus + Grafana` monitoring for SwanLab backend services.

## Architecture Overview

SwanLab self-hosted deployment uses a microservices architecture. The monitoring pipeline works as follows:

1. **Prometheus** periodically scrapes the `/metrics` endpoints exposed by each SwanLab service.
2. **Grafana** reads data from Prometheus and renders SwanLab monitoring dashboards and alert panels.
3. **[Optional] Alertmanager** or your existing alerting system sends notifications when Prometheus alert rules trigger.

## Flow Diagram

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/monitor-logging-flow.drawio.svg"/>

## Prerequisites

- SwanLab self-hosted service has been installed via Helm (see [Kubernetes Deployment Guide](./deploy.md))
- The default `release_name` is `swanlab-self-hosted`, installed in namespace `<your_namespace>` (replace with your actual namespace)
- You have the necessary permissions to access Kubernetes resources

The following SwanLab backend services currently expose Prometheus metrics:

| Service        | Description                     | Port | Path               |
| -------------- | ------------------------------- | ---- | ------------------ |
| SwanLab-Server | Core backend service            | 3000 | /metrics           |
| SwanLab-House  | Experiment metrics OLAP service | 3000 | /api/house/metrics |

Before configuring Prometheus scrape jobs, verify that each service's Prometheus Metrics endpoint is working.

- **Verify SwanLab-Server**

```bash
kubectl exec -n <your_namespace> -c server "$(
  kubectl get pod -n <your_namespace> \
    -l app.kubernetes.io/instance=swanlab-self-hosted,app.kubernetes.io/service=server \
    -o jsonpath='{.items[0].metadata.name}'
)" -- wget -qO- http://127.0.0.1:3000/metrics
```

- **Verify SwanLab-House**

```bash
kubectl exec -n <your_namespace> -c house "$(
  kubectl get pod -n <your_namespace> \
    -l app.kubernetes.io/instance=swanlab-self-hosted,app.kubernetes.io/service=house \
    -o jsonpath='{.items[0].metadata.name}'
)" -- wget -qO- http://127.0.0.1:3000/api/house/metrics
```

Notes:

- `app.kubernetes.io/instance=<release_name>` uses the default release name `swanlab-self-hosted` — replace with your actual deployment value
- `<your_namespace>` should be replaced with the namespace where SwanLab is deployed

## Integrating Monitoring Services

Choose the appropriate configuration method based on your environment:

- **Scenario 1: No Prometheus in the cluster** — Deploy an independent Prometheus + Grafana stack in the SwanLab namespace, and configure observability metrics
- **Scenario 2: Prometheus already exists in the cluster** — Integrate SwanLab metrics into your existing Prometheus monitoring system

### 1. Scenario 1: No Prometheus in the Cluster

This scenario applies when your cluster does not have an existing Prometheus monitoring setup. You will deploy a complete `Prometheus + Grafana` monitoring stack in the SwanLab namespace.

If your cluster already has a mature Prometheus observability service, you can skip this step and go to [Configure Prometheus Scrape Jobs](#_2-scenario-2-prometheus-already-exists-in-the-cluster).

#### 1.1 Create swanlab-monitor PVC

Create persistent storage for Prometheus and Grafana. This example uses 20Gi for each — adjust based on your cluster's actual usage.

::: details swanlab-monitor-pvc.yaml Example

```yaml
# ============================================================
# Prometheus + Grafana PVC Configuration
# ============================================================
# Pre-create persistent storage for Prometheus and Grafana
# Must be created and Bound before installing kube-prometheus-stack
#
# Usage:
#   kubectl apply -f swanlab-monitor-pvc.yaml
#
# Verify status:
#   kubectl get pvc -n <namespace>
#
# Notes:
#   - Prometheus PVC name must exactly match the Operator (StatefulSet) auto-generated name:
#     prometheus-<CR-name>-db-prometheus-<CR-name>-<ordinal>
#     where CR-name = <release>-kube-prome-prometheus (release name + chart name, truncated to 26 chars)
#     For release swanlab-monitor:
#     prometheus-swanlab-monitor-kube-prome-prometheus-db-prometheus-swanlab-monitor-kube-prome-prometheus-0
#     If the name doesn't match, the Operator will create a new PVC and the pre-created one will be unused
#     After installation, verify with: kubectl get pvc -n <namespace>
#   - Grafana PVC name is referenced via existingClaim in values
# ============================================================

# Prometheus data storage
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  # PVC name must exactly match the Operator auto-generated name (see header comments)
  name: prometheus-swanlab-monitor-kube-prome-prometheus-db-prometheus-swanlab-monitor-kube-prome-prometheus-0
  namespace: <your_namespace> # TODO: replace with actual namespace
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi # TODO: scale as needed
  storageClassName: <your_storageClassName> # TODO: replace with actual storage class
  volumeMode: Filesystem
---
# Grafana data storage
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  # Grafana PVC name is referenced via existingClaim in values
  name: swanlab-monitor-grafana-pvc
  namespace: <your_namespace> # TODO: replace with actual namespace
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi # TODO: scale as needed
  storageClassName: <your_storageClassName> # TODO: replace with actual storage class
  volumeMode: Filesystem
```

:::

Apply the PVC configuration:

```bash
kubectl apply -f swanlab-monitor-pvc.yaml

# Verify PVC status (ensure all are Bound)
kubectl get pvc -n <your_namespace>
```

#### 1.2 Configure swanlab-monitor-value

Create a `swanlab-monitor-value.yaml` file with Prometheus scrape job configuration and PVC references:

::: details swanlab-monitor-value.yaml Example

```yaml
# ============================================================
# kube-prometheus-stack values — using Alibaba Cloud ACR images
# ============================================================
# For deploying Prometheus + Grafana independently in the SwanLab namespace
# Auto-discovers SwanLab services via Pod Annotations
#
# Usage:
#   helm install swanlab-monitor prometheus-community/kube-prometheus-stack \
#     -n <namespace> -f swanlab-monitor-value.yaml
#
# Prerequisites:
#   1. PVC created (see swanlab-monitor-pvc.yaml)
#   2. Helm repo added:
#      helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
# ============================================================

# ---------- Grafana ----------
# Grafana visualizes Prometheus data and provides SwanLab monitoring dashboards
grafana:
  persistence:
    enabled: true
    # Use pre-created PVC (must be created and Bound before installation)
    existingClaim: swanlab-monitor-grafana-pvc
  # Grafana admin default password (default username is admin)
  # ⚠️ Security note: This is an example default password. Change it immediately after installation,
  # or use admin.existingSecret to reference a Kubernetes Secret for credential management
  adminPassword: "swanlab-monitor@default"

  # Grafana image (Alibaba Cloud ACR)
  image:
    registry: repo.swanlab.cn
    repository: public/grafana
    tag: "13.0.1-security-01"

  # initChownData container image (for initializing data directory permissions)
  initChownData:
    image:
      registry: repo.swanlab.cn
      repository: public/busybox
      tag: "1.38.0"

  # sidecar container image (for auto-loading dashboards and datasources from ConfigMap)
  sidecar:
    image:
      registry: repo.swanlab.cn
      repository: public/k8s-sidecar
      tag: "2.7.3"

  replicas: 1

  # Node scheduling configuration (fill as needed, example: node-role.kubernetes.io/monitor: "")
  nodeSelector: {}
  tolerations: []

# ---------- Prometheus ----------
# Prometheus collects and stores SwanLab metrics data
prometheus:
  prometheusSpec:
    # Allow selecting all ServiceMonitors (not limited to Helm-managed ones)
    serviceMonitorSelectorNilUsesHelmValues: false

    # Persistent storage configuration
    # Operator generates PVC for StatefulSet via volumeClaimTemplate, naming format:
    #   prometheus-<CR-name>-db-prometheus-<CR-name>-<ordinal> (CR-name = <release>-kube-prome-prometheus)
    # For this release (swanlab-monitor):
    #   prometheus-swanlab-monitor-kube-prome-prometheus-db-prometheus-swanlab-monitor-kube-prome-prometheus-0
    # This PVC has been pre-created with the correct name in swanlab-monitor-pvc.yaml
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: <your_storageClassName> # TODO: replace with actual storage class, must match swanlab-monitor-pvc.yaml
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 20Gi # must match the pre-created PVC capacity in swanlab-monitor-pvc.yaml

    replicas: 1

    # Node scheduling configuration (fill as needed, example: node-role.kubernetes.io/monitor: "")
    nodeSelector: {}
    tolerations: []

    # Prometheus image (Alibaba Cloud ACR)
    image:
      registry: repo.swanlab.cn
      repository: public/prometheus
      tag: "v3.12.0-distroless"

    # Custom scrape config — SwanLab dedicated scrape job
    # Auto-discovers SwanLab services via Pod Annotations
    additionalScrapeConfigs:
      - job_name: "swanlab"
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          # Only scrape Pods annotated with prometheus.io/scrape: "swanlab"
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: swanlab
          # Read metrics_path from annotation
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          # Read port from annotation
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            target_label: __address__
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
          # Retain common Kubernetes labels for Grafana queries and grouping
          - source_labels: [__meta_kubernetes_namespace]
            target_label: namespace
          - source_labels: [__meta_kubernetes_pod_name]
            target_label: pod
          - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_instance]
            target_label: release
          - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_service]
            target_label: service

# ---------- Alertmanager ----------
# Alertmanager handles Prometheus alert rules and sends notifications
alertmanager:
  replicas: 1
  alertmanagerSpec:
    # Node scheduling configuration (fill as needed, example: node-role.kubernetes.io/monitor: "")
    nodeSelector: {}
    tolerations: []
    image:
      registry: repo.swanlab.cn
      repository: public/alertmanager
      tag: "v0.32.2"

# ---------- Prometheus Operator ----------
# Prometheus Operator manages Prometheus and Alertmanager instances
prometheusOperator:
  replicas: 1
  # Node scheduling configuration (fill as needed, example: node-role.kubernetes.io/monitor: "")
  nodeSelector: {}
  tolerations: []
  image:
    registry: repo.swanlab.cn
    repository: public/prometheus-operator
    tag: "v0.91.0"
  # admissionWebhook validates PrometheusRule and ServiceMonitor configurations
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
  # prometheusConfigReloader auto-reloads Prometheus config on ConfigMap changes
  prometheusConfigReloader:
    image:
      registry: repo.swanlab.cn
      repository: public/prometheus-config-reloader
      tag: "v0.91.0"

# ---------- Kube State Metrics ----------
# kube-state-metrics exports cluster resource metrics from the Kubernetes API
kube-state-metrics:
  replicas: 1
  # Node scheduling configuration (fill as needed, example: node-role.kubernetes.io/monitor: "")
  nodeSelector: {}
  tolerations: []
  image:
    registry: repo.swanlab.cn
    repository: public/kube-state-metrics
    tag: "v2.19.0"

# ---------- Node Exporter (DaemonSet) ----------
# node-exporter collects node-level hardware and OS metrics
# Note: chart defaults distroless: true, which auto-appends -distroless suffix
#       So tag should be "v1.11.1" without -distroless
prometheus-node-exporter:
  # Node scheduling configuration (fill as needed, example: node-role.kubernetes.io/monitor: "")
  # ⚠️ node-exporter is a DaemonSet — setting nodeSelector limits it to matching nodes only
  nodeSelector: {}
  # Disable hostNetwork to avoid port conflicts (default is true)
  hostNetwork: false
  hostPort:
    enabled: false
  # Tolerate all taints to ensure DaemonSet runs on all nodes (including control plane)
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

:::warning ⚠️ Security Note
The Grafana admin password `swanlab-monitor@default` is an example default value. Change the admin password immediately after installation. For production, use `grafana.admin.existingSecret` to reference a Kubernetes Secret for credential management, avoiding plaintext passwords in values files.
:::

#### 1.3 Install Prometheus + Grafana

Similar to the [Upgrade & Rollback](./upgrade.md) section, choose the installation method based on whether your cluster can access `github.com`:

- **Cluster can access github**

First add the prometheus-community `kube-prometheus-stack` chart repo:

```bash
# Add Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install via Helm repo
helm install swanlab-monitor prometheus-community/kube-prometheus-stack \
  -n <your_namespace> \
  -f swanlab-monitor-value.yaml
```

- **Cluster cannot access github**

Pull the `kube-prometheus-stack` chart locally via OCI, then install:

```bash
# 1. Pull chart package locally
helm pull oci://swanlab-registry.cn-hangzhou.cr.aliyuncs.com/chart/monitoring/kube-prometheus-stack \
  --version 86.2.1
# 2. Extract
tar -zxvf kube-prometheus-stack-86.2.1.tgz
# 3. Install using local chart
helm install swanlab-monitor ./kube-prometheus-stack \
  -n <your_namespace> \
  -f swanlab-monitor-value.yaml
```

Wait for all pods and deployments to be ready:

```bash
# deployments
kubectl get deployments -n <your_namespace> | grep monitor

# pods
kubectl get pods -n <your_namespace> | grep monitor
```

### 2. Scenario 2: Prometheus Already Exists in the Cluster

This scenario applies when your cluster already has a mature Prometheus monitoring setup. You only need to integrate SwanLab metrics into the existing Prometheus scrape jobs. Three integration methods are provided — choose one:

| Method                                                                                            | Description                                           | Prerequisites                                                                                 | Requires Pod Annotation Changes |
| ------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------- |
| [2.1 Method 1: Pod Annotation (Recommended)](#_2-1-method-1-pod-annotation-recommended)           | Single job, dynamic service discovery via annotations | Requires [Step 3](#_3-configure-pod-annotations-and-update-services) to configure annotations | ✅ Yes                          |
| [2.2 Method 2: Label-Based Dual Jobs](#_2-2-method-2-label-based-dual-jobs)                       | Two independent jobs, discovery via Pod labels        | Uses default labels from SwanLab Helm Chart                                                   | ❌ No                           |
| [2.3 Method 3: Prometheus Operator](#_2-3-method-3-prometheus-operator-servicemonitor-podmonitor) | Declarative integration via ServiceMonitor/PodMonitor | Cluster has Prometheus Operator deployed                                                      | ❌ No                           |

#### 2.1 Method 1: Pod Annotation (Recommended)

Add a dedicated SwanLab scrape job to your existing `prometheus.yaml` `scrape_configs`. This approach uses a single job to collect both Server and House metrics via Pod annotations — adding new services only requires adding annotations, no Prometheus config changes needed.

::: details prometheus.yaml — Pod Annotation Scrape Job

```yaml
scrape_configs:
  - job_name: "swanlab"
    kubernetes_sd_configs:
      - role: pod

    relabel_configs:
      # Only scrape Pods annotated with prometheus.io/scrape: "swanlab"
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: swanlab

      # Use prometheus.io/path annotation as metrics_path
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)

      # Use prometheus.io/port annotation as scrape port
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        target_label: __address__
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2

      # Retain common Kubernetes labels for Grafana queries and grouping
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

When using Method 1, you must also complete [Step 3: Configure Pod Annotations](#_3-configure-pod-annotations-and-update-services).

#### 2.2 Method 2: Label-Based Dual Jobs

Add two independent scrape jobs to your `prometheus.yaml` `scrape_configs`, one for Server and one for House. This method uses the `app.kubernetes.io/instance` and `app.kubernetes.io/service` labels auto-generated by the SwanLab Helm Chart for service discovery — **no Pod annotation changes required**, works immediately after deployment.

::: details prometheus.yaml — Label-Based Dual Jobs

```yaml
scrape_configs:
  - job_name: "swanlab-server"
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - <your_namespace> # TODO: replace with SwanLab deployment namespace
    relabel_configs:
      # Only scrape Pods with release=<release_name> and service=server
      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_instance]
        action: keep
        regex: swanlab-self-hosted # TODO: replace with actual release name

      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_service]
        action: keep
        regex: server

      # Force Metrics port
      - source_labels: [__address__]
        action: replace
        target_label: __address__
        regex: ([^:]+)(?::\d+)?
        replacement: $1:3000

      - target_label: __metrics_path__
        replacement: /metrics

      # Retain common Kubernetes labels
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
            - <your_namespace> # TODO: replace with SwanLab deployment namespace
    relabel_configs:
      # Only scrape Pods with release=<release_name> and service=house
      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_instance]
        action: keep
        regex: swanlab-self-hosted # TODO: replace with actual release name

      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_service]
        action: keep
        regex: house

      # Force Metrics port
      - source_labels: [__address__]
        action: replace
        target_label: __address__
        regex: ([^:]+)(?::\d+)?
        replacement: $1:3000

      # House Metrics path differs from Server
      - target_label: __metrics_path__
        replacement: /api/house/metrics

      # Retain common Kubernetes labels
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace

      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod

      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_service]
        target_label: service
```

:::

::: tip Differences from Method 1

- **No Pod annotation changes needed**: Uses default `app.kubernetes.io/*` labels from SwanLab Helm Chart.
- **Namespace-scoped**: Uses `namespaces.names` to limit SD scope, friendlier to clusters with tightened permissions.
- **Independent Jobs**: Separate jobs for Server and House, easier to monitor and debug in Prometheus Targets panel.
- Grafana dashboard template variable `$job` values are `swanlab-server` or `swanlab-house` (not `swanlab`).
  :::

#### 2.3 Method 3: Prometheus Operator (ServiceMonitor / PodMonitor)

If your cluster's Prometheus is managed by [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator) (e.g., kube-prometheus-stack, Bitnami Helm Charts), you can integrate SwanLab declaratively via `ServiceMonitor` or `PodMonitor` resources — no manual `prometheus.yaml` editing needed.

::: details ServiceMonitor Example

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: swanlab-server
  namespace: <your_namespace> # TODO: replace with SwanLab deployment namespace
  labels:
    release: kube-prometheus-stack # TODO: match your Prometheus Operator's serviceMonitorSelector
spec:
  namespaceSelector:
    matchNames:
      - <your_namespace>
  selector:
    matchLabels:
      app.kubernetes.io/instance: swanlab-self-hosted # TODO: replace with actual release name
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
  namespace: <your_namespace> # TODO: replace with SwanLab deployment namespace
  labels:
    release: kube-prometheus-stack # TODO: match your Prometheus Operator's serviceMonitorSelector
spec:
  namespaceSelector:
    matchNames:
      - <your_namespace>
  selector:
    matchLabels:
      app.kubernetes.io/instance: swanlab-self-hosted # TODO: replace with actual release name
      app.kubernetes.io/service: house
  endpoints:
    - port: http
      path: /api/house/metrics
      interval: 30s
```

:::

::: warning

- `metadata.labels.release` must match the Prometheus Operator's `serviceMonitorSelector`, otherwise the Operator will not pick up this ServiceMonitor. Check your existing Prometheus Helm values for `prometheus.prometheusSpec.serviceMonitorSelector`.
- Scenario 1's `swanlab-monitor-value.yaml` already sets `serviceMonitorSelectorNilUsesHelmValues: false`, so adding ServiceMonitors in Scenario 1 requires no additional configuration.
  :::

### 3. Configure Pod Annotations and Update Services

> This step only applies to [Method 1 (Pod Annotation)](#_2-1-method-1-pod-annotation-recommended). If using [Method 2 (Label-Based)](#_2-2-method-2-label-based-dual-jobs) or [Method 3 (ServiceMonitor)](#_2-3-method-3-prometheus-operator-servicemonitor-podmonitor), skip this step and go to [Step 4](#_4-configure-ingress).

In your SwanLab self-hosted `swanlab-self-hosted-value.yaml`, find `service.server.customPodAnnotations` and `service.house.customPodAnnotations`, and add the following Prometheus scrape annotations for SwanLab-Server and SwanLab-House:

::: details swanlab-self-hosted-value.yaml — Pod Annotations

```yaml
....
service:
  server:
    ....
    customPodAnnotations:
      prometheus.io/scrape: "swanlab"    # SwanLab dedicated scrape identifier
      prometheus.io/port: "3000"         # Server Metrics port
      prometheus.io/path: "/metrics"     # Server Metrics path
...
  house:
    ...
    customPodAnnotations:
      prometheus.io/scrape: "swanlab"    # SwanLab dedicated scrape identifier
      prometheus.io/port: "3000"         # House Metrics port
      prometheus.io/path: "/api/house/metrics"  # House Metrics path
    ...
```

:::

:::warning
⚠️ **Note**: `prometheus.io/port` and `prometheus.io/path` are built-in SwanLab service requirements and cannot be changed.
:::

After modifying the annotations, apply the changes using `helm upgrade` (see [Upgrade & Rollback](./upgrade.md)):

```bash
# Online update
helm upgrade swanlab-self-hosted swanlab/self-hosted \
  -f swanlab-self-hosted-value.yaml \
  -n <your_namespace>

# Or update using offline chart package
helm upgrade swanlab-self-hosted ./self-hosted \
  -f swanlab-self-hosted-value.yaml \
  -n <your_namespace>
```

### 4. Configure Ingress

Similarly, the `swanlab-monitor` service does not include Ingress gateway configuration. You need to configure an external access entry on your cluster's load balancer (or Ingress) for the **80 port** of the `swanlab-monitor-grafana` Service.

If you can configure port forwarding, use the following command and open `localhost:3000`:

```bash
kubectl port-forward svc/swanlab-monitor-grafana 3000:80 -n <your_namespace>
```

### 5. Import Grafana Dashboards

SwanLab provides official Grafana dashboard templates for monitoring visualization in both scenarios.

#### 5.1 Download Dashboard Templates

- [swanlab-monitor-config-server.json](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/swanlab-monitor-config-server.json) — Server service monitoring dashboard
- [swanlab-monitor-config-house.json](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/swanlab-monitor-config-house.json) — House service monitoring dashboard

#### 5.2 Import Steps

1. In Grafana, add a `prometheus` data source: click **Connections** → **Data Sources** → **Add new data source** → select **Prometheus**, and set the URL to your Prometheus endpoint.

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260609202045722.png"/>

For example:

- `swanlab-monitor-kube-prome-prometheus` is the Prometheus Service in the same namespace, exposing port `9090`
- `tenant-shaobo` is the namespace where `swanlab-self-hosted` is deployed

Set the **Prometheus server URL** to `http://swanlab-monitor-kube-prome-prometheus.tenant-shaobo:9090/`

> The example above applies when Prometheus is installed in the same namespace as `swanlab-self-hosted` per [Scenario 1](#_1-scenario-1-no-prometheus-in-the-cluster). If your Prometheus is deployed in a different namespace or outside the cluster, replace the URL with your actual accessible Prometheus address.

2. In Grafana, navigate to **Dashboards → New → Import**

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260609200709949.png"/>

3. Paste or upload `swanlab-monitor-config-server.json` and `swanlab-monitor-config-house.json`

4. Select the corresponding **dataSource**, **namespace**, **job**, and **service**.

The dashboards use template variables to adapt to different Prometheus configurations. After importing, select from the **top dropdown menus**:

| Variable      | Description                | Example Value                                                                   |
| ------------- | -------------------------- | ------------------------------------------------------------------------------- |
| `$datasource` | Prometheus data source     | Select your configured data source                                              |
| `$namespace`  | Kubernetes namespace       | `swanlab`                                                                       |
| `$job`        | Prometheus scrape job name | `swanlab`, `swanlab-server`, or `swanlab-house` (depends on integration method) |
| `$service`    | SwanLab service name       | `server` or `house`                                                             |

> **Note**: Template variables are automatically populated from Prometheus — no manual input needed.
>
> **Important**: The "CPU Usage" and "Memory Usage" panels in the House dashboard use kubelet/cAdvisor metrics (`container_cpu_usage_seconds_total`, `container_memory_working_set_bytes`), not from SwanLab service Metrics endpoints. Scenario 1's kube-prometheus-stack collects these by default; for Scenario 2, confirm your existing Prometheus is scraping cAdvisor metrics, otherwise these panels will show no data.

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260609201624323.png"/>

Once configured correctly, you should see service health metrics:

- **SwanLab-Server**:
  <img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260609201132687.png"/>

- **SwanLab-House**:
  <img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260609201039152.png"/>

## Log Collection

> 🚧 The log collection configuration guide (e.g., `Loki + Promtail`, `ELK` etc.) is being written. Stay tuned.
> In the meantime, you can view service Pod logs via `kubectl logs`, or use your cloud provider's built-in cluster Pod log service:
>
> ```bash
> kubectl logs -n <your_namespace> <pod_name> -c <container_name>
> ```

## FAQ

### Why does the Metrics endpoint return 404?

The most likely cause is an incorrect request method. Make sure you are using `HTTP GET` to access the metrics endpoint. Additionally, ensure the service, port, and path are all correct.

### What do the metrics returned by the Metrics endpoint represent?

The Metrics endpoint follows the Prometheus format specification and typically returns information such as request QPS, request latency, and request error rate, along with internal runtime metrics for Node.js, Go, and other languages. Due to the large number of metrics, it is difficult to list all of them and their meanings. We recommend verifying the metrics endpoint as described in [Prerequisites](#prerequisites), or manually retrieving all metrics information from the Prometheus panel, and then using other tools (such as large language models) to look up the meaning of specific metrics.

### Does the Metrics endpoint return CPU, memory, and other metrics?

The Metrics endpoint does not collect CPU, memory, or other hardware metrics.

First, for performance reasons, the SwanLab service Metrics endpoint primarily exposes application runtime status metrics and does not include system resource metrics such as CPU and memory. Collecting CPU and other resource information may increase the application burden. On the other hand, CPU and memory metric collection may require higher permissions, which does not align with SwanLab's self-hosted deployment requirements. Finally, in cloud-native environments, these resource metrics are typically collected uniformly by [cAdvisor](https://github.com/google/cadvisor), [node-exporter](https://github.com/prometheus/node_exporter), or cloud provider monitoring components. Consider deploying the corresponding components to collect CPU and other data.

### Why do panels in the SwanLab monitoring dashboards show no data?

If CPU, memory, or other panels show no data, as mentioned in the previous question, you should consider deploying the corresponding hardware monitoring components. If you have already deployed the components, or if panels such as request latency show no data, the recommended troubleshooting steps are:

1. Check whether the corresponding metric exists in the Prometheus panel;
2. If it exists, the Grafana dashboard metric query configuration may be incorrect and needs to be modified;
3. If it does not exist, there is an issue with the Prometheus scrape job that needs to be investigated.

### Is monitoring of PostgreSQL, ClickHouse, and other infrastructure services supported?

PostgreSQL and ClickHouse have corresponding exporters (e.g., [postgres_exporter](https://github.com/prometheus-community/postgres_exporter)), but they require higher deployment permissions.
Future updates will consider integrating infrastructure service metrics into the Grafana dashboards.
