# 监控与日志配置指南

> 本文档介绍了利用 `Prometheus + Grafana` 监测 SwanLab 线上应用的配置方法。

:::info
受限于各种集群权限要求，在私有化 `App ≥ 3.0.0` 版本，SwanLab 采用 Prometheus + Grafana + AlertManager 独立看板的部署模式。
:::

## ☀️ 架构概述

SwanLab 私有化部署采用微服务架构，各应用服务按照职责拆分并独立运行，整体监控链路如下：

1. **Prometheus** 定期抓取 SwanLab 各个服务暴露的 `/metrics` 接口。
2. **Grafana** 从 Prometheus 读取数据，并渲染 SwanLab 的监控仪表盘和告警面板。
3. **「可选」Alertmanager** 或您已有的告警系统在 Prometheus 告警规则触发时发送通知。

## 🪜 流程示意

## 🧱 前置条件

- 已通过 Helm 安装 SwanLab 私有化服务（参考 [Kubernetes 部署指南](./deploy.md)）
- 对 SwanLab 私有化服务所在命名空间具有 admin 权限
- 应用默认 `release_name` 为 `swanlab-self-hosted`，安装命名空间为 `<your_namespace>`，存储类 storageClass 为 `<your_storageclass>`（请根据实际情况替换）

下表为 SwanLab 服务目前支持访问 metrics 信息的应用和对应接口配置、路由：

| 服务名称       | 服务说明         | 端口 | 路由     |
| -------------- | ---------------- | ---- | -------- |
| SwanLab-Server | 后端核心业务服务 | 3000 | /metrics |
| SwanLab-House  | 实验指标OLAP服务 | 3000 | /metrics |
|Vector | 指标转发缓冲队列 | 9090 | - |

如果在安装时，`Redis/PostgreSQL/ClickHouse` 基础数据库服务**未外部集成**，可以通过安装一些基本采集服务的方式，将可观测指标转发到 Promethues 。


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

## 📊 可观测监控服务


### 1. 开启 value 监控配置
在 `values.yaml` 中，对需要采集可观测指标的服务开启配置，格式例如下列所示：

```yaml
# 应用服务
service:
  server:
  ...
  # 是否开启监控采集专用 Headless Service
    monitor:
      enable: true

# 基础组件服务
dependencies:
  ...
  clickhouse:
    ...
    # 是否开启监控采集专用 Headless Service
    monitor:
      enable: true
```

> ⚠️ 注意： `denpendencies` 下的数据库依赖服务仅在未集成外部服务的情况下才能生效

修改完 `value.yaml` 后执行更新：
```bash
helm upgrade swanlab-self-hosted <path_to_chart> -n <your_namespace>
```

更新完成后，在各开启 `monitor` 配置的 SVC 下会额外新建一个独立 `monitor` headless 服务用于可观测指标采集。


### 2. 安装 SwanLab-Monitor 独立监控

SwanLab-Monitor 集成了 `Prometheus + Grafana` 的镜像和可观测指标的采集配置，需要在 SwanLab 所在命名空间下
安装两个单实例 StatefulSet 服务，模板如下所示⬇️：

#### 2.1 Prometheus + Grafanna 监控服务安装
:::details swanlab-monitor.yaml 模板
```yaml
# ============================================================
# SwanLab Monitor — Pod-level scraping via Headless Service DNS
# ============================================================
# ============================================================
# 占位符清单（搜索 # ← 替换）—— 3 个环境占位符，可被 render.sh 批量替换：
#   示例：swanlab-self-hosted  
#                        release 含 "self-hosted" → fullname = release 名（如本例）
#                        release 不含 "self-hosted"（如 swanlab-my）→ fullname = swanlab-my-self-hosted
#   <YOUR_NAMESPACE>     K8s 命名空间          默认/示例：tenant-shaobo
#   <STORAGE_CLASS_NAME> StorageClass（PVC）   默认/示例：disk-essd-auto-delete
#
# ============================================================

# ---------- Prometheus ConfigMap ----------
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-monitor-prometheus-config
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: prometheus
    app.kubernetes.io/instance: swanlab-monitor
data:
  prometheus.yml: |
    global:
      scrape_interval: 30s
      evaluation_interval: 30s
      external_labels:
        monitor: swanlab-monitor

    scrape_configs:
      # ---- SwanLab Server ----
      # dns_sd_configs 通过监控专用 Headless Service A 记录发现所有 Pod IP
      # Headless Service 名 = <fullname>-server-monitor，由 chart values monitorService: true 创建
      # 通过 relabel_configs 静态注入 namespace / service 标签，匹配 Grafana 看板变量
      - job_name: "swanlab-server"
        metrics_path: /metrics
        dns_sd_configs:
          - names:
              - swanlab-self-hosted-server-monitor.<YOUR_NAMESPACE>.svc.cluster.local
            type: A
            port: 3000
        relabel_configs:
          - target_label: namespace
            replacement: <YOUR_NAMESPACE>
          - target_label: service
            replacement: server

      # ---- SwanLab House ----
      - job_name: "swanlab-house"
        metrics_path: /metrics
        dns_sd_configs:
          - names:
              - swanlab-self-hosted-house-monitor.<YOUR_NAMESPACE>.svc.cluster.local
            type: A
            port: 3000
        relabel_configs:
          - target_label: namespace
            replacement: <YOUR_NAMESPACE>
          - target_label: service
            replacement: house

      # ---- Vector 日志聚合 ----
      # Vector 内置 prometheus_exporter sink，端口 9090，路径 /metrics
      - job_name: "swanlab-vector"
        metrics_path: /metrics
        dns_sd_configs:
          - names:
              - swanlab-self-hosted-vector-monitor.<YOUR_NAMESPACE>.svc.cluster.local
            type: A
            port: 9090
        relabel_configs:
          - target_label: namespace
            replacement: <YOUR_NAMESPACE>
          - target_label: service
            replacement: vector

      # ---- ClickHouse 数据库 ----
      # 仅在非外部托管时生效（integrations.clickhouse.enabled: false）
      # ClickHouse 内置 prometheus exporter，端口 9363（独立于 HTTP 8123），路径 /metrics
      - job_name: "swanlab-clickhouse"
        metrics_path: /metrics
        dns_sd_configs:
          - names:
              - swanlab-self-hosted-clickhouse-monitor.<YOUR_NAMESPACE>.svc.cluster.local
            type: A
            port: 9363
        relabel_configs:
          - target_label: namespace
            replacement: <YOUR_NAMESPACE>
          - target_label: service
            replacement: clickhouse

      # ---- ClickHouse per-table exporter ----
      # 独立 Deployment（见 components/ch-exporter.yaml，按需安装）
      # 查询 CH system.parts 暴露每表 bytes/rows/parts，端口 9364
      # 注：swanlab-monitor-* 为本监控栈固定 Service 名（非 swanlab-self-hosted），仅 namespace 需替换
      - job_name: "swanlab-clickhouse-tables"
        metrics_path: /metrics
        static_configs:
          - targets:
              - swanlab-monitor-ch-table-exporter.<YOUR_NAMESPACE>:9364
        relabel_configs:
          - target_label: namespace
            replacement: <YOUR_NAMESPACE>
          - target_label: service
            replacement: clickhouse

      # ---- PostgreSQL exporter ----
      # 独立 Deployment（见 components/postgres-exporter.yaml，按需安装）
      # prometheuscommunity/postgres-exporter，端口 9187
      - job_name: "swanlab-postgres"
        metrics_path: /metrics
        static_configs:
          - targets:
              - swanlab-monitor-postgres-exporter.<YOUR_NAMESPACE>:9187
        relabel_configs:
          - target_label: namespace
            replacement: <YOUR_NAMESPACE>
          - target_label: service
            replacement: postgres

      # ---- Redis exporter ----
      # 独立 Deployment（见 components/redis-exporter.yaml，按需安装）
      # oliver006/redis_exporter，端口 9121
      - job_name: "swanlab-redis"
        metrics_path: /metrics
        static_configs:
          - targets:
              - swanlab-monitor-redis-exporter.<YOUR_NAMESPACE>:9121
        relabel_configs:
          - target_label: namespace
            replacement: <YOUR_NAMESPACE>
          - target_label: service
            replacement: redis

      # ---- Prometheus 自身 ----
      - job_name: "prometheus"
        static_configs:
          - targets: ["localhost:9090"]

    rule_files:
      - /etc/prometheus/rules/*.yml

    # ---- 对接 Alertmanager ----
    # firing 告警发送到 monitor-alerting.yaml 部署的 Alertmanager，再路由到各 IM 通道
    # Alertmanager 是已知固定地址，用 static_configs 即可（无需 dns_sd）
    # 注：swanlab-monitor-alertmanager 为固定 Service 名（由 monitor-alerting-template.yaml 创建）
    alerting:
      alertmanagers:
        - static_configs:
            - targets:
                - swanlab-monitor-alertmanager.<YOUR_NAMESPACE>:9093

---
# ---------- Prometheus StatefulSet ----------
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: swanlab-monitor-prometheus
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: prometheus
    app.kubernetes.io/instance: swanlab-monitor
spec:
  serviceName: swanlab-monitor-prometheus
  replicas: 1
  updateStrategy:
    type: RollingUpdate
  selector:
    matchLabels:
      app.kubernetes.io/name: prometheus
      app.kubernetes.io/instance: swanlab-monitor
  template:
    metadata:
      labels:
        app.kubernetes.io/name: prometheus
        app.kubernetes.io/instance: swanlab-monitor
    spec:
      # 显式声明使用 default SA + 禁用 token 自动挂载
      serviceAccountName: default
      automountServiceAccountToken: false
      securityContext:
        fsGroup: 65534
        runAsUser: 65534
        runAsGroup: 65534
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: prometheus
          image: repo.swanlab.cn/public/prometheus:v3.12.0-distroless
          imagePullPolicy: IfNotPresent
          securityContext:                # 容器级加固：禁提权 + 弃所有 capabilities + 只读根 FS
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            readOnlyRootFilesystem: true
          args:
            - "--config.file=/etc/prometheus/prometheus.yml"
            - "--storage.tsdb.path=/prometheus"
            - "--storage.tsdb.retention.time=7d" # ← 可按需修改 retention.time（7d/15d/30d 等） 
            - "--storage.tsdb.retention.size=15GiB" # ← 可按需修改 retention.size（如 15GiB），不得超过 PVC 容量
            - "--web.enable-lifecycle"
          ports:
            - name: web
              containerPort: 9090
          volumeMounts:
            - name: config
              mountPath: /etc/prometheus
              readOnly: true
            - name: rules
              mountPath: /etc/prometheus/rules
              readOnly: true
            - name: data
              mountPath: /prometheus
            - name: tmp                   # readOnlyRootFilesystem 下供 Prometheus 写临时文件
              mountPath: /tmp
          readinessProbe:
            httpGet:
              path: /-/ready
              port: web
            initialDelaySeconds: 10
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /-/healthy
              port: web
            initialDelaySeconds: 30
            periodSeconds: 30
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "1000m"
              memory: "1Gi"
      volumes:
        - name: config
          configMap:
            name: swanlab-monitor-prometheus-config
        - name: rules
          configMap:
            name: swanlab-monitor-prometheus-rules
        - name: tmp                      # 配合 readOnlyRootFilesystem
          emptyDir: {}
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: "20Gi"
        storageClassName: <STORAGE_CLASS_NAME>
        volumeMode: Filesystem

---
# ---------- Prometheus Service ----------
apiVersion: v1
kind: Service
metadata:
  name: swanlab-monitor-prometheus
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: prometheus
    app.kubernetes.io/instance: swanlab-monitor
spec:
  type: ClusterIP
  ports:
    - name: web
      port: 9090
      targetPort: web
  selector:
    app.kubernetes.io/name: prometheus
    app.kubernetes.io/instance: swanlab-monitor

---
# ---------- Grafana Datasources ConfigMap ----------
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-monitor-grafana-datasources
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: grafana
    app.kubernetes.io/instance: swanlab-monitor
data:
  datasources.yaml: |
    apiVersion: 1
    datasources:
      - name: Prometheus
        type: prometheus
        access: proxy
        url: http://swanlab-monitor-prometheus.<YOUR_NAMESPACE>:9090
        isDefault: true
        editable: true

---
# ---------- Grafana StatefulSet ----------
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: swanlab-monitor-grafana
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: grafana
    app.kubernetes.io/instance: swanlab-monitor
spec:
  serviceName: swanlab-monitor-grafana
  replicas: 1
  updateStrategy:
    type: RollingUpdate
  selector:
    matchLabels:
      app.kubernetes.io/name: grafana
      app.kubernetes.io/instance: swanlab-monitor
  template:
    metadata:
      labels:
        app.kubernetes.io/name: grafana
        app.kubernetes.io/instance: swanlab-monitor
    spec:
      serviceAccountName: default
      automountServiceAccountToken: false
      securityContext:
        fsGroup: 472
        runAsNonRoot: true
        runAsUser: 472
        runAsGroup: 472
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: grafana
          image: repo.swanlab.cn/public/grafana:13.0.1-security-01
          imagePullPolicy: IfNotPresent
          securityContext:                # 容器级加固：禁提权 + 弃所有 capabilities + 只读根 FS
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            readOnlyRootFilesystem: true
          env:
            - name: GF_SECURITY_ADMIN_PASSWORD
              value: "swanlab-monitor@default"   
            - name: GF_USERS_ALLOW_SIGN_UP
              value: "false"
            - name: GF_SERVER_HTTP_PORT
              value: "3000"
          ports:
            - name: http
              containerPort: 3000
          volumeMounts:
            - name: data
              mountPath: /var/lib/grafana
            - name: provisioning-datasources
              mountPath: /etc/grafana/provisioning/datasources
              readOnly: true
            - name: tmp                   # readOnlyRootFilesystem 下供 Grafana 写临时文件
              mountPath: /tmp
          readinessProbe:
            httpGet:
              path: /api/health
              port: http
            initialDelaySeconds: 15
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /api/health
              port: http
            initialDelaySeconds: 45
            periodSeconds: 30
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "1"
              memory: "1Gi"
      volumes:
        - name: provisioning-datasources
          configMap:
            name: swanlab-monitor-grafana-datasources
        - name: tmp                      # 配合 readOnlyRootFilesystem
          emptyDir: {}
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: "20Gi"
        storageClassName: <STORAGE_CLASS_NAME>
        volumeMode: Filesystem

---
# ---------- Grafana Service ----------
apiVersion: v1
kind: Service
metadata:
  name: swanlab-monitor-grafana
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: grafana
    app.kubernetes.io/instance: swanlab-monitor
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 80
      targetPort: http
  selector:
    app.kubernetes.io/name: grafana
    app.kubernetes.io/instance: swanlab-monitor

---
# ---------- Prometheus Rules ConfigMap ----------
apiVersion: v1
kind: ConfigMap
metadata:
  name: swanlab-monitor-prometheus-rules
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: prometheus
    app.kubernetes.io/instance: swanlab-monitor
data:
  swanlab-alerts.yml: |
    # firing 告警由 Alertmanager 处理（见 monitor-alerting.yaml），按 receiver 路由到各 IM 通道。
    # 通道开关 / 收件人 / 分组策略均在 alertmanager.yml（Secret: swanlab-monitor-alertmanager-config）中配置。
    groups:
      - name: swanlab-alerts
        interval: 30s
        rules:
          # ---- 抓取健康 ----
          # 任一 swanlab job 抓取失败 5 分钟即告警
          - alert: SwanLabScrapeDown
            expr: up{job=~"swanlab-(server|house)"} == 0
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "{{ $labels.job }} 抓取失败"
              description: "instance={{ $labels.instance }} 已离线超过 5 分钟，Prometheus 无法抓取 /metrics"

          # ---- 基础设施抓取健康 ----
          # Vector / ClickHouse 抓取失败 5 分钟即告警
          - alert: SwanLabInfraScrapeDown
            expr: up{job=~"swanlab-(vector|clickhouse)"} == 0
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "{{ $labels.job }} 抓取失败"
              description: "instance={{ $labels.instance }} 已离线超过 5 分钟，Prometheus 无法抓取 /metrics"

          # ---- 服务端 5xx / panic 错误率 ----
          # 任一服务 5xx+exception 错误率 > 5% 持续 5 分钟
          - alert: SwanLabHigh5xxRate
            expr: |
              sum by (service, namespace) (
                rate(http_error_requests_total{error_type=~"server_error|exception", route!="/metrics"}[5m])
              )
              /
              sum by (service, namespace) (
                rate(http_requests_total{route!="/metrics"}[5m])
              ) > 0.05
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "{{ $labels.service }} 5xx 错误率过高"
              description: "{{ $labels.service }} 的服务端错误率超过 5%，持续 5 分钟"

          # ---- panic 异常 ----
          # 任一服务出现 panic（被 recover 捕获记为 exception）即告警
          - alert: SwanLabPanicSpike
            expr: rate(http_error_requests_total{error_type="exception", route!="/metrics"}[5m]) > 0
            for: 1m
            labels:
              severity: critical
            annotations:
              summary: "{{ $labels.service }} 检测到 panic"
              description: "instance={{ $labels.instance }} 在过去 5 分钟内出现 panic（被中间件 recover 捕获）"

          # ---- P99 延迟过高 ----
          # 任一服务 P99 > 5s 持续 5 分钟
          - alert: SwanLabLatencyP99High
            expr: |
              histogram_quantile(0.99,
                sum by (le, service, namespace) (
                  rate(http_request_duration_seconds_bucket{route!="/metrics"}[5m])
                )
              ) > 5
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "{{ $labels.service }} P99 延迟过高"
              description: "{{ $labels.service }} 的 P99 延迟超过 5s，持续 5 分钟"

          # ---- Pod 频繁重启 ----
          # instance 的 process_start_time_seconds 在 10 分钟内变化 > 2 次
          - alert: SwanLabPodRestart
            expr: changes(process_start_time_seconds{job=~"swanlab-(server|house)"}[10m]) > 2
            for: 0m
            labels:
              severity: warning
            annotations:
              summary: "{{ $labels.service }} pod 频繁重启"
              description: "instance={{ $labels.instance }} 在 10 分钟内重启超过 2 次"

          # ---- ClickHouse 磁盘使用率过高 ----
          # default disk 使用率 > 85% 持续 10 分钟
          - alert: SwanLabClickHouseDiskHigh
            expr: |
              ClickHouseAsyncMetrics_DiskUsed_default{job="swanlab-clickhouse"}
              /
              ClickHouseAsyncMetrics_DiskTotal_default{job="swanlab-clickhouse"} > 0.85
            for: 10m
            labels:
              severity: warning
            annotations:
              summary: "ClickHouse 磁盘使用率过高"
              description: "instance={{ $labels.instance }} 磁盘使用率超过 85%，持续 10 分钟"

          # ---- ClickHouse Parts 数过高（TooManyParts 风险）----
          # 单分区最大 part 数 > 100 持续 10 分钟
          - alert: SwanLabClickHouseTooManyParts
            expr: ClickHouseAsyncMetrics_MaxPartCountForPartition{job="swanlab-clickhouse"} > 100
            for: 10m
            labels:
              severity: warning
            annotations:
              summary: "ClickHouse Parts 数过高"
              description: "instance={{ $labels.instance }} 单分区最大 Parts 数超过 100，存在 TooManyParts 风险"

          # ---- Vector Disk Buffer 积压 ----
          # disk buffer 使用率 > 50% 持续 10 分钟（说明 ClickHouse sink 消费不及时）
          - alert: SwanLabVectorDiskBufferBacklog
            expr: |
              (
                vector_buffer_byte_size{buffer_type="disk", job="swanlab-vector"}
                / on (component_id, host)
                vector_buffer_max_byte_size{buffer_type="disk", job="swanlab-vector"}
              ) > 0.5
            for: 10m
            labels:
              severity: warning
            annotations:
              summary: "Vector Disk Buffer 积压"
              description: "component={{ $labels.component_id }} host={{ $labels.host }} 磁盘缓冲区使用率超过 50%，持续 10 分钟"

          # ---- PostgreSQL 宕机 ----
          # pg_up == 0 持续 1 分钟（exporter 连不上 PG 或 PG 进程挂了）
          - alert: SwanLabPostgresDown
            expr: pg_up{job="swanlab-postgres"} == 0
            for: 1m
            labels:
              severity: critical
            annotations:
              summary: "PostgreSQL 宕机"
              description: "instance={{ $labels.instance }} PostgreSQL 不可用，持续 1 分钟"

          # ---- PostgreSQL 连接数过高 ----
          # 活跃连接 > max_connections * 80% 持续 5 分钟
          - alert: SwanLabPostgresConnectionsHigh
            expr: |
              sum(pg_stat_activity_count{job="swanlab-postgres"})
              /
              pg_settings_max_connections{job="swanlab-postgres"} > 0.8
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "PostgreSQL 连接数过高"
              description: "活跃连接数超过最大连接数的 80%，持续 5 分钟"

          # ---- PostgreSQL 死锁 ----
          # 出现新死锁即告警（deadlocks 是累计计数器，rate > 0 表示有新增）
          - alert: SwanLabPostgresDeadlocks
            expr: rate(pg_stat_database_deadlocks{job="swanlab-postgres"}[5m]) > 0
            for: 1m
            labels:
              severity: warning
            annotations:
              summary: "PostgreSQL 检测到死锁"
              description: "database={{ $labels.datname }} 出现新的死锁"

          # ---- Redis 宕机 ----
          # redis_up == 0 持续 1 分钟（exporter 连不上 Redis 或 Redis 进程挂了）
          - alert: SwanLabRedisDown
            expr: redis_up{job="swanlab-redis"} == 0
            for: 1m
            labels:
              severity: critical
            annotations:
              summary: "Redis 宕机"
              description: "instance={{ $labels.instance }} Redis 不可用，持续 1 分钟"

          # ---- Redis 内存使用率过高 ----
          # used_memory / maxmemory > 85% 持续 5 分钟（maxmemory=0 即未限时通过 > 0 跳过）
          - alert: SwanLabRedisMemoryHigh
            expr: |
              redis_memory_used_bytes{job="swanlab-redis"}
              / redis_memory_max_bytes{job="swanlab-redis"} > 0.85
              and on(instance)
              redis_memory_max_bytes{job="swanlab-redis"} > 0
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "Redis 内存使用率过高"
              description: "instance={{ $labels.instance }} 内存使用率超过 85%，持续 5 分钟"

          # ---- Redis 拒绝连接（maxclients 打满）----
          # 出现被拒绝连接即告警（rejected_connections_total 为累计计数器）
          - alert: SwanLabRedisRejectedConnections
            expr: increase(redis_rejected_connections_total{job="swanlab-redis"}[5m]) > 0
            for: 1m
            labels:
              severity: warning
            annotations:
              summary: "Redis 拒绝了新连接"
              description: "instance={{ $labels.instance }} 达到 maxclients，出现被拒绝连接"

```
:::

其中:
- ` <YOUR_NAMESPACE>`: 安装 SwanLab 私有化服务的命名空间;
- `<STORAGE_CLASS_NAME>` StorageClass（PVC）: 存储 Prometheus 指标和Grafana配置文件的持久化存储类；
- `rentationTime` 和 `rentationSize` 分别代表可观测时序数据的过期时间和轮转存储大小，默认按照 7 天/ 15Gi 的大小进行配置，可以按需调整；
- 抓取配置与告警规则无需做额外修改；

所有服务的 DNS 服务地址均以 `swanlab-self-hosted` 为默认的 release 名称进行预设，如果您此前安装时**自行指定了 `release` 名称**，例如： `swanlab-my`，则相关服务的 host 前缀需要变更为 `swanlab-my-<SVC_NAME>-self-hosted`

在替换完对应字段后，可以安装 `Prometheus` + `Grafana` 两个独立 StatefulSet 服务
```bash
kubectl apply -f swanlab-monitor.yaml -n <your_namespace>
```


#### 「可选」2.2 数据库 Exporter 服务安装
针对 `Redis/PostgreSQL/ClickHouse` 数据库服务，需要各自额外安装 `SideCar` 采集服务用于转发指标

:::details DB-Exporter
:::code-group
```yaml [postgres-exporter.yaml]
# ============================================================
# SwanLab Monitor 组件 — PostgreSQL Exporter（可选，按需安装）
# ============================================================
# prometheuscommunity/postgres-exporter，查询 PG pg_stat_* 系统视图
# 暴露连接数 / 事务 / 锁 / 复制 / 缓存命中 / 数据库大小等 100+ 指标，端口 9187。
# 抓取 job 在 monitor-raw.yaml 的 prometheus 配置中（job_name: swanlab-postgres）。
#
# 依赖：chart 已部署 PostgreSQL（dependencies.postgres），凭据 Secret 由 chart 创建。
# 连接 PG 的凭据来自 chart 创建的 Secret（<POSTGRES_SECRET_NAME>）
# ============================================================

---
# ---------- PostgreSQL Exporter ----------
apiVersion: apps/v1
kind: Deployment
metadata:
  name: swanlab-monitor-postgres-exporter
  namespace: tenant-shaobo
  labels:
    app.kubernetes.io/name: postgres-exporter
    app.kubernetes.io/instance: swanlab-monitor
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: postgres-exporter
      app.kubernetes.io/instance: swanlab-monitor
  template:
    metadata:
      labels:
        app.kubernetes.io/name: postgres-exporter
        app.kubernetes.io/instance: swanlab-monitor
    spec:
      containers:
        - name: exporter
          image: <POSTGRES_EXPORTER_IMAGE>
          imagePullPolicy: Always
          ports:
            - containerPort: 9187
              name: metrics
          env:
            - name: PG_USER
              valueFrom:
                secretKeyRef:
                  name: <POSTGRES_SECRET_NAME>
                  key: username
            - name: PG_PASS
              valueFrom:
                secretKeyRef:
                  name: <POSTGRES_SECRET_NAME>
                  key: password
            - name: DATA_SOURCE_NAME
              value: "postgresql://$(PG_USER):$(PG_PASS)@swanlab-self-hosted-postgres.tenant-shaobo.svc.cluster.local:5432/app?sslmode=disable"
          securityContext:
            runAsNonRoot: true
            runAsUser: 65534
            runAsGroup: 65534
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            readOnlyRootFilesystem: true
          volumeMounts:
            - name: tmp
              mountPath: /tmp
          resources:
            requests:
              cpu: 10m
              memory: 32Mi
            limits:
              cpu: 100m
              memory: 128Mi
      volumes:
        - name: tmp
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: swanlab-monitor-postgres-exporter
  namespace: tenant-shaobo
  labels:
    app.kubernetes.io/name: postgres-exporter
    app.kubernetes.io/instance: swanlab-monitor
spec:
  clusterIP: None
  selector:
    app.kubernetes.io/name: postgres-exporter
    app.kubernetes.io/instance: swanlab-monitor
  ports:
    - port: 9187
      targetPort: metrics
      name: metrics

```

```yaml [redis-exporter.yaml]
# ============================================================
# SwanLab Monitor 组件 — Redis Exporter（可选，按需安装）
# ============================================================
# oliver006/redis_exporter，查询 Redis INFO 暴露内存/连接/命令统计/键空间等指标，端口 9121。
# 抓取 job 在 monitor-raw.yaml 的 prometheus 配置中（job_name: swanlab-redis）。
#
# 依赖：chart 已部署 Redis（dependencies.redis）。
# 当前 chart 的 Redis 无密码（secret 仅存 url，无 password key），故仅需 REDIS_ADDR。
# 连接 Redis 的地址为主 ClusterIP Service（Redis 单副本 + Recreate，无需逐 Pod 发现）。
# ============================================================

---
# ---------- Redis Exporter ----------
apiVersion: apps/v1
kind: Deployment
metadata:
  name: swanlab-monitor-redis-exporter
  namespace: tenant-shaobo
  labels:
    app.kubernetes.io/name: redis-exporter
    app.kubernetes.io/instance: swanlab-monitor
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: redis-exporter
      app.kubernetes.io/instance: swanlab-monitor
  template:
    metadata:
      labels:
        app.kubernetes.io/name: redis-exporter
        app.kubernetes.io/instance: swanlab-monitor
    spec:
      containers:
        - name: exporter
          image: <REDIS_EXPORTER_IMAGE>
          imagePullPolicy: Always
          ports:
            - containerPort: 9121
              name: metrics
          env:
            - name: REDIS_ADDR
              value: "redis://<HELM_RELEASE_NAME>-redis.tenant-shaobo.svc.cluster.local:6379"
          securityContext:
            runAsNonRoot: true
            runAsUser: 65534
            runAsGroup: 65534
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            readOnlyRootFilesystem: true
          resources:
            requests:
              cpu: 10m
              memory: 32Mi
            limits:
              cpu: 100m
              memory: 64Mi
---
apiVersion: v1
kind: Service
metadata:
  name: swanlab-monitor-redis-exporter
  namespace: tenant-shaobo
  labels:
    app.kubernetes.io/name: redis-exporter
    app.kubernetes.io/instance: swanlab-monitor
spec:
  clusterIP: None
  selector:
    app.kubernetes.io/name: redis-exporter
    app.kubernetes.io/instance: swanlab-monitor
  ports:
    - port: 9121
      targetPort: metrics
      name: metrics

```

```yaml [clickhouse-exporter.yaml]
# ============================================================
# SwanLab Monitor 组件 — ClickHouse Exporter（可选，按需安装）
# ============================================================
# CH 内置 exporter 只暴露聚合指标，此 Deployment 查询 system.parts
# 暴露每表 bytes/rows/parts 作为 Prometheus gauge 指标，端口 9364。
# 抓取 job 在 monitor-raw.yaml 的 prometheus 配置中（job_name: swanlab-clickhouse-tables）。
#
# 依赖：chart 已部署 ClickHouse（dependencies.clickhouse），凭据 Secret 由 chart 创建。
# 镜像: <CH_TABLE_EXPORTER_IMAGE>（已推送到 ACR）
# 连接 CH 的凭据来自 chart 创建的 Secret（<CLICKHOUSE_SECRET_NAME>）
# ============================================================

---
# ---------- ClickHouse Per-Table Exporter ----------
apiVersion: apps/v1
kind: Deployment
metadata:
  name: swanlab-monitor-ch-table-exporter
  namespace: tenant-shaobo
  labels:
    app.kubernetes.io/name: ch-table-exporter
    app.kubernetes.io/instance: swanlab-monitor
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: ch-table-exporter
      app.kubernetes.io/instance: swanlab-monitor
  template:
    metadata:
      labels:
        app.kubernetes.io/name: ch-table-exporter
        app.kubernetes.io/instance: swanlab-monitor
    spec:
      containers:
        - name: exporter
          image: <CH_TABLE_EXPORTER_IMAGE>
          imagePullPolicy: Always
          ports:
            - containerPort: 9364
              name: metrics
          env:
            - name: CH_HOST
              value: "swanlab-self-hosted-clickhouse.tenant-shaobo.svc.cluster.local"
            - name: CH_PORT
              value: "8123"
            - name: CLICKHOUSE_USER
              valueFrom:
                secretKeyRef:
                  name: <CLICKHOUSE_SECRET_NAME>
                  key: username
            - name: CLICKHOUSE_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: <CLICKHOUSE_SECRET_NAME>
                  key: password
          securityContext:
            runAsNonRoot: true
            runAsUser: 1000
            runAsGroup: 1000
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            readOnlyRootFilesystem: true
          resources:
            requests:
              cpu: 10m
              memory: 32Mi
            limits:
              cpu: 100m
              memory: 64Mi
---
apiVersion: v1
kind: Service
metadata:
  name: swanlab-monitor-ch-table-exporter
  namespace: tenant-shaobo
  labels:
    app.kubernetes.io/name: ch-table-exporter
    app.kubernetes.io/instance: swanlab-monitor
spec:
  clusterIP: None
  selector:
    app.kubernetes.io/name: ch-table-exporter
    app.kubernetes.io/instance: swanlab-monitor
  ports:
    - port: 9364
      targetPort: metrics
      name: metrics

```
:::


确认好需要观测的基础数据库服务后，可以执行如下命令进行安装：
```bash
# Redis
kubectl apply -f redis-exporter.yaml -n <your_namespace>

# PostgreSQL
kubectl apply -f postgres-exporter.yaml  -n <your_namespace>

# ClickHouse
kubectl apply -f clickhouse-exporter.yaml  -n <your_namespace>
```

安装完成后需要重启 Prometheus + Grafana 服务才能生效:

```bash
kubectl rollout restart statefulset swanlab-monitor-prometheus -n <your_namespace>

kubectl rollout restart statefulset swanlab-monitor-grafana -n <your_namespace>
```



### 3. 配置仪表盘

根据开启的可观测服务，可根据需要在 Grafana 中导入对应的看板配置

| 服务 | 配置模板 | 
| --- | ---- |
| SwanLab-Server | server_url |
| SwanLab-House | server_url |
| Vector | server_url |
| Redis | server_url |
| PostgreSQL | server_url |
| ClickHouse | server_url |


<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260609201624323.png"/>

配置正常后可以看到相关的服务检测指标

- **SwanLab-Server**:
  <img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260609201132687.png"/>

- **SwanLab-House**:
  <img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260609201039152.png"/>

### 4. 「可选」 AlertManager 通知告警服务

在 `swanlab-monitor.yaml` 中配置了关于服务指标异常时的告警阈值，但并未配置触发渠道，因此如果要实现阈值自动告警，需要安装额外的组件用于配置告警通知

#### 4.1 AlertManager 服务安装

:::details swanlab-monitor-alertmanager.yaml 配置
```yaml
# ============================================================
# SwanLab Monitor — 告警通道统一凭据 Secret
# ============================================================
# 所有 IM 通道的密钥收敛在这一个 Secret 里，部署时只需填一次。
# 各组件的读取方式：
#   Alertmanager — 挂载到 /etc/alertmanager/secrets/，用 api_url_file / url_file 引用
#   DingTalk 桥 — 挂载 dingtalk_config.yml 作为 config.yml
#   Feishu 桥   — envFrom 注入 FEISHU_WEBHOOK_URL / FEISHU_SECRET / MESSAGE_TYPE
#   WeCom 桥    — 无需凭据（key 在 wecom_webhook_url 里，Alertmanager 读取）
#
# 占位符清单（搜索 # ← 替换）：
#   <SLACK_TOKEN>           Slack Incoming Webhook 的 services/ 之后部分
#   <WECOM_BOT_KEY>         企业微信群机器人 webhook 的 key 参数
#   <DINGTALK_ACCESS_TOKEN> 钉钉机器人 webhook 的 access_token
#   <DINGTALK_SECRET>       钉钉机器人加签 secret（未开加签可留空字符串）
#   <FEISHU_WEBHOOK_URL>    飞书自定义机器人 webhook 完整 URL
#   <FEISHU_SECRET>         飞书机器人签名校验 secret（未开校验留空字符串）
# ============================================================
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-monitor-channels-credentials
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: alertmanager-channels
    app.kubernetes.io/instance: swanlab-monitor
type: Opaque
stringData:
  # ---- Slack（Alertmanager 通过 api_url_file 读取此文件）----
  slack_webhook_url: "https://hooks.slack.com/services/<SLACK_TOKEN>"       # ← Slack Incoming Webhook 完整 URL

  # ---- 企业微信（Alertmanager 通过 url_file 读取此文件，含完整 URL + key）----
  wecom_webhook_url: "http://swanlab-monitor-wecom-bridge.<YOUR_NAMESPACE>:5001/send?key=<WECOM_BOT_KEY>"   # ← 企业微信群机器人 key

  # ---- 钉钉（桥通过 subPath 挂载此 key 作为 config.yml）----
  dingtalk_config.yml: |
    targets:
      swanlab:
        url: https://oapi.dingtalk.com/robot/send?access_token=<DINGTALK_ACCESS_TOKEN>   # ← 钉钉机器人 access_token
        secret: <DINGTALK_SECRET>                                                         # ← 加签 secret；未开加签留空字符串
        mention:
          all: false

  # ---- 飞书（桥通过 envFrom 注入以下环境变量）----
  FEISHU_WEBHOOK_URL: "<FEISHU_WEBHOOK_URL>"           # ← 飞书自定义机器人完整 webhook URL
  FEISHU_SECRET: "<FEISHU_SECRET>"                     # ← 签名校验 secret；未开校验留空字符串
  MESSAGE_TYPE: "interactive"                           # interactive=卡片消息，text=纯文本

---
# ============================================================
# Alertmanager — 配置 + StatefulSet + Service
# ============================================================
# Prometheus (monitor.yaml) --firing--> Alertmanager --routing--> 各 IM 通道
#                                                               ├─ slack_configs     → Slack（原生，api_url_file 读 Secret）
#                                                               ├─ webhook → dingtalk-bridge:8060  ──→ 钉钉群
#                                                               ├─ webhook → feishu-bridge:8080    ──→ 飞书群
#                                                               └─ webhook (url_file) → wecom-bridge:5001 ──→ 企业微信群
#
# alertmanager.yml 本身不含任何密钥——所有 token/key 通过 file 引用从统一 Secret 读取。
# 通道开关：注释 receiver 里对应的配置块即禁用。
# ============================================================

# ---------- Alertmanager Config Secret ----------
# 纯路由配置，无密钥（密钥在 channels-credentials Secret 里）
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-monitor-alertmanager-config
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: alertmanager
    app.kubernetes.io/instance: swanlab-monitor
type: Opaque
stringData:
  alertmanager.yml: |
    global:
      resolve_timeout: 5m

    route:
      receiver: im-all
      group_by: ['alertname', 'service', 'namespace']
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 4h

    receivers:
      - name: im-all
        # ┌─────────────────────────────────────────────────────────────────┐
        # │ 通道开关：注释整块 = 禁用；取消注释 = 启用                        │
        # │ 至少保留一个通道启用                                              │
        # │ 密钥不在本文件——从 /etc/alertmanager/secrets/ 读取（file 引用）  │
        # └─────────────────────────────────────────────────────────────────┘

        # ---- [enable] Slack（原生 slack_configs，密钥从 Secret 文件读取）----
        slack_configs:
          - api_url_file: /etc/alertmanager/secrets/slack_webhook_url   # 从统一 Secret 读取 Slack webhook URL
            channel: '#swanlab-alerts'
            send_resolved: true

        # ---- 以下 webhook_configs 每条 = 一个 IM 通道桥，注释单条即禁用 ----
        webhook_configs:
          # ---- [enable] 钉钉（需部署下方 dingtalk-bridge）----
          - url: 'http://swanlab-monitor-dingtalk-bridge.<YOUR_NAMESPACE>:8060/dingtalk/swanlab/send'
            send_resolved: true

          # ---- [enable] 飞书（需部署下方 feishu-bridge）----
          - url: 'http://swanlab-monitor-feishu-bridge.<YOUR_NAMESPACE>:8080/webhook'
            send_resolved: true

          # ---- [enable] 企业微信（url 从 Secret 文件读取，含 key）----
          - url_file: /etc/alertmanager/secrets/wecom_webhook_url        # 从统一 Secret 读取完整 WeCom webhook URL（含 key）
            send_resolved: true

---
# ---------- Alertmanager StatefulSet ----------
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: swanlab-monitor-alertmanager
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: alertmanager
    app.kubernetes.io/instance: swanlab-monitor
spec:
  serviceName: swanlab-monitor-alertmanager
  replicas: 1
  updateStrategy:
    type: RollingUpdate
  selector:
    matchLabels:
      app.kubernetes.io/name: alertmanager
      app.kubernetes.io/instance: swanlab-monitor
  template:
    metadata:
      labels:
        app.kubernetes.io/name: alertmanager
        app.kubernetes.io/instance: swanlab-monitor
    spec:
      serviceAccountName: default
      automountServiceAccountToken: false
      securityContext:
        fsGroup: 65534
        runAsUser: 65534
        runAsGroup: 65534
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: alertmanager
          image: repo.swanlab.cn/public/alertmanager:v0.32.2
          imagePullPolicy: IfNotPresent
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            readOnlyRootFilesystem: true
          args:
            - "--config.file=/etc/alertmanager/alertmanager.yml"
            - "--storage.path=/alertmanager"
          ports:
            - name: web
              containerPort: 9093
          volumeMounts:
            - name: config
              mountPath: /etc/alertmanager
              readOnly: true
            - name: secrets               # 统一凭据 Secret 挂载（供 api_url_file / url_file 读取）
              mountPath: /etc/alertmanager/secrets
              readOnly: true
            - name: data
              mountPath: /alertmanager
            - name: tmp
              mountPath: /tmp
          readinessProbe:
            httpGet:
              path: /-/ready
              port: web
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /-/healthy
              port: web
            initialDelaySeconds: 30
            periodSeconds: 30
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 300m
              memory: 256Mi
      volumes:
        - name: config
          secret:
            secretName: swanlab-monitor-alertmanager-config
        - name: secrets                   # 统一凭据 Secret
          secret:
            secretName: swanlab-monitor-channels-credentials
        - name: tmp
          emptyDir: {}
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 20Gi
        storageClassName: <STORAGE_CLASS_NAME>
        volumeMode: Filesystem

---
# ---------- Alertmanager Service ----------
apiVersion: v1
kind: Service
metadata:
  name: swanlab-monitor-alertmanager
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: alertmanager
    app.kubernetes.io/instance: swanlab-monitor
spec:
  type: ClusterIP
  ports:
    - name: web
      port: 9093
      targetPort: web
  selector:
    app.kubernetes.io/name: alertmanager
    app.kubernetes.io/instance: swanlab-monitor


```
:::


#### 4.2 WebhookIM 告警通知配置

根据不同的 IM Channel，可以安装对应的告警服务

:::details Webhook IM 告警通知配置
:::code-group 
```yaml [dingtalk.yaml]
# ============================================================
# 钉钉桥 —— timonwong/prometheus-webhook-dingtalk
# 端点：/<target>，target 名 = config.yml 里的 key（此处 swanlab）
# Alertmanager POST 到 http://<svc>:8060/dingtalk/swanlab/send
# 密钥（access_token + 加签 secret）在统一凭据 Secret 里：
#   swanlab-monitor-channels-credentials → dingtalk_config.yml
# ============================================================

---
# ---------- DingTalk Bridge Deployment ----------
apiVersion: apps/v1
kind: Deployment
metadata:
  name: swanlab-monitor-dingtalk-bridge
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: dingtalk-bridge
    app.kubernetes.io/instance: swanlab-monitor
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: dingtalk-bridge
      app.kubernetes.io/instance: swanlab-monitor
  template:
    metadata:
      labels:
        app.kubernetes.io/name: dingtalk-bridge
        app.kubernetes.io/instance: swanlab-monitor
    spec:
      serviceAccountName: default
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        runAsUser: 65534
        runAsGroup: 65534
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: dingtalk-bridge
          image: repo.swanlab.cn/public/prometheus-webhook-dingtalk:v2.1.0
          imagePullPolicy: Always
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            readOnlyRootFilesystem: true
          args:
            - "--config.file=/etc/prometheus-webhook-dingtalk/config.yml"
            - "--web.listen-address=:8060"
            - "--web.enable-lifecycle"
          ports:
            - name: http
              containerPort: 8060
          volumeMounts:
            - name: credentials              # 从统一凭据 Secret 读取 dingtalk_config.yml
              mountPath: /etc/prometheus-webhook-dingtalk
              readOnly: true
            - name: tmp
              mountPath: /tmp
          readinessProbe:
            tcpSocket:
              port: 8060
            initialDelaySeconds: 5
            periodSeconds: 10
          resources:
            requests:
              cpu: 50m
              memory: 32Mi
            limits:
              cpu: 100m
              memory: 64Mi
      volumes:
        - name: credentials                  # 统一凭据 Secret（dingtalk_config.yml key → config.yml 文件）
          secret:
            secretName: swanlab-monitor-channels-credentials
            items:
              - key: dingtalk_config.yml
                path: config.yml             # 挂载为 config.yml（桥期望的文件名）
        - name: tmp
          emptyDir: {}

---
# ---------- DingTalk Bridge Service ----------
apiVersion: v1
kind: Service
metadata:
  name: swanlab-monitor-dingtalk-bridge
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: dingtalk-bridge
    app.kubernetes.io/instance: swanlab-monitor
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 8060
      targetPort: http
  selector:
    app.kubernetes.io/name: dingtalk-bridge
    app.kubernetes.io/instance: swanlab-monitor

```

```yaml [feishu.yaml]
# ============================================================
# 飞书桥 —— nirvam/alertmanager-feishu (dev 分支)
# 端点：POST /webhook，FastAPI + uvicorn，监听 8080
# 配置走环境变量：FEISHU_WEBHOOK_URL / FEISHU_SECRET / MESSAGE_TYPE
# 密钥在统一凭据 Secret 里（envFrom 注入）：
#   swanlab-monitor-channels-credentials → FEISHU_WEBHOOK_URL / FEISHU_SECRET / MESSAGE_TYPE
# ============================================================

---
# ---------- Feishu Bridge Deployment ----------
apiVersion: apps/v1
kind: Deployment
metadata:
  name: swanlab-monitor-feishu-bridge
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: feishu-bridge
    app.kubernetes.io/instance: swanlab-monitor
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: feishu-bridge
      app.kubernetes.io/instance: swanlab-monitor
  template:
    metadata:
      labels:
        app.kubernetes.io/name: feishu-bridge
        app.kubernetes.io/instance: swanlab-monitor
    spec:
      serviceAccountName: default
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: feishu-bridge
          image: repo.swanlab.cn/public/alertmanager-feishu:dev
          imagePullPolicy: IfNotPresent
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            readOnlyRootFilesystem: true
          command: ["/app/.venv/bin/alertmanager-feishu", "serve"]
          envFrom:
            - secretRef:
                name: swanlab-monitor-channels-credentials   # 从统一凭据 Secret 注入 FEISHU_* 环境变量
          volumeMounts:
            - name: tmp
              mountPath: /tmp
          ports:
            - name: http
              containerPort: 8080
          readinessProbe:
            tcpSocket:
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            tcpSocket:
              port: 8080
            initialDelaySeconds: 20
            periodSeconds: 30
          resources:
            requests:
              cpu: 100m
              memory: 64Mi
            limits:
              cpu: 200m
              memory: 128Mi
      volumes:
        - name: tmp
          emptyDir: {}

---
# ---------- Feishu Bridge Service ----------
apiVersion: v1
kind: Service
metadata:
  name: swanlab-monitor-feishu-bridge
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: feishu-bridge
    app.kubernetes.io/instance: swanlab-monitor
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 8080
      targetPort: http
  selector:
    app.kubernetes.io/name: feishu-bridge
    app.kubernetes.io/instance: swanlab-monitor
```

```yaml [wecom.yaml]

# ============================================================
# 企业微信桥 —— rea1shane/a2w
# 端点：/send?key=<KEY>，无状态（key 在 Alertmanager 的 webhook URL 里，不在桥内配置）
# 监听 5001，默认时区 Asia/Shanghai
# Alertmanager POST 到 http://<svc>:5001/send?key=<WECOM_BOT_KEY>
# 如需 @指定用户：URL 追加 &mention=user1&mention=user2
# ============================================================
---
# ---------- WeCom Bridge Deployment ----------
apiVersion: apps/v1
kind: Deployment
metadata:
  name: swanlab-monitor-wecom-bridge
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: wecom-bridge
    app.kubernetes.io/instance: swanlab-monitor
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: wecom-bridge
      app.kubernetes.io/instance: swanlab-monitor
  template:
    metadata:
      labels:
        app.kubernetes.io/name: wecom-bridge
        app.kubernetes.io/instance: swanlab-monitor
    spec:
      serviceAccountName: default
      automountServiceAccountToken: false
      securityContext:                # 最小权限 + PSS restricted 兼容（非 root + seccomp）
        runAsNonRoot: true
        runAsUser: 65534
        runAsGroup: 65534
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: wecom-bridge
          image: repo.swanlab.cn/public/a2w:latest
          imagePullPolicy: IfNotPresent
          securityContext:            # 容器级加固：禁提权 + 弃所有 capabilities + 只读根 FS
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
            readOnlyRootFilesystem: true
          env:
            - name: TZ
              value: Asia/Shanghai          # a2w 用本地时区显示告警时间
          ports:
            - name: http
              containerPort: 5001
          readinessProbe:                    # a2w 无标准健康端点，用 tcpSocket 探端口
            tcpSocket:
              port: 5001
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            tcpSocket:
              port: 5001
            initialDelaySeconds: 20
            periodSeconds: 30
          resources:
            requests:
              cpu: 50m
              memory: 32Mi
            limits:
              cpu: 100m
              memory: 64Mi

---
# ---------- WeCom Bridge Service ----------
apiVersion: v1
kind: Service
metadata:
  name: swanlab-monitor-wecom-bridge
  namespace: <YOUR_NAMESPACE>
  labels:
    app.kubernetes.io/name: wecom-bridge
    app.kubernetes.io/instance: swanlab-monitor
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 5001
      targetPort: http
  selector:
    app.kubernetes.io/name: wecom-bridge
    app.kubernetes.io/instance: swanlab-monitor
```
:::


## 📝 日志采集服务

> 🚧 日志采集（如 `Loki + Promtail`、`ELK` 等方案）的配置指南正在编写中，敬请期待。
> 在此之前，您可以通过 `kubectl logs` 查看各服务 Pod 的运行日志，或通过公有云自带的集群 Pod 日志服务进行观测：


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
