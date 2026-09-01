# 常见问题

> 本文档记录了 SwanLab K8s 私有化版本部署过程中的常见问题。

## 【权限要求】部署服务是否需要较高部署权限（如部署 CRD 或 Controller）？

- 不需要。

## 【数据迁移】迁移数据时能否保证原服务不停机？

- ❌ **不能**

迁移过程中必须停机原服务。如果不停止原服务会出现数据 gap。  
此时可考虑使用[swanlab sync](../../api/cli-swanlab-sync.md)将数据上传至新服务。

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
`customNodeSelector` 与 `customTolerations` 为所有服务的通用字段，包括应用服务（`gateway`、`vector`、`service.server`、`service.auth`、`service.house`、`service.cloud`、`service.next`）和基础服务（`dependencies.postgres`、`dependencies.redis`、`dependencies.clickhouse`、`dependencies.s3`），按需为各服务单独配置即可。
:::

## 【慢响应】如何测试集群与外部数据库之间的 RTT？

外接 `PostgreSQL`、`Redis` 或 `ClickHouse` 时，可以在集群中创建临时测试 Pod，测量集群节点与数据库实例之间的 RTT（Round-Trip Time，往返时延）。

**PostgreSQL**：

```bash
# 替换为实际的 PostgreSQL 连接串
kubectl run pg-client --rm -i --tty=false \
  --image=repo.swanlab.cn/public/postgres:16.1 \
  --restart=Never \
  -n <your_namespace> \
  -- sh -c '
export DATABASE_URL="postgres://xxxx:xx@<url>:<port_number>/app"
psql "$DATABASE_URL" -X -qAt <<'"'"'SQL'"'"'
\timing on
select 1;
select 1;
select 1;
select 1;
select 1;
select 1;
select 1;
select 1;
SQL
'
```

**Redis**：

```bash
# 替换为实际的 Redis 连接串
kubectl run redis-rtt --rm -i \
  --image=repo.swanlab.cn/self-hosted/redis-stack:7.4.0-v8 \
  --image-pull-policy=IfNotPresent \
  --restart=Never -n <your_namespace> -- \
  sh -c 'redis-cli -u "redis://<user>:<password>@<redis_host>:6379/0" --latency | awk "{printf \"min: %s ms | max: %s ms | avg: %s ms | samples: %s\n\", \$1, \$2, \$3, \$4}"'
```

**ClickHouse**：

```bash
# 替换为实际的 ClickHouse 用户名和密码
kubectl run ch-rtt --rm -i \
  --image=repo.swanlab.cn/self-hosted/clickhouse-server:24.3 \
  --image-pull-policy=IfNotPresent \
  --restart=Never -n <your_namespace> -- sh -c '
clickhouse-benchmark --concurrency 1 --iterations 1000 \
  --host <clickhouse_host> --port 9000 \
  --user <your_username> --password <your_passwd> \
  --query "SELECT 1" 2>&1 \
| awk "
/QPS:/ { split(\$0, x, \"QPS: \"); split(x[2], y, \",\"); qps=y[1]+0; avg=1000/qps }
/^99\\.000%/ { split(\$0, t, \" \"); p99=t[2]*1000 }
END { printf \"clickhouse RTT: avg=%.3f ms  p99=%.3f ms  (QPS %.2f)\n\", avg, p99, qps }
"
'
```

::: tip
建议集群节点与数据库实例之间的 RTT 在 **0.3ms** 以内，尽量保证 SwanLab 服务所在节点与数据库处于同可用区，详见 [自定义 value 配置](./configuration.md)。
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

**针对使用本地数据库的情况**：

- 部署过程中，每一个 PVC 申请对应一块**独立云SSD硬盘**，支持无感扩容。
- 由云硬盘本身做持久化存储，配置以天为单位的 **快照策略**，TTL 过期时间建议设置 2~7 天，保证每日数据可靠性。

**针对外接云数据库的情况**：

- 可由 IaaS 公有云本身的数据库主从同步进行保障，可联系各公有云厂商的云数据库产品技术支持、或自建集群的 DBA 进行相关对接配置。

## 【对象存储】实验图片上传失败/CSV和日志无法下载/头像显示异常？

此类问题与 `S3对象存储` 配置问题强相关，可以在 `swanlab-house` 对应的 pod 中定位到对应的服务报错日志，推荐排查顺序:

### `value.yaml` 配置校验

- 首先校验一下 `integrations.s3` 中的配置是否正确，详见 [外部 S3 集成配置](./configuration.md#【建议】外部对象存储-s3-integrations-s3)

### 存储桶跨域规则配置

- 以阿里云 OSS 对象存储为例，配置示例为：

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260602112952339.png" width="80%"/>

- **来源**：建议放开到您公司内网域名的最顶级域名，如：您的内网域名为 ：`domain.com`，那么可以将来源设置为 `*.domain.com`
- **允许 Methods**: GET, POST, PUT, HEAD
- **允许 Headers**：填写 \* 通配符
- **返回 Vary:Origin**

### public 桶ACL配置

- SwanLab 的默认用户头像使用「**彩色 SVG**」，如果无法正确显示，一般是 public 桶的公共读被关闭，以 阿里云 OSS 为例，可以在如下设置中开启
- 「**权限控制**」 -> 「**阻止公共访问**」，将按钮关闭
  <img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260602120512627.png" width="80%"/>

- 「**权限控制**」 -> 「**读写权限**」，开启公有读的 Bucket ACL
  <img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260602121636204.png" width="80%"/>
