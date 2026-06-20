# FAQ

> This document records frequently asked questions during the deployment of SwanLab K8s self-hosted version.

## [Permissions] Does deploying the service require elevated permissions (such as deploying CRDs or Controllers)?

- No.

## [Data Migration] Can the original service remain online during data migration?

- ❌ **No**

The original service must be stopped during migration. If the original service is not stopped, data gaps will occur.
In such cases, you may consider using [swanlab sync](/api/cli-swanlab-sync.md) to upload data to the new service.

## [Replica Count] How should the recommended service replica count be configured for the SwanLab self-hosted service?

The following are recommended replica configuration best practices based on online operational experience. You can adjust them by modifying the `replicas` field of the corresponding service in `values.yaml`:

::: warning
In the current `swanlab-self-hosted` deployment, both `postgres` and `clickhouse` use a **single-replica scheme**. Database master-slave replication involves significant architectural changes and is **not currently supported** in this self-hosted deployment version. Please do not adjust the replica count of database-related services.
:::

| Service Name   | Replica Count | Description                                                                             |
| -------------- | ------------- | --------------------------------------------------------------------------------------- |
| clickhouse     | 1             | [Not modifiable] Column database, responsible for experiment metrics storage            |
| postgres       | 1             | [Not modifiable] Relational database, responsible for metadata and relational records   |
| redis          | 1             | [Not modifiable] In-memory database, caching session data                               |
| vector         | 2             | [Not modifiable] ClickHouse metrics write buffer queue                                  |
| traefik        | 2             | [Adjustable] Main gateway, distributes service traffic                                  |
| swanlab-server | ≥ 3           | [Adjustable] SwanLab core service, dynamically adjust based on service load             |
| swanlab-house  | ≥ 3           | [Adjustable] SwanLab metrics analysis service, dynamically adjust based on service load |
| swanlab-next   | 2             | [Adjustable] SwanLab frontend framework                                                 |
| swanlab-cloud  | 1             | [Adjustable] SwanLab frontend experiment page                                           |

## [Node Assignment] How to schedule SwanLab self-hosted service Pods to specific nodes?

In `values.yaml`, all services support specifying a node selector through the `customNodeSelector` field. Kubernetes will only schedule Pods to nodes with the corresponding labels.

**Label a node**:

```bash
kubectl label nodes <node-name> swanlab=true
```

**Example**: Schedule SwanLab Server to nodes with the `swanlab=true` label:

```yaml
service:
  server:
    customNodeSelector: { "swanlab": "true" }
```

The gateway also supports this:

```yaml
gateway:
  customNodeSelector: { "swanlab": "true" }
```

If you need to run on nodes with Taints, you can use it together with `customTolerations`:

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
`customNodeSelector` and `customTolerations` are common fields for all services, including application services (`gateway`, `vector`, `service.server`, `service.house`, `service.cloud`, `service.next`) and base services (`dependencies.postgres`, `dependencies.redis`, `dependencies.clickhouse`, `dependencies.s3`). Configure them individually for each service as needed.
:::

## [Resource Limits] How to limit the CPU and memory usage of SwanLab services?

In `values.yaml`, all application services support setting CPU and memory Requests / Limits through the `resources` field, with the format consistent with Kubernetes native `resources`.

**Example**: Limit resource usage for SwanLab Server:

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

All services can be configured as needed. When not set, there are no limits by default. Base services (`dependencies.postgres`, `dependencies.redis`, `dependencies.clickhouse`, `dependencies.s3`) also support the `resources` field.

## [Images] The cluster cannot access the public internet, how to download and update images?

- You can **pre-download in a public network environment** by manually pulling all required images from the SwanLab public image repository (`repo.swanlab.cn`) using `docker pull`, and then uploading them to your internal private image repository (`docker push`).

## [High Availability] How to ensure service high availability and data security?

Based on database configuration, there are two main scenarios:

「✅ **Recommended**」For local database usage:

- During deployment, each PVC request corresponds to an **independent cloud SSD disk**, supporting seamless expansion.
- The cloud disk itself handles persistent storage. Configure a **snapshot policy** on a daily basis, with a TTL expiration time recommended to be set to 2~7 days, ensuring daily data reliability.

「⚠️ **Not Recommended**」For external cloud database usage:

- This can be ensured by the cloud provider's own database master-slave synchronization. You can contact the cloud database product technical support of each public cloud provider, or the DBA of your self-built cluster for related configuration.

## [Object Storage] Experiment image upload failed / CSV and logs cannot be downloaded / Avatar display abnormal?

These issues are strongly related to `S3 Object Storage` configuration problems. You can locate the corresponding service error logs in the `swanlab-house` pod. Recommended troubleshooting order:

### `values.yaml` Configuration Verification

- First verify whether the configuration in `integrations.s3` is correct. For details, see [External S3 Integration Configuration](/self_host/kubernetes/configuration.md#external-object-storage-s3-integrations-s3)

### Storage Bucket CORS Rule Configuration

- Using Alibaba Cloud OSS object storage as an example, the configuration is:

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260602112952339.png" width="80%"/>

- **Origin**: It is recommended to open it to the top-level domain of your company's internal domain. For example, if your internal domain is `domain.com`, you can set the origin to `*.domain.com`
- **Allowed Methods**: GET, POST, PUT, HEAD
- **Allowed Headers**: Enter \* wildcard
- **Return Vary:Origin**

### Public Bucket ACL Configuration

- SwanLab's default user avatar uses **colorful SVG**. If it cannot be displayed correctly, it is usually because the public read permission of the public bucket has been disabled. Using Alibaba Cloud OSS as an example, you can enable it in the following settings:
- "**Permission Control**" -> "**Block Public Access**", turn off the button
  <img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260602120512627.png" width="80%"/>

- "**Permission Control**" -> "**Read/Write Permissions**", enable public read Bucket ACL
  <img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260602121636204.png" width="80%"/>
