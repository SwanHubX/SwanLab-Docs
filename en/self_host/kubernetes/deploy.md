# Deploying with Kubernetes

> If you need to migrate from the Docker version to the Kubernetes version, please refer to [this document](../docker/migration-docker-kubernetes.md).
> The SwanLab Python SDK version supported by the Kubernetes version is >= 0.7.4

If you want to use [Kubernetes](https://kubernetes.io/) for self-hosted deployment of SwanLab, please follow the installation process below.

![swanlab kubernetes logo](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/zh/guide_cloud/self_host/kubernetes/logo.png)

[[toc]]

<br>

**Resources and Concepts:**

- [SwanHubX/charts - Self-Hosted Service Releases](https://github.com/SwanHubX/charts/releases): SwanLab's Kubernetes Helm Chart repository
- `swanlab-self-hosted`: The **default RELEASE name** for the SwanLab self-hosted service deployed in the cluster
- `<your_namespace>`: The namespace for the SwanLab self-hosted service deployed in the cluster, please replace with the actual namespace used for deployment

::: info
**Current APP_VERSION: v2.9.0**
:::

## 🧱 Prerequisites

To deploy the self-hosted version of SwanLab using Kubernetes, please ensure your Kubernetes and related infrastructure meet the following requirements:

| Software/Infrastructure | Version/Configuration Requirement | Necessity Explanation                                                                                                                                                                                                                          |
| ----------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| kubernetes              | v1.24 and above                   | Official testing and validation cover v1.24+ versions. To ensure API compatibility and system stability, it is not recommended to deploy in clusters with versions lower than this.                                                            |
| helm                    | version>=3.9                      | SwanLab Chart packages require features from Helm v3.9 or newer, and are not compatible with earlier versions or Helm v2 (Tiller mode).                                                                                                        |
| RBAC Permissions        | Namespace Admin                   | The deploying account needs to have **write permissions** within the namespace for the SwanLab self-hosted service. Core resources include: `Deployment, StatefulSet, Service, PVC, Secret, ConfigMap`, etc.                                   |
| Network Access (Egress) | \*.swanlab.cn                     | Cluster nodes need to have the ability to access the public internet (or have a configured NAT gateway):<br>1. `repo.swanlab.cn`: Used to pull application images. <br>2. `api.swanlab.cn`: Used for online License activation and validation. |
| Object Storage          | AWS S3 protocol compatible        | Media resources and other files reported by SwanLab are stored in object storage by default. To save storage costs, **external object storage integration** is recommended, ensuring S3 API compatibility                                      |

## 🧾 Resource Inventory

### Application Service Images

> ⚠️ Note: The `values.yaml` image tags **are set to empty strings by default**, which automatically syncs the latest version number from the template as the image tag. Generally, no modification is needed. Special hotfix patch version images require manual tag filling.

| Component      | Image Address                                              | values.yaml Config Path | Description                                   |
| -------------- | ---------------------------------------------------------- | ----------------------- | --------------------------------------------- |
| swanlab-server | `repo.swanlab.cn/self-hosted/swanlab-server:<APP_VERSION>` | `service.server.image`  | Backend core service                          |
| swanlab-auth   | `repo.swanlab.cn/self-hosted/swanlab-auth:<APP_VERSION>`   | `service.auth.image`    | Authentication and authorization service      |
| swanlab-house  | `repo.swanlab.cn/self-hosted/swanlab-house:<APP_VERSION>`  | `service.house.image`   | Backend experiment metrics OLAP service       |
| swanlab-cloud  | `repo.swanlab.cn/self-hosted/swanlab-cloud:<APP_VERSION>`  | `service.cloud.image`   | Frontend experiment chart rendering component |
| swanlab-next   | `repo.swanlab.cn/self-hosted/swanlab-next:<APP_VERSION>`   | `service.next.image`    | Frontend UI                                   |

### Infrastructure Images

> ⚠️Note: When a storage component chooses to [customize base service resources](./deploy.md#_3-1-customizing-base-service-resources), the corresponding images below can be ignored (using self-built external services).

::: warning
The **database of the SwanLab self-hosted service uses a single-instance mode**, and there will be architectural changes in the future. **To ensure consistency between architecture and testing behavior**, except for **S3 object storage**, we **do not recommend using cloud databases** for integration. We recommend using **cloud SSD disks** as the storageClass for the corresponding base service PVC storage resources.
:::

| Component  | Image Address                                                          | values.yaml Config Path         | Description                                                                      |
| ---------- | ---------------------------------------------------------------------- | ------------------------------- | -------------------------------------------------------------------------------- |
| Traefik    | `repo.swanlab.cn/public/traefik:3.6`                                   | `service.gateway.`              | Reverse proxy / gateway entry                                                    |
| Identify   | `repo.swanlab.cn/public/swanlab-helper/identify:v1.2`                  | `service.gateway.identifyImage` | Gateway authentication auxiliary image                                           |
| Busybox    | `repo.swanlab.cn/public/busybox:1.37.0`                                | `helper.image`                  | Deployment auxiliary init container                                              |
| Vector     | `repo.swanlab.cn/public/vector:0.51.1-debian`                          | `vector.image`                  | Experiment metrics collection buffer queue                                       |
| PostgreSQL | `repo.swanlab.cn/self-hosted/postgres:16.1`                            | `dependencies.postgres.image`   | PostgreSQL relational database (users, projects, experiment metadata)            |
| Redis      | `repo.swanlab.cn/self-hosted/redis-stack:7.4.0-v8`                     | `dependencies.redis.image`      | Cache and session storage                                                        |
| ClickHouse | `repo.swanlab.cn/self-hosted/clickhouse-server:24.3`                   | `dependencies.clickhouse.image` | Experiment metrics and logs column database                                      |
| MinIO      | `repo.swanlab.cn/self-hosted/minio/minio:RELEASE.2025-09-07T16-13-09Z` | `dependencies.s3.image`         | S3 compatible object storage (experiment media resources and exported log files) |
| MinIO MC   | `repo.swanlab.cn/self-hosted/minio/mc:RELEASE.2025-08-13T08-35-41Z`    | `dependencies.s3.mcImage`       | MinIO client tool (bucket initialization, etc.)                                  |

## 🪜 Installation Guide

::: info
This guide follows the best practices for SwanLab K8s self-hosted service installation. The recommended approach is **[External S3 Object Storage Integration] + [Cluster Databases via PVC-mounted Cloud Disks]**. If you have special integration requirements, please refer to [Custom Value Configuration](./configuration.md) for modifications.
:::

### 1. Create S3 Secret

It is recommended to use an existing online object storage service (must be compatible with AWS S3 protocol). Please ensure you have created **AK/SK with write permissions**, refer to the following yaml:
::: details swanlab-self-hosted-secret.yaml Template

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: swanlab-self-hosted-secret-s3
  namespace: <your_namespace> # Please replace with the actual namespace
type: Opaque
stringData:
  accessKey: xxxx
  secretKey: xxxx
```

:::

```bash
# Create secret in the cluster
kubectl apply -f swanlab-self-hosted-secret.yaml

# View secret
kubectl get secret swanlab-self-hosted-secret-s3 -n <your_namespace>
```

### 2. Create PVC

The SwanLab self-hosted service mainly depends on the following base service storage resources:

| Base Resource         | PVC Name                                                                  | Recommended Storage Size                            |
| --------------------- | ------------------------------------------------------------------------- | --------------------------------------------------- |
| Redis                 | `swanlab-redis-pvc`                                                       | ≥ 50G                                               |
| PostgreSQL            | `swanlab-postgres-pvc`                                                    | ≥ 100G                                              |
| ClickHouse            | `swanlab-clickhouse-pvc`                                                  | ≥ 1000G                                             |
| Vector                | `data-swanlab-self-hosted-vector-0` / `data-swanlab-self-hosted-vector-1` | Each replica ≥ 60G, two independent storage volumes |
| 「**Optional**」MinIO | `swanlab-minio-pvc`                                                       | ≥ 1000G, online object storage is preferred         |

::: warning

- `storageClassName` should be based on **the cloud disk type mounted in your cluster** (e.g., Tencent Cloud's default cloud disk `cbs`), requiring **support for dynamic expansion and snapshot policies**
- Vector is deployed as a `StatefulSet`, PVC names are **not modifiable** by default
- Ensure that all PVCs related to `postgres/redis/clickhouse` are in **Bound** status before proceeding with subsequent installation steps
  :::

::: details swanlab-self-hosted-pvc.yaml Template

```yaml
# ----------------------------
# PersistentVolumeClaim for Redis
# ----------------------------
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: swanlab-redis-pvc
  namespace: <your_namespace> # Replace with actual namespace
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: "" # Replace with the actual storageClass in your cluster
  volumeMode: Filesystem
---
# ----------------------------
# PersistentVolumeClaim for PostgreSQL
# ----------------------------
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: swanlab-postgres-pvc
  namespace: <your_namespace> # Replace with actual namespace
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: "" # Replace with the actual storageClass in your cluster
  volumeMode: Filesystem
---
# ----------------------------
# PersistentVolumeClaim for ClickHouse
# ----------------------------
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: swanlab-clickhouse-pvc
  namespace: <your_namespace> # Replace with actual namespace
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1000Gi
  storageClassName: "" # Replace with the actual storageClass in your cluster
  volumeMode: Filesystem
---
# ----------------------------
# PersistentVolumeClaim for Vector-0
# ----------------------------
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-swanlab-self-hosted-vector-0
  namespace: <your_namespace> # Replace with actual namespace
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 60Gi
  storageClassName: "" # Replace with the actual storageClass in your cluster
  volumeMode: Filesystem
---
# ----------------------------
# PersistentVolumeClaim for Vector-1
# ----------------------------
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-swanlab-self-hosted-vector-1
  namespace: <your_namespace> # Replace with actual namespace
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 60Gi
  storageClassName: "" # Replace with the actual storageClass in your cluster
  volumeMode: Filesystem
```

:::

```bash
# Create PVC
kubectl apply -f swanlab-self-hosted-pvc.yaml

# Verify PVC status (ensure all except vector are Bound)
kubectl get pvc -n <your_namespace>
```

### 3. Fill in Values

For the complete `values.yaml` configuration options, please refer to the [Custom Value Configuration](./configuration.md) documentation.

You can view the original value template used by `swanlab-self-hosted` at [values.yaml template](https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml), or view it with the following command:

```bash
helm show values swanlab/self-hosted
```

We **strongly recommend saving the `values.yaml` used for deployment to ensure smooth upgrades later**. Below are the related `values.yaml` modification instructions:

#### 3.1 PVC Binding

The `existingClaim` of each component under the `dependencies` field must match the PVC names created in Step 2:

| Component  | values.yaml Path                                    | Corresponding PVC Name   |
| ---------- | --------------------------------------------------- | ------------------------ |
| PostgreSQL | `dependencies.postgres.persistence.existingClaim`   | `swanlab-postgres-pvc`   |
| Redis      | `dependencies.redis.persistence.existingClaim`      | `swanlab-redis-pvc`      |
| ClickHouse | `dependencies.clickhouse.persistence.existingClaim` | `swanlab-clickhouse-pvc` |

Modification example:

::: details dependencies PVC Modification Example

::: code-group

```yaml [PostgreSQL]
dependencies:
  postgres: # PostgreSQL database configuration
    fullnameOverride: ""

    image: # Image configuration
      # Full image repository path (e.g., ghcr.io/cloudnative-pg/postgresql)
      repository: repo.swanlab.cn/self-hosted/postgres
      # Image tag/version. SwanLab recommends 16.x and above
      tag: "16.1"
      # Kubernetes image pull policy (Always, IfNotPresent, Never)
      pullPolicy: "IfNotPresent"

    username: ""
    password: ""

    persistence: # Persistent storage configuration
      # Kubernetes StorageClass for dynamic provisioning
      storageClass: "disk-cloud-auto"
      # Database storage volume size
      storageSize: "10Gi"

      # Use an existing PVC (leave empty to auto-create a new PVC)
      existingClaim: "swanlab-postgres-pvc" # ⚠️: Usually only this name needs to be modified to match the created postgres PVC name

    customLabels: {}
    customAnnotations: {}
    customPodLabels: {}
    customPodAnnotations: {}
    customTolerations: []
    customNodeSelector: {}
    resources: {}
```

```yaml [Redis]
redis: # Redis configuration
  fullnameOverride: ""

  image: # Image configuration
    # Full image repository path (e.g., redis/redis-stack)
    repository: repo.swanlab.cn/self-hosted/redis-stack
    # Image tag/version
    tag: "7.4.0-v8"
    # Kubernetes image pull policy (Always, IfNotPresent, Never)
    pullPolicy: "IfNotPresent"

  persistence: # Persistent storage configuration
    storageClass: "disk-cloud-auto"
    storageSize: "10Gi"

    # Use an existing PVC (leave empty to auto-create a new PVC)
    existingClaim: "swanlab-redis-pvc" # ⚠️: Usually only this name needs to be modified to match the created redis PVC name

  customLabels: {}
  customAnnotations: {}
  customPodLabels: {}
  customPodAnnotations: {}
  customTolerations: []
  customNodeSelector: {}
  resources: {}
```

```yaml [ClickHouse]
clickhouse: # ClickHouse configuration
  fullnameOverride: ""

  image: # Image configuration
    # Full image repository path (e.g., clickhouse/clickhouse-server)
    repository: repo.swanlab.cn/self-hosted/clickhouse-server
    # Image tag/version
    tag: "24.3"
    pullPolicy: "IfNotPresent"

  # Authentication credentials
  # Leave empty if using existingSecret
  username: ""
  password: ""

  persistence: # Persistent storage configuration
    storageClass: "disk-cloud-auto"
    storageSize: "20Gi"

    # Use an existing PVC (leave empty to auto-create a new PVC)
    existingClaim: "swanlab-clickhouse-pvc" # ⚠️: Usually only this name needs to be modified to match the created clickhouse PVC name

  customLabels: {}
  customAnnotations: {}
  customPodLabels: {}
  customPodAnnotations: {}
  customTolerations: []
  customNodeSelector: {}
  resources: {}
```

:::

#### 3.2 Application Image Tags

The `tag` of the five application images under `service` (server / auth / house / cloud / next) should be set to **empty strings**, and the Chart will automatically inject the correct version number during rendering.

#### 3.3 Vector Storage

`vector.persistence.storageClass` and `vector.persistence.storageSize` should be consistent with the size when creating the vector PVC. The default size needs to be changed to `60Gi`.

#### 3.4 External S3 Integration Configuration

The `integrations.s3` field needs to be manually filled based on the object storage service you use. It is recommended to separate public and private buckets. If your cloud provider distinguishes S3 protocol endpoint access, please pay special attention to filling in the S3 endpoint. For detailed field descriptions and configuration examples, please refer to the [External S3 Integration](./configuration.md#external-object-storage-s3-integrations-s3) section.

### 4. Add Helm Repository

You can install the SwanLab self-hosted K8S service via [helm](https://helm.sh/).

First, set up local repository mapping:

```bash
helm repo add swanlab https://helm.swanlab.cn
# Update repository
helm repo update
# List all chart versions
helm search repo swanlab/self-hosted --versions

```

### 5. Execute Installation

You can choose one of the following installation methods based on your cluster's network environment.

#### Option 1: Helm Repository Installation

If your **cluster nodes can directly access the Helm repository** (i.e., cluster nodes can directly access `github.com`), you can execute the installation with the following commands:

> ⚠️ Note: The chart packages at https://helm.swanlab.cn are indexed by version tags in [GitHub Release](https://github.com/SwanHubX/charts/releases). **Please confirm network connectivity in advance!**

```bash
# It is recommended to use --dry-run first to verify template compatibility
helm install swanlab-self-hosted swanlab/self-hosted \
  -f <your_own_values.yaml> \
  --namespace <your_namespace> \
  --dry-run
```

- After confirming there are no errors, remove the `--dry-run` option to execute the installation

#### Option 2: Local Chart Package Installation

If your **cluster nodes cannot directly access the Helm repository** (i.e., cluster nodes cannot directly access `github.com`), you can pull the chart package to local via OCI method and then execute the installation:

> If you encounter a 401 authentication failure, you can clear any existing helm login state on your machine with `helm registry logout xxx.com`.

```bash
# Pull chart package to local
helm pull oci://swanlab-registry.cn-hangzhou.cr.aliyuncs.com/chart/self-hosted --version <latest_chart_version>
# Extract chart package, expecting only one self-hosted/ folder
tar -zxvf self-hosted-<latest_chart_version>.tgz
```

Then use the local chart package to verify:

```bash
# It is recommended to use --dry-run first to verify template compatibility
helm install swanlab-self-hosted ./self-hosted/ \
  -f <your_own_values.yaml> \
  --namespace <your_namespace> \
  --dry-run
```

- After confirming there are no errors, remove the `--dry-run` option to execute the installation

By installing `swanlab/self-hosted`, you can install the SwanLab self-hosted application on k8s. The installation result will print similar information in the terminal:

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
   https://docs.swanlab.cn/en/self_host/kubernetes/deploy.html
```

As shown above, the `swanlab-self-hosted` self-hosted service cannot be directly accessed via external network by default. You can access this service locally using the `port-forward` functionality.
If you wish to **enable external access (via IP or domain name)**, please refer to [Configuring Application Access Entrypoint](./configuration.md#configuring-application-access-entrypoint).

Here is an example of accessing it locally; open a terminal and execute:

```bash
kubectl port-forward --namespace self-hosted svc/swanlab-self-hosted 8080:80
```

Then you can access it in your browser at: `http://127.0.0.1:8080` to see the SwanLab page:

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/zh/guide_cloud/self_host/docker-deploy/create-account.png)

## 📖 License Activation

### Personal Edition Activation

Now, you need to activate your main account. Activation requires 1 License. For personal use, you can apply for a free one on the [SwanLab official website](https://swanlab.cn) under "Settings" - "Account & License".

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/zh/guide_cloud/self_host/docker-deploy/apply-license.png)

After obtaining the License, return to the activation page, fill in the username, password, confirm password, and License, then click activate to complete the creation.

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/zh/guide_cloud/self_host/docker-deploy/quick-start.png)

### Enterprise Edition Activation

If you need to test enterprise edition capabilities, please contact [contact@swanlab.cn](mailto:contact@swanlab.cn), and we will send a test License to that email.

## ⚙️ Verification Testing

After deployment is complete, you can use the following Python code to verify scalar and media metrics reporting

::: details Metrics Reporting Test

```python
import swanlab
import random
import numpy as np  # Add NumPy import

swanlab.login(
    api_key="xxxxx",
    host="https://xxxxx"
)

# Create a SwanLab project
swanlab.init(
    # Set project name
    project="my-first-project",
    experiment_name="my-first-experiment",
    # Set hyperparameters
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10
    }
)

# Simulate a training session
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    swanlab.log({
        "step_time": acc,
        "speed": loss
    })

# Generate random noise image (64x64 RGB, random pixel values)
random_noise = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
img = swanlab.Image(random_noise, caption="Random Noise")
swanlab.log({
    "image": img
})

# [Optional] Finish training, this is necessary in notebook environments
swanlab.finish()

```

:::

Expected result is shown below⬇️：

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260603151219153.png"  style="width:60%"/>

You can verify the following features:

- ⬜ Check if metrics can be reported normally
- ⬜ Check if images can be uploaded and displayed normally
- ⬜ Check if metric CSV downloads work smoothly
- ⬜ Check if user avatars can be displayed normally

## 🧱 Additional Notes

You can view all configurable options for `swanlab-self-hosted` [here](https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml).

For detailed field descriptions and configuration practices, please refer to the [Custom Value Configuration](./configuration.md) documentation, which covers:

- **Global Configuration**: Pod anti-affinity, login domain, etc.
- **Application Services**: Replica count, resource limits, labels and annotations, etc.
- **Gateway Configuration**: Application access entrypoint, ports, etc.
- **Built-in Base Services**: PostgreSQL / Redis / ClickHouse / MinIO storage resource configuration
- **External Service Integration**: Connecting external PostgreSQL, Redis, ClickHouse, S3 object storage

### Updates and Rollback

To update the SwanLab version or rollback after a failed update, please refer to the [Update & Rollback](./upgrade.md) documentation.

### Prometheus Observability Integration Guide

Please refer to the [Monitor & Logging](./monitor-logging.md) documentation.
