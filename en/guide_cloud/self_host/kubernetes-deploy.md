# Deploying with Kubernetes

> If you need to migrate from the Docker version to the Kubernetes version, please refer to [this document](/en/guide_cloud/self_host/migration-docker-kubernetes.md).

If you want to use [Kubernetes](https://kubernetes.io/) for self-hosted deployment of SwanLab, please follow the installation process below.

![swanlab kubernetes logo](./kubernetes/logo.png)

---

[[toc]]

<br>

---

**Resources and Concepts:**

- [SwanHubX/charts](https://github.com/SwanHubX/charts/tree/main/charts/self-hosted): SwanLab's Kubernetes Helm Chart repository
- `self-hosted`: The deployed SwanLab Kubernetes cluster



## Prerequisites

To deploy the self-hosted version of SwanLab using Kubernetes, please ensure your Kubernetes and related infrastructure meet the following requirements:

| Software/Infrastructure | Version/Configuration Requirement | Necessity Explanation |
| --- | --- | --- |
| kubernetes | v1.24 and above | Official testing and validation cover v1.24+ versions. To ensure API compatibility and system stability, it is not recommended to deploy in clusters with versions lower than this. |
| helm | version>=3 | SwanLab Chart packages are built using the Helm v3 standard and are not compatible with Helm v2 (Tiller mode). |
| RBAC Permissions | Namespace Admin | The deploying account needs to have **write permissions** within the current namespace. Core resources include: `Deployment, StatefulSet, Service, PVC, Secret, ConfigMap`, etc. |
| Network Access (Egress) | *.swanlab.cn | Cluster nodes need to have the ability to access the public internet (or have a configured NAT gateway):<br>1. `repo.swanlab.cn`: Used to pull application images.<br>2. `api.swanlab.cn`: Used for online License activation and validation. |

## 1. Quick Start

You can install the K8S version of the SwanLab self-hosted service via [helm](https://helm.sh/).

First, add the local repository mapping:

```bash
helm repo add swanlab https://helm.swanlab.cn
```

The `swanlab` repository will contain all official open-source Charts from SwanLab. You can install the SwanLab self-hosted service with the following command:

```bash
helm install swanlab-self-hosted swanlab/self-hosted
```

:::warning Tip
You can view all configurable options for `self-hosted` [here](https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml). Meanwhile, we recommend that you save your `values.yaml`.
:::

By installing `swanlab/self-hosted` (referred to below as `self-hosted`), you can install the self-hosted version of the SwanLab application on k8s. The installation results will be printed in the terminal:

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
   https://docs.swanlab.cn/guide_cloud/self_host/kubernetes-deploy.html
```

As shown above, `self-hosted` cannot be directly accessed via an external network by default. You can access this service locally using the `port-forward` functionality. If you wish to **enable external access (via IP or domain name)**, please refer to [3.6 Configuring the Application Access Entrypoint](/en/guide_cloud/self_host/kubernetes-deploy.md#_3-6-configuring-the-application-access-entrypoint).

Here is an example of accessing it locally; open a terminal and execute:

```bash
kubectl port-forward --namespace self-hosted svc/self-hosted 8080:80
```

Then you can access it in your browser at: `http://127.0.0.1:8080` to see the SwanLab page:

![](./docker-deploy/create-account.png)

Now, you need to activate your main account. Activation requires 1 License. For personal use, you can apply for a free one on the [SwanLab official website](https://swanlab.cn) under 「Settings」-「Account & License」.

![](./docker-deploy/apply-license.png)

After obtaining the License, return to the activation page, fill in the username, password, confirm password, and License, then click activate to complete the creation.

![](./docker-deploy/quick-start.png)


## 2. Resource Inventory

To help you better understand the service status of SwanLab, this section will list all deployment resources included in the SwanLab service operation and their corresponding characteristics. `self-hosted` roughly contains two types of resources: basic service resources and application service resources.

### 2.1 Basic Service Resources

Basic service resources refer to necessary resources that the SwanLab application depends on, such as databases, object storage, etc. They include:

1. **PostgreSQL Single Instance**: Stores SwanLab core data.
2. **redis Single Instance**: Stores service cache.
3. **clickhouse Single Instance**: Stores experiment log resources.
4. **minio Single Instance**: Stores media resources.
5. **traefik Single Instance**: Gateway and application entrypoint.

### 2.2 Application Service Resources

Application service resources refer to SwanLab's core business resources — the images for these services will change with `self-hosted` version updates. They include:

1. **SwanLab-Server**: SwanLab core backend service.
2. **SwanLab-House**: SwanLab metric calculation and analysis service.
3. **SwanLab-Cloud**: SwanLab frontend display component.
4. **SwanLab-Next**: SwanLab frontend display component.
5. **Traefik-Proxy**: Gateway component based on Traefik.

Typically, you can freely modify the replica count of these application service resources. All configurable fields can be obtained with the following command:

```bash
helm show values swanlab/self-hosted
```


## 3. Configuring Custom Resources

You can view all configurable items for self-hosted [here](https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml). This section will explain some commonly used and SwanLab-officially-recommended configuration practices.


### 3.1 Customizing Basic Service Resources

As you can see, all basic services deployed by `self-hosted` are single instances. If you are seeking enterprise-level stability, this may not meet your needs. Therefore, `self-hosted` supports linking to external basic service resources — you can configure them via the `integrations` section. Next, we will explain how to use various basic service resources.

We have written detailed comments and key data structure explanations in [values.yaml](https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml). Note that if you enable any integrated basic service resource (e.g., set `integrations.postgres.enabled` to `true`), the single-instance service deployed by `self-hosted` will be destroyed.


#### 3.1.1 Postgres

If you wish to use your own deployed cnpg cluster or a cloud provider's service, you simply need to:

1. Set `integrations.postgres.enabled` to `true`.
2. Create a Secret and pass its name via `integrations.postgres.existingSecret`. The secret should contain the following information:

| `.data.<keys>` | Explanation |
| --- | --- |
| `username` | Read-write username. |
| `password` | Read-write user password. |
| `primaryUrl` | Read-write database connection string, format similar to: `postgresql://{username}:${password}@postgres:5432/app?schema=public`. |
| `replicaUrl` | Read-only database connection string, generally used for load balancing and identical to primaryUrl except for credentials. If a read-only user/cluster is not configured, the read-write connection string can be used instead. |

3. Configure other settings for `integrations.postgres` via `values.yaml`.

Final configuration example:
```yaml
integrations:
  postgres:
    enabled: true
    host: "example.postgres"
    port: 5432
    database: "app"
    existingSecret: integration-postgres
```

> Please ensure the above configuration corresponds with the information in the secret.


#### 3.1.2 Redis

If you wish to use your own deployed redis cluster or a cloud provider's service, you simply need to:
1. Set `integrations.redis.enabled` to `true`.
2. Create a Secret and pass its name via `integrations.redis.existingSecret`. The secret should contain the following information:

| `.data.<keys>` | Explanation |
| --- | --- |
| `url` | Database connection string, format similar to: `redis://{username}:${password}@redis:6379`. |

3. Configure other settings for `integrations.redis` via `values.yaml`.

Final configuration example:

```yaml
integrations:
  redis:
    enabled: true
    host: "example.redis"
    port: 6379
    database: "0"
    existingSecret: integration-redis
```

> Please ensure the above configuration corresponds with the information in the secret.


#### 3.1.3 Clickhouse

If you wish to use your own deployed clickhouse cluster or a cloud provider's service, you simply need to:
1. Set `integrations.clickhouse.enabled` to `true`.
2. Create a Secret and pass its name via `integrations.clickhouse.existingSecret`. The secret should contain the following information:

| `.data.<keys>` | Explanation |
| --- | --- |
| `username` | Read-write username. |
| `password` | Read-write user password. |

3. Configure other settings for `integrations.clickhouse` via `values.yaml`.

Final configuration example:

```yaml
integrations:
  clickhouse:
    enabled: true
    host: "example.clickhouse"
    httpPort: 8123
    tcpPort: 9000
    database: "app"
    existingSecret: integration-clickhouse
```

> Please ensure the above configuration corresponds with the information in the secret.

    
#### 3.1.4 Object Storage

If you wish to use your own deployed minio cluster or a cloud provider's service (must be S3 protocol compatible), you simply need to:
1. Set `integrations.s3.enabled` to `true`.
2. Create a Secret and pass its name via `integrations.s3.existingSecret`. The secret should contain the following information:

| `.data.<keys>` | Explanation |
| --- | --- |
| `accessKey` | Object storage access key. |
| `secretKey` | Object storage secret key. |

3. Configure other settings for `integrations.s3` via `values.yaml`.
Final configuration example:
```yaml
integrations:
  s3:
    enabled: true
    public:
      ssl: true
      endpoint: "xxx.s3.com"
      region: "cn-beijing"
      pathStyle: false
      port: 443
      domain: "https://xxx.xxxx.s3.com"
      bucket: "swanlab-public"
    private:
      ssl: true
      endpoint: "xxx.s3.com"
      region: "cn-beijing"
      pathStyle: false
      port: 443
      bucket: "swanlab-private"
    existingSecret: integration-s3
```

> Please ensure the above configuration corresponds with the information in the secret.

:::warning
- The permission for the publicBucket is **public read, private write**. The permission for the privateBucket is **private read-write**.
- When you choose a custom object storage service, please ensure your object storage service can be accessed directly from outside (via IP or domain name).
- Your object storage secret key must have write permissions and S3 signing permissions for both the **publicBucket** and **privateBucket**.
:::


### 3.2 Customizing Storage Resources

If you wish to use the single-instance basic services deployed by `self-hosted`, we recommend that you declare your own `storage-class` to support data persistence.

Before customizing the storage class for basic resources, please ensure:
1. The `integrations` are not enabled for this basic service resource.
2. Ensure your `storage-class` or `claim` exists in the cluster.


#### 3.2.1 Storage Class for Basic Service Resources

> For the definition of basic service resources, please refer to section 2.1 of this document.

You can configure basic service resources via the `dependencies` section. Taking `postgres` as an example:

1. If you want `self-hosted` to generate a PersistentVolumeClaim, you can configure the `storageClass` and `storageSize` under `dependencies.postgres.persistence` to set the storage class type and size.
2. If you already have a PersistentVolumeClaim, you can configure an existing claim via `dependencies.postgres.persistence.existingClaim`.

Typically, configuring `dependencies.postgres.persistence.existingClaim` is a recommended practice, as it ensures the storage resources are managed by you.


#### 3.2.2 Storage Class for Application Service Resources

> For the definition of application service resources, please refer to section 2.2 of this document.

Due to current technical limitations, `swanlab-house` is deployed as a `StatefulSet`, so you need to mount a PersistentVolume for it. Similar to configuring basic service resources, you need to configure the fields under `service.house.persistence`. Note that `existingClaim` is not allowed here.

:::warning
`swanlab-house` will store some metric intermediate products in the PersistentVolume. Generally, you do not need to care about the data in this volume.
We plan to remove this design in the future.
:::


### 3.3 Increasing Application Replicas to Improve Service Quality

We provide a `replicas` interface for all services under the `service` field. You can freely change their count. Based on SwanLab's operational experience, in most scenarios:
1. The replica count for the `server` service is 3.
2. The replica count for the `house` service is 3.
3. The replica count for the `next` service is 2.
4. The replica count for the `cloud` service is 1.
Of course, application performance is a complex calculation metric, and it also typically depends on resource limits. We also provide interfaces such as `resources` to allow you to configure the resource usage of applications.

### 3.4 Changing Pod Anti-Affinity

You can set Pod anti-affinity to improve disaster recovery capabilities by setting `global.podAntiAffinityPreset`:

```yaml
global:
  # Kubernetes Pod Affinity/Anti-Affinity settings
  # We use Topology Spread Constraints to achieve pod distribution across nodes.
  podAntiAffinityPreset: "soft" # soft, hard, or none
```
The default is `soft`, which means all Pods will be evenly distributed across Nodes. You can set it to `hard` to ensure Pods of the same service are not scheduled on the same Node, or set it to `none` to disable Pod anti-affinity.


### 3.5 Defining Declarations, Labels, and Other Metadata

For any service, we define the following interfaces to facilitate your scheduling of SwanLab application containers:

1. `resources`: Limits the CPU and RAM resources required by the service.
2. `customLabels`: Custom application labels.
3. `customAnnotations`: Custom application annotations.
4. `customTolerations`: Custom tolerations.
5. `customNodeSelector`: Custom node selector.

You can use these resources to freely manage and schedule SwanLab applications.

### 3.6 Configuring the Application Access Entrypoint

The domain name of the application service within the cluster is the release name you deployed. For example, assuming your `cluster domain` is `cluster.local` and the deployment command is:

```bash
helm install swanlab-self-hosted swanlab/self-hosted -n self-hosted
```

- The domain name of the application within the `self-hosted` namespace is `swanlab-self-hosted`.
- The domain name of the application within the `kubernetes` cluster is: `swanlab-self-hosted.self-hosted.svc.cluster.local`

You can write your load balancing strategy based on the above information. It is generally recommended to prioritize using **dedicated domain names (Host-based)** to configure access policies to avoid routing conflicts caused by complex or changing path rules.

**Based on the principle of architectural decoupling**, Self-hosted does not have a built-in Ingress controller. You need to configure the external access entrypoint on the cluster's load balancer (or Ingress), which is also responsible for **TLS termination (HTTPS offloading)**.

**Regarding security policies**, the application trusts all `X-Forwarded-*` request headers by default. If you need stricter header validation or forwarding control, be sure to implement it uniformly at the load balancing layer — this may affect the effectiveness of internal S3 signatures. If you use an external object storage service, you don't need to worry about this.


### 3.7 Changing the Domain Name Displayed in swanlab.login

By default, the **login host** displayed on the `<Your Host>/space/~/quick-start` page automatically uses the domain name `<Your Host>` you are currently using to access the frontend.

If you need to modify this value, you can specify it to your desired domain name by configuring `global.settings.loginHost`. **Please note**, this setting does not affect the actual backend service address; you need to configure the corresponding forwarding rules yourself.

### 3.8 Updating and Rolling Back Services

#### Update

We will update the helm chart version, which will include new features and bug fixes. You can choose to update your deployed service at an appropriate time.

:::warning 
Before updating, please ensure you have backed up the corresponding PVC data!
:::

You can run the following command to update your local repos, which should include the swanlab repo:

```bash
# Update repositories
helm repo update

# List all versions
helm search repo swanlab/self-hosted --versions
```

Then choose the version number you need to complete the update:

```bash
helm upgrade self-hosted swanlab-charts/self-hosted \
  --version x.x.x \
  -f my-values.yaml \
  --namespace xxx
```

#### Rollback

If the service fails to start after an update (e.g., CrashLoopBackOff), immediately roll back to the previous stable version with the following command:

```bash
helm rollback self-hosted x.x.x -n xxx
```

### 3.9 Integrating with Prometheus

SwanLab's application services currently do not support integration with `Prometheus`. This feature is under development, please stay tuned!