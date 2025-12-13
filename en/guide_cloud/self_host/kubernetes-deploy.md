# Kubernetes Deployment

> If you need to migrate from the Docker version to the Kubernetes version, please refer to [this document](/en/guide_cloud/self_host/migration-docker-kubernetes.md).

If you wish to use [Kubernetes](https://kubernetes.io/) for a self-hosted deployment of SwanLab, please follow the process below for installation.

![swanlab kubernetes logo](./kubernetes/logo.png)

-----

[[toc]]

<br>

-----

**Resources and Concepts:**

  - [SwanHubX/charts](https://github.com/SwanHubX/charts/tree/main/charts/self-hosted): SwanLab's Kubernetes Helm Chart repository
  - `self-hosted`: The deployed SwanLab Kubernetes cluster

## Prerequisites

To deploy the SwanLab private/self-hosted version using Kubernetes, please ensure your Kubernetes and related infrastructure meet the following requirements:

| Software/Infrastructure | Requirement | Explanation |
| --- | --- | --- |
| kubernetes | version >= 1.24 | SwanLab has only been tested on versions above this; lower versions are not guaranteed to work. |
| helm | version >= 3 | SwanLab uses Helm 3 to write SwanLab's Kubernetes packages. |
| NAT Whitelist | Allow cluster access to `swanlab.cn` root domain and subdomains | The cluster needs to pull images via `repo.swanlab.cn`, and the commercial version needs to complete License verification via `api.swanlab.cn`. |

## 1\. Quick Start

You can install the SwanLab self-hosted service K8s version via [helm](https://helm.sh/).

First, add the local repository mapping:

```bash
helm repo add swanlab https://helm.swanlab.cn
```

The `swanlab` repository contains all officially open-sourced Charts by SwanLab. You can use the following command to install the SwanLab self-hosted service:

```bash
helm install swanlab-self-hosted swanlab/self-hosted
```

By installing `swanlab/self-hosted` (hereinafter referred to as `self-hosted`), you can install the SwanLab self-hosted application on K8s.

> You can view all configurable items for self-hosted [here](https://www.google.com/search?q=https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml).

## 2\. Resource List

To help you better understand the status of SwanLab services, this section lists all deployment resources and corresponding characteristics included in the SwanLab service operation—`self-hosted` roughly contains two types of resources: Basic Service Resources and Application Service Resources.

### 2.1 Basic Service Resources

Basic service resources refer to necessary resources depended upon by the SwanLab application, such as databases and object storage. They include:

1.  **PostgreSQL Single Instance**: Stores core SwanLab data.
2.  **Redis Single Instance**: Stores service cache.
3.  **ClickHouse Single Instance**: Stores experiment log resources.
4.  **MinIO Single Instance**: Stores media resources.

### 2.2 Application Service Resources

Application service resources refer to SwanLab's core business resources—the images for these services will change with `self-hosted` version updates—they include:

1.  **Swanlab-Server**: SwanLab core backend service.
2.  **SwanLab-House**: SwanLab metric calculation and analysis service.
3.  **SwanLab-Cloud**: SwanLab frontend display component.
4.  **SwanLab-Next**: SwanLab frontend display component.
5.  **Traefik-Proxy**: Gateway component based on Traefik encapsulation.

Typically, you can modify the replica count of these application service resources at will. All configurable fields can be obtained via the following command:

```bash
helm show values swanlab/self-hosted
```

## 3\. Configuring Custom Resources

You can view all configurable items for self-hosted [here](https://www.google.com/search?q=https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml). In this section, we will explain some common configuration practices recommended by SwanLab.

### 3.1 Customizing Basic Service Resources

As you can see, all basic services deployed by `self-hosted` are single instances. If you are seeking enterprise-level stability, this will not meet your needs. Therefore, `self-hosted` supports attaching external basic service resources—you can configure them via the `integrations` section. Next, we will explain how to use various basic service resources respectively.

We have written detailed comments and secret data structure descriptions in [values.yaml](https://www.google.com/search?q=https://github.com/SwanHubX/charts/blob/main/charts/self-hosted/values.yaml). Note that if you enable any integrated basic service resource (e.g., setting `integrations.postgres.enabled` to `true`), the single-instance service deployed by `self-hosted` will be destroyed.

#### 3.1.1 Postgres

If you wish to use your own deployed Cloud Native PostgreSQL (CNPG) cluster or use cloud provider services, you just need to:

1.  Set `integrations.postgres.enabled` to `true`.
2.  Set a Secret, passing this key name via `integrations.postgres.existingSecret`. The key information includes:

| `.data.<keys>` | Explanation |
| --- | --- |
| `database` | The database name used; we recommend setting it to `app`. |
| `username` | Read-write username. |
| `password` | Read-write user password. |
| `primaryUrl` | Read-write database connection string, format similar to: `postgresql://{username}:${password}@postgres:5432/app?schema=public` |
| `replicaUrl` | Read-only database connection string, generally used for load balancing and is identical to `primaryUrl` except for account credentials. If no read-only user/cluster is configured, the read-write connection string can be used instead. |

#### 3.1.2 Redis

If you wish to use your own deployed Redis cluster or use cloud provider services, you just need to:

1.  Set `integrations.redis.enabled` to `true`.
2.  Set a Secret, passing this key name via `integrations.redis.existingSecret`. The key information includes:

| `.data.<keys>` | Explanation |
| --- | --- |
| `url` | Database connection string, format similar to: `redis://{username}:${password}@redis:6379` |

#### 3.1.3 ClickHouse

If you wish to use your own deployed ClickHouse cluster or use cloud provider services, you just need to:

1.  Set `integrations.clickhouse.enabled` to `true`.
2.  Set a Secret, passing this key name via `integrations.clickhouse.existingSecret`. The key information includes:

| `.data.<keys>` | Explanation |
| --- | --- |
| `database` | The database name used; we recommend setting it to `app`. |
| `username` | Read-write username. |
| `password` | Read-write user password. |
| `host` | ClickHouse service address. |
| `httpPort` | ClickHouse HTTP service port, usually 8123. |
| `tcpPort` | ClickHouse TCP service port, usually 9000. |

#### 3.1.4 Object Storage

If you wish to use your own deployed MinIO cluster or use cloud provider services, you just need to:

1.  Set `integrations.s3.enabled` to `true`.
2.  Set a Secret, passing this key name via `integrations.s3.existingSecret`. The key information includes:

| `.data.<keys>` | Explanation |
| --- | --- |
| `accessKey` | Object storage access key. |
| `secretKey` | Object storage secret key. |
| `endpoint` | Object storage address. |
| `privateBucket` | Private bucket name; we recommend setting it to `swanlab-private`. |
| `publicBucket` | Public bucket name; we recommend setting it to `swanlab-public`. |
| `region` | Object storage region. If you use self-deployed MinIO or similar services, this field may not exist; set it to `local`. |

3.  Ensure that the `privateBucket` and `publicBucket` configured in the secret already exist in the object storage service.

:::warning
If you plan to configure your own MinIO or other object storage services in the cluster, you should ensure this service is accessible from the public network—because the SwanLab frontend service also needs to access this service, and `self-hosted` does not configure load balancing strategies for third-party services by default.
:::

### 3.2 Customizing Storage Resources

If you wish to use the single-instance basic services deployed by `self-hosted`, we suggest declaring a `storage-class` yourself to support data persistence, because `self-hosted` uses `local-storage` to declare PVCs by default.

Before configuring custom storage classes for basic resources, please ensure:

1.  The basic service resource does not have `integrations` enabled.
2.  Ensure your `storage-class` or `claim` exists in the cluster.

#### 3.2.1 Storage Class for Basic Service Resources

> For the definition of basic service resources, please see section 2.1 of this document.

You can configure basic service resources via the `dependencies` section. Taking Postgres as an example:

1.  If you want `self-hosted` to generate storage volumes, you can configure the storage volume type and size via `storageClass` and `storageSize` under `dependencies.postgres.persistence`.
2.  If you already have a storage volume, you can configure an existing volume via `dependencies.postgres.persistence.existingClaim`.

Generally, configuring `dependencies.postgres.persistence.existingClaim` is a recommended practice, as this ensures storage resources are managed by you.

#### 3.2.2 Storage Class for Application Service Resources

> For the definition of application service resources, please see section 2.2 of this document.

Due to current technical limitations, `swanlab-house` is deployed as a StatefulSet, so you need to mount a storage volume for it. Similar to configuring basic service resources, you need to configure fields under `service.house.persistence`. Note that configuring `existingClaim` is **not** allowed here.

:::warning
`swanlab-house` will store some metric intermediates under the storage volume; generally, you do not need to care about the data in this volume.
We will remove this design in the future.
:::

### 3.3 Increasing Application Replicas to Improve Service Quality

We provide the `replicas` interface for all services under the `service` field. You can modify their quantities freely. Based on SwanLab's operational experience, in most scenarios:

1.  The replica count for the `server` service is 3.
2.  The replica count for the `house` service is 3.
3.  The replica count for the `next` service is 2.
4.  The replica count for the `cloud` service is 1.

Of course, application performance is a complex calculation metric. Usually, it also depends on resource limits; we also provide interfaces like `resources` to allow you to configure application resource usage.

### 3.4 Defining Declarations, Labels, and Other Metadata

For any service, we have defined the following interfaces to facilitate your scheduling of SwanLab application containers:

1.  `customLabels`: Custom application labels
2.  `customAnnotations`: Custom application annotations
3.  `customTolerations`: Custom tolerations
4.  `customNodeSelector`: Custom node selectors

You can manage and schedule SwanLab applications freely using these resources.

### 3.5 Configuring Custom Load Balancing, Domains, and TLS Services

`self-hosted` itself does not provide ingress; within K8s, you need to use external load balancing for access. First, ensure you have changed the application service type to NodePort or ClusterIP. Furthermore, to avoid unnecessary accidents, we usually recommend performing TLS termination at the load balancer side and requiring the load balancer to pass `X-Forwarded-*` related request headers.

Finally, besides configuring traffic forwarding rules on your own load balancer, please configure Traefik to trust the `X-Forwarded-*` request headers set by upstream services. Refer to [this document](https://doc.traefik.io/traefik/reference/install-configuration/entrypoints/#opt-forwardedHeaders-trustedIPs):

```yaml
ingress:
  traefik:
    ports:
      web:
        forwardedHeaders:
          trustedIPs: [] # Set your upstream load balancer's internal IP here
```

### 3.6 Changing the Domain Displayed in swanlab.login

By default, the login host displayed on the `<Your Host>/space/~/quick-start` page automatically uses the current frontend domain `<Your Host>` you are accessing.

If you need to modify this value, you can specify your desired domain by configuring `env.apiHost`.

Note that this configuration is merely a display change and will not affect actual routing forwarding rules. Additionally, this configuration conflicts with `ingress.host`; the latter will configure strict domain forwarding rules, causing clients to be unable to access `env.apiHost`. In this case, we suggest deploying a load balancer layer above the `self-hosted` service to take over traffic forwarding rules and implement TLS termination. See "Changing Application Service Types" for details.

### 3.7 Integrating Prometheus

SwanLab application services do not currently support `Prometheus` integration. This feature is under development, stay tuned\!