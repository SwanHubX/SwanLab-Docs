# Update & Rollback

> This page introduces the version update and rollback procedures for SwanLab Kubernetes deployment.

[[toc]]

## 1. Pre-Update Preparation

### 1.1 Understand Version Information

Before updating, it is recommended to first check the current deployment version information and available update versions.

View current release and history versions:

```bash
helm list -n <your_namespace>
helm history swanlab-self-hosted -n <your_namespace>
```

Where:

- `swanlab-self-hosted` is the default release name, which can be **adjusted to the release name in your cluster as needed**;
- `REVISION` is the **rollback index**, which can be used to initiate a rollback at any time if the update fails or there are other compatibility issues.

### 1.2 Sync Remote Repository

```bash
# Add Helm repository (skip if already added)
helm repo add swanlab https://helm.swanlab.cn

# Update repository index
helm repo update

# List all available versions
helm search repo swanlab/self-hosted --versions
```

### 1.3 Pre-Update Checklist

:::warning
Before updating, please ensure you have completed the following:
:::

1. **PVC and Snapshot Policy Configured**: Please confirm that PVCs for storage resource services have been successfully created, and that corresponding snapshot policies have been configured to ensure data security.
2. **Confirm Image Repository Accessibility**: You need to ensure the cluster can access `repo.swanlab.cn` to pull images normally (otherwise you need to pull the known images and push them to a private repository). For details, see [SwanLab Self-Hosted Resource Inventory](/self_host/kubernetes/deploy#🧾-resource-inventory)
3. **Check Image Tag Configuration**: In your `values.yaml`, ensure the image **tags for the following applications are empty strings or specified version tags**, not `latest`:
   - `swanlab-cloud`
   - `swanlab-next`
   - `swanlab-house`
   - `swanlab-server`

```yaml
# values.yaml example
image:
  repository: repo.swanlab.cn/self-hosted/swanlab-cloud
  # tag set to empty string or version tag, e.g., v2.8.0, do not set to latest
  tag: ""
  pullPolicy: "IfNotPresent"
```

## 2. Execute Update

You can choose one of the following update methods based on your cluster's network environment.

### Option 1: Helm Repository Update

If your **cluster nodes can directly access the Helm repository** (i.e., cluster nodes can directly access `github.com`), you can execute the update with the following commands:

> ⚠️ Note: The chart packages at https://helm.swanlab.cn are indexed by version tags in [GitHub Release](https://github.com/SwanHubX/charts/releases). **Please confirm network connectivity in advance!**

```bash
# It is recommended to use --dry-run first to verify template compatibility
helm upgrade swanlab-self-hosted swanlab/self-hosted \
  --version <target_version> \
  -f <your_own_values.yaml> \
  --namespace <your_namespace> \
  --dry-run
```

- After confirming there are no errors, remove the `--dry-run` option to execute the update

### Option 2: Local Chart Package Update

If your **cluster nodes cannot directly access the Helm repository** (i.e., cluster nodes cannot directly access `github.com`), you can pull the chart package to local via OCI method and then execute the update:

> If you encounter a 401 authentication failure, you can clear any existing helm login state on your machine with `helm registry logout xxx.com`.

```bash
# Pull chart package to local
helm pull oci://swanlab-registry.cn-hangzhou.cr.aliyuncs.com/chart/self-hosted --version <target_version>
# Extract chart package, expecting only one self-hosted/ folder
tar -zxvf self-hosted-<target_version>.tgz
```

Then use the local chart package to verify:

```bash
# It is recommended to use --dry-run first to verify template compatibility
helm upgrade swanlab-self-hosted ./self-hosted/ \
  -f <your_own_values.yaml> \
  --namespace <your_namespace> \
  --dry-run
```

- After confirming there are no errors, remove the `--dry-run` option to execute the update

> Note: `--dry-run` is used to verify the compatibility of the update template. It is recommended to do template syntax verification before each update.

## 3. Update Verification

After the update is complete, please follow the steps below to verify that the service is running normally.

### 3.1 Release Version Verification

Confirm the release version has been updated:

```bash
helm list -n <your_namespace>
```

Ensure that the Service and Pod in the cluster are in normal status, and the status is `deployed`.

### 3.2 Pod Health Status Check

Ensure all Pods are running normally:

```bash
kubectl get pods -n <your_namespace>
```

All Pods should be in `Running` or `Completed` status, and there should be no abnormal statuses such as `CrashLoopBackOff` or `Error`.

### 3.3 Metrics Reporting Test

- **Page Access**: Confirm that the frontend page can be accessed normally and metrics can be downloaded normally
- **Python SDK**: Confirm that the SDK connection is normal and experiments can be uploaded normally

```python
import swanlab
import random
import numpy as np
import time

swanlab.login(
    api_key="xxxxx", # Valid api_key under your private swanlab service
    host="xxxxxx"  # Your private swanlab service domain
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

## 4. Rollback

If the service fails to start after an update (e.g., `CrashLoopBackOff`), or there are other compatibility issues, please immediately rollback with the following command:

### 4.1 Rollback to Previous Version

```bash
helm rollback swanlab-self-hosted -n <your_namespace>
```

### 4.2 Rollback to Specified Version

First view the history versions to determine the `REVISION` number to rollback to:

```bash
helm history swanlab-self-hosted -n <your_namespace>
```

Then execute the rollback:

```bash
helm rollback swanlab-self-hosted <revision_number> -n <your_namespace>
```

After the rollback is complete, please refer to the [Update Verification](#_3-update-verification) steps to confirm the service status again.
