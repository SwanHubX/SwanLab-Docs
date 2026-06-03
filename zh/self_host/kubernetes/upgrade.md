# 更新与回滚

> 本页介绍 SwanLab Kubernetes 部署的版本更新与回滚流程。

[[toc]]

---

## 1. 更新前准备

### 1.1 了解版本信息

更新前，建议先查看当前部署的版本信息和可用的更新版本。

查看当前 release 和历史版本：

```bash
helm list -n <your_namespace>
helm history swanlab-self-hosted -n <your_namespace>
```

其中：

- `swanlab-self-hosted` 为默认的 release 名称，可以**按需调整为您集群中的 release 名**；
- `REVISION` 为**回滚索引**，在更新失败或存在其他适配性问题时，该数字可用于随时发起回滚。

### 1.2 同步远程仓库

```bash
# 添加 Helm 仓库（如已添加可跳过）
helm repo add swanlab https://helm.swanlab.cn

# 更新仓库索引
helm repo update

# 列出所有可用版本
helm search repo swanlab/self-hosted --versions
```

### 1.3 更新前检查清单

:::warning
在更新前，请您确保已完成以下事项：
:::

1. **已配置 PVC 及快照策略**：请确认用于存储资源服务的 PVC 已成功创建，并已配置好相应的快照策略以保障数据安全。
2. **确认镜像仓库可达**：您需要保证集群可以访问 `repo.swanlab.cn`，以保证能够正常拉取镜像 （否则您需要将已知的镜像拉取下来推送到私有仓库）。详情见 [SwanLab 私有化版资源清单](/self_host/kubernetes/deploy#🧾-资源清单)
3. **检查镜像 tag 配置**：在您的 `values.yaml` 中，确保以下应用的镜像 **tag 为空字符串或指定版本 tag**，而非 `latest`：
   - `swanlab-cloud`
   - `swanlab-next`
   - `swanlab-house`
   - `swanlab-server`

```yaml
# values.yaml 示例
    image:
      repository: repo.swanlab.cn/self-hosted/swanlab-cloud
      # tag 置为空字符串 或者版本 tag，如 v2.8.0，不要设置为 latest
      tag: ""
      pullPolicy: "IfNotPresent"
```

## 2. 执行更新

您可以根据集群网络环境选择以下两种更新方式之一。

### 选项一：Helm 仓库更新

如果您的**集群节点可以直接访问 Helm 仓库**（即集群节点可以直接访问 `github.com`），可以参考如下命令执行更新：
> ⚠️ 注意：https://helm.swanlab.cn 的 chart 包在 [GitHub Release](https://github.com/SwanHubX/charts/releases )做版本 tag 索引 ， **请提前确认网络连通性!**

```bash
# 建议先使用 --dry-run 验证模板兼容性
helm upgrade swanlab-self-hosted swanlab/self-hosted \
  --version <target_version> \
  -f <your_own_values.yaml> \
  --namespace <your_namespace> \
  --dry-run
```
- 确认无报错后，去掉 `--dry-run` 选项执行更新


### 选项二：本地 Chart 包更新

如果您的**集群节点无法直接访问 Helm 仓库**（即集群节点无法直接访问 `github.com`），可以通过 OCI 方式拉取 chart 包到本地后执行更新：

> 如遇到 401 认证失败问题，可以通过 `helm registry logout xxx.com` 的形式清除本机此前存在的 helm 登录态。

```bash
# 拉取 chart 包到本地
helm pull oci://swanlab-registry.cn-hangzhou.cr.aliyuncs.com/chart/self-hosted --version <target_version>
# 解压 chart 包，预期只有一个 self-hosted/ 文件夹
tar -zxvf self-hosted-<target_version>.tgz
```

然后使用本地 chart 包更新验证：
```bash
# 建议先使用 --dry-run 验证模板兼容性
helm upgrade swanlab-self-hosted ./self-hosted/ \
  -f <your_own_values.yaml> \
  --namespace <your_namespace> \
  --dry-run
```
- 确认无报错后，去掉 `--dry-run` 选项执行更新


> 其中：`--dry-run` 用于验证更新模板的兼容性，建议每次更新前均做一下模板语法验证。

## 3. 更新验证

更新完成后，请按照以下步骤验证服务是否正常运行。

### 3.1 Release 版本号验证

确认 release 版本号已更新：

```bash
helm list -n <your_namespace>
```

确保集群中的 Service 和 Pod 均为正常状态， 已经为 `deployed`。

### 3.2 Pod 健康状态检查

确保所有 Pod 运行正常：

```bash
kubectl get pods -n <your_namespace>
```

所有 Pod 应处于 `Running` 或 `Completed` 状态，不应有 `CrashLoopBackOff`、`Error` 等异常状态。

### 3.3 指标上报测试

- **页面访问**：确认前端页面可以正常访问，指标可以正常下载
- **Python SDK**：确认 SDK 连接正常，可正常上传实验

```python
import swanlab
import random
import numpy as np 
import time

swanlab.login(
    api_key="xxxxx", # 私有化 swanlab 服务下的有效 api_key
    host="xxxxxx"  # 您的私有化 swanlab 服务域名
)

# 创建一个SwanLab项目
swanlab.init(
    # 设置项目名
    project="my-first-project",
    experiment_name="my-first-experiment",
    # 设置超参数
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10
    }
)

# 模拟一次训练
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    swanlab.log({
        "step_time": acc,
        "speed": loss
    })

# 生成随机噪声图像（64x64 RGB，随机像素值）
random_noise = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
img = swanlab.Image(random_noise, caption="Random Noise")
swanlab.log({
    "image": img
})

# [可选] 完成训练，这在notebook环境中是必要的
swanlab.finish()
```

## 4. 回滚

如果在更新后发现服务无法启动（如 `CrashLoopBackOff`），或存在其他适配性问题，请通过以下命令立即回滚：

### 4.1 回滚到上一个版本

```bash
helm rollback swanlab-self-hosted -n <your_namespace>
```

### 4.2 回滚到指定版本

首先查看历史版本，确定要回滚的 `REVISION` 号：

```bash
helm history swanlab-self-hosted -n <your_namespace>
```

然后执行回滚：

```bash
helm rollback swanlab-self-hosted <revision_number> -n <your_namespace>
```

回滚完成后，请参照[更新验证](#_3-更新验证)步骤再次确认服务状态。
