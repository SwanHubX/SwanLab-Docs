# 阿里云计算巢应用部署

:::warning 关于第三方部署

第三方部署是由社区贡献的部署方式，官方不保证能实时同步最新版本。

:::

目前 SwanLab 社区版本已上线阿里云计算巢服务市场，欢迎各位训练师通过阿里云一键部署使用~


## ⚠️ 前提条件

部署 SwanLab 社区版服务实例，需要对部分阿里云资源进行访问和创建操作。因此您的账号需要包含如下资源的权限。
**说明**：当您的账号是RAM账号时，才需要添加此权限。

| 权限策略名称                          | 备注                         |
|---------------------------------|----------------------------|
| AliyunECSFullAccess             | 管理云服务器服务（ECS）的权限           |
| AliyunVPCFullAccess             | 管理专有网络（VPC）的权限             |
| AliyunROSFullAccess             | 管理资源编排服务（ROS）的权限           |
| AliyunComputeNestUserFullAccess | 管理计算巢服务（ComputeNest）的用户侧权限 |

## 💰 计费说明

SwanLab社区版在计算巢部署的费用主要涉及：

- 所选vCPU与内存规格
- 系统盘类型及容量
- 公网带宽

## 🚀 部署流程

1. 访问计算巢SwanLab社区版[部署链接](https://computenest.console.aliyun.com/service/instance/create/cn-hangzhou?type=user&ServiceId=service-cb2da57160444c3ebdbf)
，按提示填写部署参数：
<img src="./alibabacloud-computenest/deploy_service_instance.jpg" width="800"/>

2. 参数填写完成后可以看到对应询价明细，确认参数后点击**下一步：确认订单**。

3. 确认订单完成后同意服务协议并点击**立即创建**进入部署阶段。

4. 等待部署完成后就可以开始使用服务，进入服务实例详情点击服务地址。
   <img src="./alibabacloud-computenest/get_service_instance.jpg" width="800"/>

5. 访问服务地址注册账号并使用SwanLab服务。
   <img src="./alibabacloud-computenest/swanlab_service.jpg" width="800"/>
