# REST API

SwanLab 开放 API 文档，涵盖公有云 OAuth2 授权接口与私有化实例后台管理接口。若您计划将 SwanLab 集成至第三方平台或企业内部系统，可参考本文档进行对接。

## 基础信息

| 接口类型       | Base URL                              | 适用范围                                                       |
| -------------- | ------------------------------------- | -------------------------------------------------------------- |
| OAuth2 授权    | `https://api.swanlab.cn/api`          | 仅公有云                                                       |
| 私有化后台管理 | `https://your-swanlab-domain.com/api` | 仅**私有化商业版**（个人版暂不支持，版本要求 **APP ≥ 3.6.0**） |

⚠️ 注意：

- 所有接口的请求体与响应体格式均为 `application/json`。
- 调用私有化后台管理接口时，请将 `your-swanlab-domain.com` 替换为您的 SwanLab 私有化实例实际域名。

## 认证方式

| 接口                       | 认证头格式                                       |
| -------------------------- | ------------------------------------------------ |
| `POST /auth/oauth2/token`  | `Authorization: Basic base64(app_id:app_secret)` |
| `GET /auth/oauth2/user/*`  | `Authorization: Bearer <access_token>`           |
| `POST /self_hosted/root`   | 无需认证                                         |
| 其余 `/self_hosted/*` 接口 | `Authorization: ApiKey <root 用户的 API Key>`    |

::: warning ⚠️ 注意
`/self_hosted/*` 系列接口的 `Authorization` 请求头必须包含 `ApiKey` 前缀（区分大小写，与密钥之间以单个空格分隔）。若仅传入密钥本身，将被视为未携带有效凭证并返回 `401 Unauthorized`；若使用非 `root` 用户的 API Key，则返回 `403 Forbidden`。
:::

## OAuth2 授权

面向已接入 SwanLab 的第三方应用，统一通过公有云地址 `https://api.swanlab.cn/api` 调用：

::: tip 接入咨询
OAuth2 授权仅支持公有云服务。如需申请应用接入，请联系我们：<contact@swanlab.cn>
:::

- 使用 `app_id` 与 `app_secret` 验证应用身份并换取 `access_token`；
- 在令牌有效期内获取当前授权用户的公开资料，或获取该用户的 API Key 以调用后续开放接口。

| 接口                            | 功能概述                                              | 文档链接                                   |
| ------------------------------- | ----------------------------------------------------- | ------------------------------------------ |
| `POST /auth/oauth2/token`       | 使用授权码或 `refresh_token` 获取/刷新 `access_token` | [查看](/api/rest/oauth2-token.html)        |
| `GET /auth/oauth2/user/info`    | 凭 `access_token` 获取当前授权用户的公开信息          | [查看](/api/rest/oauth2-user-info.html)    |
| `GET /auth/oauth2/user/api_key` | 凭 `access_token` 获取该用户的最新 API Key            | [查看](/api/rest/oauth2-user-api_key.html) |

## 私有化后台管理

面向**私有化商业版**实例的账号生命周期与权限管理接口，需通过实例域名（`https://your-swanlab-domain.com/api`）调用。

### 用户管理

首次部署时需凭商业版 License 初始化唯一的根用户（`root`），后续支持批量创建用户、重置密码、停用/恢复/删除用户，以及获取任意启用用户的最新 API Key。

| 接口                                   | 功能概述                                                 | 文档链接                                           |
| -------------------------------------- | -------------------------------------------------------- | -------------------------------------------------- |
| `POST /self_hosted/root`               | 首次部署时初始化唯一的根用户（需验证 License）           | [查看](/api/rest/self-hosted-root.html)            |
| `POST /self_hosted/users`              | 批量创建用户                                             | [查看](/api/rest/self-hosted-create-users.html)    |
| `POST /self_hosted/user/password`      | 修改用户密码（根用户可重置任意用户密码）                 | [查看](/api/rest/self-hosted-change-password.html) |
| `POST /self_hosted/user/pause`         | 停用用户，并同步清除其 API Key 与活跃会话                | [查看](/api/rest/self-hosted-pause-user.html)      |
| `POST /self_hosted/user/unpause`       | 恢复已停用的用户                                         | [查看](/api/rest/self-hosted-unpause-user.html)    |
| `POST /self_hosted/user/del`           | 软删除用户并清理关联会话、缓存及 SSO 绑定                | [查看](/api/rest/self-hosted-delete-user.html)     |
| `GET /self_hosted/user/key/{username}` | 获取指定用户的最新 API Key（若尚无密钥则自动创建并返回） | [查看](/api/rest/self-hosted-user-latest-key.html) |

### 组织管理

创建团队组织空间并指定归属人，管理组织成员及其角色。

| 接口                                | 功能概述                   | 文档链接                                              |
| ----------------------------------- | -------------------------- | ----------------------------------------------------- |
| `POST /self_hosted/team`            | 创建组织空间并指定归属用户 | [查看](/api/rest/self-hosted-create-team.html)        |
| `POST /self_hosted/team/member`     | 添加组织成员或变更成员角色 | [查看](/api/rest/self-hosted-upsert-team-member.html) |
| `POST /self_hosted/team/member/del` | 从组织中移除指定成员       | [查看](/api/rest/self-hosted-delete-team-member.html) |

### 项目协作

管理项目协作者及其读写权限。

| 接口                                         | 功能概述                       | 文档链接                                               |
| -------------------------------------------- | ------------------------------ | ------------------------------------------------------ |
| `POST /self_hosted/project/collaborator`     | 添加项目协作者或变更其权限角色 | [查看](/api/rest/self-hosted-upsert-collaborator.html) |
| `POST /self_hosted/project/collaborator/del` | 移除项目协作者                 | [查看](/api/rest/self-hosted-delete-collaborator.html) |

### SSO 绑定

管理员可直接将现有 SwanLab 用户与第三方 SSO 账号进行绑定，无需用户手动操作；同时支持解绑与换绑，并可清理已停用账号或已停用 Provider 遗留的绑定关系。

| 接口                             | 功能概述                                                      | 文档链接                                      |
| -------------------------------- | ------------------------------------------------------------- | --------------------------------------------- |
| `POST /self_hosted/sso/bind`     | 将用户与第三方 SSO 账号绑定（重复绑定返回 `409`）             | [查看](/api/rest/self-hosted-bind-sso.html)   |
| `POST /self_hosted/sso/bind/del` | 解绑第三方 SSO 账号，支持清理已停用 Provider 下的遗留绑定记录 | [查看](/api/rest/self-hosted-unbind-sso.html) |
