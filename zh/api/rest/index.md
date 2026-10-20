# 后台管理

SwanLab 后台管理开放 API，若您有意将 SwanLab 集成到第三方应用中，该部分的内容将会对您有所帮助。

## 应用场景

- **OAuth2 授权**：面向已接入 SwanLab 的第三方应用。
  - 用 `app_id` & `app_secret` 换取 Access Token；
  - 在 Token 有效期内获取当前用户信息，或获取该用户的 API Key，用于后续调用开放接口。
- **私有化用户管理**：面向私有化部署实例的账号运维。
  - 首次部署创建唯一的根用户（需要 License），之后由根用户统一管理全员账号；
  - 支持批量建号、重置密码、停用/恢复/删除用户，以及取用任一用户的最新 API Key，省去逐台分发。
- **组织与协作管理**：面向多团队、多项目的权限编排。
  - 创建组织空间并指定归属人，调整成员角色；
  - 为项目添加协作者并设定读写权限，成员与协作者均可统一移除。
- **SSO 账号绑定**：面向已接入企业身份提供商的实例。
  - 管理员可直接把已有用户与三方账号绑定，无需用户本人操作；
  - 支持解绑与换绑，停用账号或停用 Provider 下的残留绑定也能清理。

## 基础信息

- **Base URL**：`https://api.swanlab.cn/api`
- **数据格式**：请求与响应均为 `application/json`

## 认证

| 接口                       | 认证方式                                         |
| -------------------------- | ------------------------------------------------ |
| `POST /auth/oauth2/token`  | `Authorization: Basic base64(app_id:app_secret)` |
| `GET /auth/oauth2/user/*`  | `Authorization: Bearer <access_token>`           |
| `POST /self_hosted/root`   | 无需认证                                         |
| 其余 `/self_hosted/*` 接口 | `Authorization: ApiKey <root 用户的 API Key>`    |

## 接口汇总

| 接口                                         | 功能概述                                          | 文档链接                                               |
| -------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------ |
| `POST /auth/oauth2/token`                    | 换取或刷新 Access Token（授权码 / refresh token） | [查看](/api/rest/oauth2-token.html)                    |
| `GET /auth/oauth2/user/info`                 | 用 Access Token 获取当前用户公开信息              | [查看](/api/rest/oauth2-user-info.html)                |
| `GET /auth/oauth2/user/api_key`              | 用 Access Token 获取该用户最新 API Key            | [查看](/api/rest/oauth2-user-api_key.html)             |
| `POST /self_hosted/root`                     | 首次部署创建唯一的根用户（需 License）            | [查看](/api/rest/self-hosted-root.html)                |
| `POST /self_hosted/user/password`            | 修改密码（根用户可修改所有用户）                  | [查看](/api/rest/self-hosted-change-password.html)     |
| `POST /self_hosted/users`                    | 批量创建用户                                      | [查看](/api/rest/self-hosted-create-users.html)        |
| `POST /self_hosted/user/pause`               | 停用用户，同时清空其 API Key 与会话               | [查看](/api/rest/self-hosted-pause-user.html)          |
| `POST /self_hosted/user/unpause`             | 恢复已停用的用户                                  | [查看](/api/rest/self-hosted-unpause-user.html)        |
| `POST /self_hosted/user/del`                 | 软删除用户并清理会话与缓存                        | [查看](/api/rest/self-hosted-delete-user.html)         |
| `GET /self_hosted/user/key/{username}`       | 获取用户最新 API Key，没有则自动创建              | [查看](/api/rest/self-hosted-user-latest-key.html)     |
| `POST /self_hosted/team/member`              | 邀请成员加入组织，或修改其角色                    | [查看](/api/rest/self-hosted-upsert-team-member.html)  |
| `POST /self_hosted/team/member/del`          | 从组织中移除成员                                  | [查看](/api/rest/self-hosted-delete-team-member.html)  |
| `POST /self_hosted/project/collaborator`     | 邀请项目协作者，或修改其读写权限                  | [查看](/api/rest/self-hosted-upsert-collaborator.html) |
| `POST /self_hosted/project/collaborator/del` | 移除项目协作者                                    | [查看](/api/rest/self-hosted-delete-collaborator.html) |
| `POST /self_hosted/team`                     | 创建组织空间并指定归属用户                        | [查看](/api/rest/self-hosted-create-team.html)         |
| `POST /self_hosted/sso/bind`                 | 将用户与三方账号绑定（重复绑定返回 409）          | [查看](/api/rest/self-hosted-bind-sso.html)            |
| `POST /self_hosted/sso/bind/del`             | 解绑三方账号，可清理停用 Provider 下的残留绑定    | [查看](/api/rest/self-hosted-unbind-sso.html)          |
