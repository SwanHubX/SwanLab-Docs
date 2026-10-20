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
