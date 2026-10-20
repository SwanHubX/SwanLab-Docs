# 后台管理

SwanLab 后台管理开放 API，若您有意将 SwanLab 集成到第三方应用中，该部分的内容将会对您有所帮助。

## 应用场景

- **OAuth2 授权**：面向已接入 SwanLab 的第三方应用。
  - 用 `app_id` & `app_secret` 调用 `POST /auth/oauth2/token` 换取 Access Token，然后
  - 在 Token 有效期内调用 `GET /auth/oauth2/user/info` 获取当前用户信息，或调用 `GET /auth/oauth2/user/api_key` 获取该用户的 API Key，用于后续调用开放接口。
- **私有化用户管理**：面向私有化部署实例的运维管理。
  - 首次部署需调用 `POST /self_hosted/root` 创建唯一的根用户（需要 License）；
  - 创建完成后，由根用户使用自己的 API Key 进行根用户、用户、协作者管理操作。

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
