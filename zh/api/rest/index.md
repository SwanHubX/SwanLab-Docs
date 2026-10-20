# 后台管理

SwanLab 后台管理开放 API，若您有意将 SwanLab 集成到第三方应用中，该部分的内容将会对您有所帮助。

该组接口覆盖 OAuth2 授权与私有化用户管理场景。

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
