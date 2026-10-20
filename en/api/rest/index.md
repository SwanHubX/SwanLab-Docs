# Admin Console

Open API for the SwanLab admin console. If you plan to integrate SwanLab into a third-party application, this section will help.

These endpoints cover OAuth2 authorization and self-hosted user management.

## Basics

- **Base URL**: `https://api.swanlab.cn/api`
- **Content type**: `application/json` for requests and responses

## Authentication

| Endpoint                         | Authentication                                   |
| -------------------------------- | ------------------------------------------------ |
| `POST /auth/oauth2/token`        | `Authorization: Basic base64(app_id:app_secret)` |
| `GET /auth/oauth2/user/*`        | `Authorization: Bearer <access_token>`           |
| `POST /self_hosted/root`         | No authentication                                |
| Other `/self_hosted/*` endpoints | `Authorization: ApiKey <root user's API Key>`    |
