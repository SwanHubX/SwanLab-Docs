# Admin Console

Open API for the SwanLab admin console. If you plan to integrate SwanLab into a third-party application, this section will help.

## Scenarios

- **OAuth2 authorization**: for third-party applications integrated with SwanLab.
  - Exchange `app_id` & `app_secret` for an Access Token via `POST /auth/oauth2/token`, then
  - Use the token to call `GET /auth/oauth2/user/info` for the current user's profile, or `GET /auth/oauth2/user/api_key` for that user's API Key, which authorizes further calls to the open API.
- **Self-hosted user management**: operational management of a self-hosted instance.
  - After deployment, call `POST /self_hosted/root` to create the single root user (requires a License);
  - the root user then uses its own API Key to manage root users, users, and collaborators.

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
