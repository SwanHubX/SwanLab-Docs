# REST API

SwanLab OpenAPI reference covering Public Cloud OAuth2 authorization and self-hosted instance administration. Use these endpoints to integrate SwanLab into third-party applications or internal enterprise workflows.

## Overview

| API Category               | Base URL                              | Availability                                                                                 |
| -------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------- |
| OAuth2 Authorization       | `https://api.swanlab.cn/api`          | Public Cloud only                                                                            |
| Self-Hosted Administration | `https://your-swanlab-domain.com/api` | **Self-Hosted Commercial Edition** only (not available in Personal Edition; **App ≥ 3.6.0**) |

- **Content-Type**: Both request and response payloads use `application/json`.
- **Base URL substitution**: For self-hosted administration endpoints, replace `your-swanlab-domain.com` with the actual domain of your self-hosted SwanLab instance.

## Authentication

| Endpoint                         | Authorization Header                             |
| -------------------------------- | ------------------------------------------------ |
| `POST /auth/oauth2/token`        | `Authorization: Basic base64(app_id:app_secret)` |
| `GET /auth/oauth2/user/*`        | `Authorization: Bearer <access_token>`           |
| `POST /self_hosted/root`         | No authentication required                       |
| Other `/self_hosted/*` endpoints | `Authorization: ApiKey <root user's API Key>`    |

::: warning ⚠️ Important
Requests to `/self_hosted/*` endpoints must include the case-sensitive `ApiKey` prefix followed by a single space and the key. Passing a bare key without the prefix is treated as unauthenticated and returns `401 Unauthorized`; using a non-root user's API Key returns `403 Forbidden`.
:::

## OAuth2 Authorization

Designed for third-party applications integrating with SwanLab Public Cloud (`https://api.swanlab.cn/api`):

::: tip Integration Inquiry
OAuth2 authorization is available on SwanLab Public Cloud only. To register an OAuth application, please contact us at <contact@swanlab.cn>.
:::

- Authenticate with `app_id` and `app_secret` to exchange an authorization code or refresh token for an `access_token`.
- Use a valid `access_token` to retrieve the authenticated user's public profile or obtain their latest API Key for subsequent OpenAPI calls.

| Endpoint                        | Description                                                       | Reference                                     |
| ------------------------------- | ----------------------------------------------------------------- | --------------------------------------------- |
| `POST /auth/oauth2/token`       | Issue or refresh an `access_token` via authorization code / token | [View](/en/api/rest/oauth2-token.html)        |
| `GET /auth/oauth2/user/info`    | Retrieve the authenticated user's public profile                  | [View](/en/api/rest/oauth2-user-info.html)    |
| `GET /auth/oauth2/user/api_key` | Retrieve the authenticated user's latest API Key                  | [View](/en/api/rest/oauth2-user-api_key.html) |

## Self-Hosted Administration

Account lifecycle and access control APIs for **Self-Hosted Commercial Edition** instances, invoked via your instance domain (`https://your-swanlab-domain.com/api`).

### User Management

Initialize the single root administrator (`root`) on first deployment using a commercial license, then provision users in bulk, reset passwords, deactivate/restore/delete accounts, and retrieve any active user's latest API Key.

| Endpoint                               | Description                                                                   | Reference                                             |
| -------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------- |
| `POST /self_hosted/root`               | Initialize the single root user on first deployment (requires valid License)  | [View](/en/api/rest/self-hosted-root.html)            |
| `POST /self_hosted/users`              | Bulk create user accounts                                                     | [View](/en/api/rest/self-hosted-create-users.html)    |
| `POST /self_hosted/user/password`      | Change a user's password (the root user can reset any user's password)        | [View](/en/api/rest/self-hosted-change-password.html) |
| `POST /self_hosted/user/pause`         | Deactivate a user and revoke their API Keys and active sessions               | [View](/en/api/rest/self-hosted-pause-user.html)      |
| `POST /self_hosted/user/unpause`       | Restore a deactivated user account                                            | [View](/en/api/rest/self-hosted-unpause-user.html)    |
| `POST /self_hosted/user/del`           | Soft-delete a user and purge associated sessions, cache, and SSO bindings     | [View](/en/api/rest/self-hosted-delete-user.html)     |
| `GET /self_hosted/user/key/{username}` | Retrieve a user's latest API Key (automatically generates one if none exists) | [View](/en/api/rest/self-hosted-user-latest-key.html) |

### Organization Management

Create organization workspaces with a designated owner and manage organization memberships and roles.

| Endpoint                            | Description                                              | Reference                                                |
| ----------------------------------- | -------------------------------------------------------- | -------------------------------------------------------- |
| `POST /self_hosted/team`            | Create an organization workspace with a designated owner | [View](/en/api/rest/self-hosted-create-team.html)        |
| `POST /self_hosted/team/member`     | Add an organization member or update their role          | [View](/en/api/rest/self-hosted-upsert-team-member.html) |
| `POST /self_hosted/team/member/del` | Remove a member from an organization                     | [View](/en/api/rest/self-hosted-delete-team-member.html) |

### Project Collaboration

Manage project collaborators and their read/write/admin permissions.

| Endpoint                                     | Description                                           | Reference                                                 |
| -------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------- |
| `POST /self_hosted/project/collaborator`     | Add a project collaborator or update their permission | [View](/en/api/rest/self-hosted-upsert-collaborator.html) |
| `POST /self_hosted/project/collaborator/del` | Remove a collaborator from a project                  | [View](/en/api/rest/self-hosted-delete-collaborator.html) |

### SSO Binding

Administrators can directly bind existing SwanLab users to external SSO accounts without requiring user interaction. Unbinding and rebinding are supported, including cleaning up residual bindings under disabled providers or deactivated users.

| Endpoint                         | Description                                                                   | Reference                                        |
| -------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------ |
| `POST /self_hosted/sso/bind`     | Bind a SwanLab user to a third-party SSO account (duplicate returns `409`)    | [View](/en/api/rest/self-hosted-bind-sso.html)   |
| `POST /self_hosted/sso/bind/del` | Unbind a third-party SSO account (supports cleaning up residual SSO bindings) | [View](/en/api/rest/self-hosted-unbind-sso.html) |
