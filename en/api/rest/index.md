# REST API

Open API documentation for SwanLab, covering OAuth2 authorization and self-hosted administration. If you plan to integrate SwanLab into a third-party application, this section will help.

## Basics

- **Base URL**:
  - OAuth2 authorization: `https://api.swanlab.cn/api`
  - Self-hosted administration: `https://your-swanlab-domain.com/api` (replace with your instance domain)
- **Content type**: `application/json` for requests and responses

## Authentication

| Endpoint                         | Authentication                                   |
| -------------------------------- | ------------------------------------------------ |
| `POST /auth/oauth2/token`        | `Authorization: Basic base64(app_id:app_secret)` |
| `GET /auth/oauth2/user/*`        | `Authorization: Bearer <access_token>`           |
| `POST /self_hosted/root`         | No authentication                                |
| Other `/self_hosted/*` endpoints | `Authorization: ApiKey <root user's API Key>`    |

::: warning ⚠️ Warning
`/self_hosted/*` requests must include the `ApiKey` prefix (case-sensitive, followed by a single space and the key). A bare key is treated as no credentials and returns `401`; a non-root user's key returns `403`.

:::

## OAuth2 Authorization

For third-party applications integrated with SwanLab, using the public cloud at `https://api.swanlab.cn/api`:

- exchange `app_id` & `app_secret` for an Access Token;
- while the token is valid, fetch the current user's profile or that user's API Key, which authorizes further calls to the open API.

| Endpoint                        | What it does                                                 | Docs                                          |
| ------------------------------- | ------------------------------------------------------------ | --------------------------------------------- |
| `POST /auth/oauth2/token`       | Exchange a code or refresh token for an Access Token         | [View](/en/api/rest/oauth2-token.html)        |
| `GET /auth/oauth2/user/info`    | Fetch the current user's public profile with an Access Token | [View](/en/api/rest/oauth2-user-info.html)    |
| `GET /auth/oauth2/user/api_key` | Fetch the user's latest API Key with an Access Token         | [View](/en/api/rest/oauth2-user-api_key.html) |

## Self-Hosted Administration

Account and collaboration operations for self-hosted SwanLab instances, called through your instance domain (`https://your-swanlab-domain.com/api`).

### User Management

The first deployment creates the single root user (requires a License); afterwards, create users in bulk, reset passwords, deactivate/restore/delete users, and retrieve any user's latest API Key.

| Endpoint                               | What it does                                                       | Docs                                                  |
| -------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------- |
| `POST /self_hosted/root`               | Create the single root user on first deployment (License required) | [View](/en/api/rest/self-hosted-root.html)            |
| `POST /self_hosted/users`              | Create users in bulk                                               | [View](/en/api/rest/self-hosted-create-users.html)    |
| `POST /self_hosted/user/password`      | Change a password (the root user can change any user's)            | [View](/en/api/rest/self-hosted-change-password.html) |
| `POST /self_hosted/user/pause`         | Deactivate a user and clear their API Key and sessions             | [View](/en/api/rest/self-hosted-pause-user.html)      |
| `POST /self_hosted/user/unpause`       | Restore a deactivated user                                         | [View](/en/api/rest/self-hosted-unpause-user.html)    |
| `POST /self_hosted/user/del`           | Soft-delete a user and purge sessions and cache                    | [View](/en/api/rest/self-hosted-delete-user.html)     |
| `GET /self_hosted/user/key/{username}` | Get a user's latest API Key, creating one when none exists         | [View](/en/api/rest/self-hosted-user-latest-key.html) |

### Organization Management

Create organization spaces with a designated owner and adjust member roles.

| Endpoint                            | What it does                                            | Docs                                                     |
| ----------------------------------- | ------------------------------------------------------- | -------------------------------------------------------- |
| `POST /self_hosted/team`            | Create an organization space with a designated owner    | [View](/en/api/rest/self-hosted-create-team.html)        |
| `POST /self_hosted/team/member`     | Invite a member to an organization or change their role | [View](/en/api/rest/self-hosted-upsert-team-member.html) |
| `POST /self_hosted/team/member/del` | Remove a member from an organization                    | [View](/en/api/rest/self-hosted-delete-team-member.html) |

### Project Collaboration

Add project collaborators with read/write permissions.

| Endpoint                                     | What it does                                             | Docs                                                      |
| -------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------- |
| `POST /self_hosted/project/collaborator`     | Invite a project collaborator or change their permission | [View](/en/api/rest/self-hosted-upsert-collaborator.html) |
| `POST /self_hosted/project/collaborator/del` | Remove a project collaborator                            | [View](/en/api/rest/self-hosted-delete-collaborator.html) |

### SSO Binding

Administrators bind existing users to third-party accounts directly; unbinding and rebinding are supported, and leftovers under deactivated users or Providers can be cleaned up.

| Endpoint                         | What it does                                                     | Docs                                             |
| -------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------ |
| `POST /self_hosted/sso/bind`     | Bind a user to a third-party account (duplicate returns 409)     | [View](/en/api/rest/self-hosted-bind-sso.html)   |
| `POST /self_hosted/sso/bind/del` | Unbind an account, including leftovers under a disabled Provider | [View](/en/api/rest/self-hosted-unbind-sso.html) |
