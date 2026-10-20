# Admin Console

Open API for the SwanLab admin console. If you plan to integrate SwanLab into a third-party application, this section will help.

## Scenarios

- **OAuth2 authorization**: for third-party applications integrated with SwanLab.
  - exchange `app_id` & `app_secret` for an Access Token;
  - while the token is valid, fetch the current user's profile or that user's API Key, which authorizes further calls to the open API.
- **Self-hosted user management**: account operations for a self-hosted instance.
  - the first deployment creates the single root user (requires a License), and that root user then manages all accounts;
  - bulk user creation, password resets, deactivate/restore/delete, and retrieval of any user's latest API Key, so keys no longer need to be distributed by hand.
- **Organization & collaboration management**: permission orchestration across teams and projects.
  - create organization spaces with a designated owner and adjust member roles;
  - add project collaborators with read/write permissions, and remove members or collaborators in bulk.
- **SSO account binding**: for instances connected to a corporate identity provider.
  - administrators bind existing users to third-party accounts directly, with no self-service step;
  - unbinding and rebinding are supported, and leftover bindings under deactivated users or Providers can be cleaned up.

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

## Endpoint summary

| Endpoint                                     | What it does                                                       | Docs                                                      |
| -------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------- |
| `POST /auth/oauth2/token`                    | Exchange a code or refresh token for an Access Token               | [View](/en/api/rest/oauth2-token.html)                    |
| `GET /auth/oauth2/user/info`                 | Fetch the current user's public profile with an Access Token       | [View](/en/api/rest/oauth2-user-info.html)                |
| `GET /auth/oauth2/user/api_key`              | Fetch the user's latest API Key with an Access Token               | [View](/en/api/rest/oauth2-user-api_key.html)             |
| `POST /self_hosted/root`                     | Create the single root user on first deployment (License required) | [View](/en/api/rest/self-hosted-root.html)                |
| `POST /self_hosted/user/password`            | Change a password (the root user can change any user's)            | [View](/en/api/rest/self-hosted-change-password.html)     |
| `POST /self_hosted/users`                    | Create users in bulk                                               | [View](/en/api/rest/self-hosted-create-users.html)        |
| `POST /self_hosted/user/pause`               | Deactivate a user and clear their API Key and sessions             | [View](/en/api/rest/self-hosted-pause-user.html)          |
| `POST /self_hosted/user/unpause`             | Restore a deactivated user                                         | [View](/en/api/rest/self-hosted-unpause-user.html)        |
| `POST /self_hosted/user/del`                 | Soft-delete a user and purge sessions and cache                    | [View](/en/api/rest/self-hosted-delete-user.html)         |
| `GET /self_hosted/user/key/{username}`       | Get a user's latest API Key, creating one when none exists         | [View](/en/api/rest/self-hosted-user-latest-key.html)     |
| `POST /self_hosted/team/member`              | Invite a member to an organization or change their role            | [View](/en/api/rest/self-hosted-upsert-team-member.html)  |
| `POST /self_hosted/team/member/del`          | Remove a member from an organization                               | [View](/en/api/rest/self-hosted-delete-team-member.html)  |
| `POST /self_hosted/project/collaborator`     | Invite a project collaborator or change their permission           | [View](/en/api/rest/self-hosted-upsert-collaborator.html) |
| `POST /self_hosted/project/collaborator/del` | Remove a project collaborator                                      | [View](/en/api/rest/self-hosted-delete-collaborator.html) |
| `POST /self_hosted/team`                     | Create an organization space with a designated owner               | [View](/en/api/rest/self-hosted-create-team.html)         |
| `POST /self_hosted/sso/bind`                 | Bind a user to a third-party account (duplicate returns 409)       | [View](/en/api/rest/self-hosted-bind-sso.html)            |
| `POST /self_hosted/sso/bind/del`             | Unbind an account, including leftovers under a disabled Provider   | [View](/en/api/rest/self-hosted-unbind-sso.html)          |
