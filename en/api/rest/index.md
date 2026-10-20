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
