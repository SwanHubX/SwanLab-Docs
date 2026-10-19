# Admin Open API

> This section is a **vitepress-openapi integration spike**. The endpoints use mock data and are only meant to evaluate the rendered result.

Automation endpoints for administrators of self-hosted deployments, covering user and project governance.

## Authentication

All endpoints authenticate with `Authorization: Bearer <API_KEY>`. Generate an API key under "Settings → API Key" in the admin console.

## Basics

- **Base URL**: `https://your-swanlab-server.com` (replace with your self-hosted domain)
- **Content type**: `application/json` for requests and responses
- **Pagination**: list endpoints use `page` / `size`; responses include `total`

## Error codes

| Status | Meaning                                        |
| ------ | ---------------------------------------------- |
| 401    | Missing or invalid API key                     |
| 403    | Insufficient permission or forbidden operation |
| 404    | Resource not found                             |
| 409    | Conflict (e.g. username already exists)        |

Pick an endpoint from the sidebar to inspect its parameters and examples.
