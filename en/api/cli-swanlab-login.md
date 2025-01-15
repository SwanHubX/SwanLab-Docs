# swanlab login

``` bash
swanlab login [OPTIONS]
```

| Option | Description |
| --- | --- |
| `--relogin` | Re-login. |

## Introduction

Log in to your SwanLab account to synchronize experiments to the cloud.

After running the following command, if it's your first login, you will be prompted to fill in your [API_KEY](https://swanlab.cn/settings):

```bash
swanlab login
```

After the first login, the credentials will be saved locally, and you won't need to log in again via `swanlab.login` or `swanlab login`.

## Re-login

If you need to log in with a different account, use the following command:

```bash
swanlab login --relogin
```

This will prompt you to enter a new API Key to re-login.

## Logout

```bash
swanlab logout
```