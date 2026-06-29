# swanlab login

```bash
swanlab login [OPTIONS]
```

| Option             | Description                                                                                                                   |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `-r`, `--relogin`  | Re-login.                                                                                                                     |
| `-h`, `--host`     | Specify the host where the SwanLab service is located. For example, `http://localhost:8000`.                                  |
| `-k`, `--api-key`  | Specify the API Key. This allows automatic login if you prefer not to enter the API key via the command line.                 |
| `-w`, `--web-host` | Specify the web host where the SwanLab frontend is located.                                                                   |
| `--local`          | Save login credentials at the project level (`.swanlab/` folder in the current directory) instead of the user home directory. |

## Introduction

Log in to your SwanLab account to synchronize experiments to the cloud.

After executing the following command, if it's your first time logging in, you will be prompted to enter your [API_KEY](https://swanlab.cn/settings):

```bash
swanlab login
```

After logging in once, the credentials will be saved locally, overwriting any previously saved credentials, eliminating the need to log in again via `swanlab.login` or `swanlab login`.

> If you do not want the credentials to be saved locally, use [swanlab.login()](./py-login.md) in a Python script to log in.

If your computer is not suitable for entering the API Key via the command line (e.g., some Windows CMD), you can use:

```bash
swanlab login -k <api-key>
```

## Re-login

If you need to log in with a different account, use the following command:

```bash
swanlab login --relogin
```

This will prompt you to enter a new API Key to log in again.

## Logging in to a Private Service

```bash
swanlab login --host <host>
```

## Local Login

By default, login credentials are saved in the user's home directory and apply to all projects. If you want to save credentials only at the project level (in the `.swanlab/` folder of the current directory), use the `--local` option:

```bash
swanlab login --local
```

> After local login, SwanLab automatically creates a `.gitignore` file in the `.swanlab/` directory to prevent credentials from being accidentally committed to a Git repository.

::: tip
The `--local` option is only available in SDK ≥ 0.8.0.
:::

Local login is ideal for shared servers and multi-project isolation — each project can use a different account independently.

## Logout

```bash
swanlab logout
```

To logout from local login:

```bash
swanlab logout --local
```
