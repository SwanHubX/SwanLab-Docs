# Avoiding Key Conflicts on Shared Servers

**Usage Scenarios:**

1. **Laboratory**: When multiple trainers share a server for training, it's necessary to avoid SwanLab API Key conflicts that could cause experimental data to be mistakenly uploaded to other people's accounts.
2. **Public Server**: On public servers, prevent data security issues caused by SwanLab API Key leakage.

## Temporary Login

In your Python code, add the following line:

```python
swanlab.login(api_key="<your_api_key>")
```

Using `swanlab.login()` for authentication does not save login information locally. This ensures that experiments are definitely uploaded to your account, and prevents others from uploading to your account.

## Logout

Using `swanlab logout` will clear the locally stored login information. It is recommended to execute this before leaving a public server.

```bash
swanlab logout
```