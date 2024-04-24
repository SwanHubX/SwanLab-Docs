# swanlab.login

``` bash
login(
    api_key: str
    ):
```

| 参数 | 描述 |
| --- | --- |
| `api_key` | 如果已经登录，强制重新登录。|

## 介绍

登录SwanLab账号，以同步实验到云端。[API_KEY获取地址]()

```python
import swanlab

swanlab.login(api_key='your-api-key')

```

登录过一次后，凭证会保存到本地，无需再次通过`swanlab.login`或`swanlab login`登录。