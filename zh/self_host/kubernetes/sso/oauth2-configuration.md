# # 管理员配置 OAuth2 SSO 登录（以 Keycloak 为例）

本文说明如何在 SwanLab 中配置 OAuth2 SSO Provider，以及 IdP 侧需要填写的回调地址和字段映射。

## 一、功能简介

OAuth2 SSO 适用于只提供 OAuth2 Authorization Code 流程的身份服务。SwanLab 会跳转到 IdP 授权页，使用授权码换取 Access Token，再调用 Userinfo 接口读取第三方用户信息。

账号策略：

- 首次 OAuth2 登录必须绑定已有 SwanLab 账号。
- 已绑定用户：认证成功后直接登录。

## 二、操作流程

### 2.1 SwanLab 配置

1. 在 SwanLab 管理员控制面板点击创建 OAuth2

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626145929761.png)

2. 填写基础配置

| 字段     | 是否必填 | 说明                                                                |
| -------- | -------- | ------------------------------------------------------------------- |
| 名称     | 是       | Provider 唯一标识，最多 25 个字符，只允许字母、数字、下划线、连字符 |
| 展示名称 | 是       | 登录按钮展示的名称，最多 100 个字符                                 |

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626150631613.png)

3. 填写 OAuth2 配置

| 字段              | 是否必填 | 说明                                                                                            |
| ----------------- | -------- | ----------------------------------------------------------------------------------------------- |
| Client ID         | 是       | IdP 上配置的 Client ID，一般在 IDP 上配置时手动填写                                             |
| Client Secret     | 是       | IdP 分配给 SwanLab 的 OAuth2 Client Secret，一般为 IDP 自动生成，需要从 IDP 获取                |
| 授权地址          | 是       | OAuth2 Authorization Endpoint，不同认证服务可能不同，需要查看 IDP 文档获取                      |
| Access Token 地址 | 是       | 使用授权码换取 Access Token 的 Token Endpoint，不同认证服务可能不同，需要查看 IDP 文档获取      |
| 用户信息地址      | 是       | 使用 Access Token 拉取用户资料的 Userinfo Endpoint，不同认证服务可能不同，需要查看 IDP 文档获取 |
| Scopes            | 否       | 空格分隔的授权范围，留空时默认 `openid profile`，但建议手动填写                                 |

其中“授权地址”、“Access Token 地址”、“用户信息地址”是 OAuth2 协议内容，但不同服务并没有统一的路径要求，例如对于“授权地址”：

- keycloak：`https://<IDP URL>/realms/<realm name>/protocol/openid-connect/auth`
- authentik: `https://<IDP URL>/application/o/authorize/`

在这里以 `kecloak` 为例：

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626152250790.png)

4. 填写用户映射

| 字段         | 是否必填 | 说明                           |
| ------------ | -------- | ------------------------------ |
| 用户 ID 字段 | 是       | 第三方用户唯一 ID 对应的字段名 |
| 用户名字段   | 是       | 第三方用户名对应的字段名       |

| 用户唯一 ID 推荐字段          | 用户名推荐字段                            |
| ----------------------------- | ----------------------------------------- |
| `id`、`uid`、`user_id`、`sub` | `username`、`login`、`preferred_username` |

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626152613209.png)

5. 填写展示配置

图标 URL 推荐填写在线链接

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626152945594.png)

6. 启用 IDP

创建后即可在列表中查看当前已有 IDP，启用后即可进行测试、使用：

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626153553498.png)

7. 在登录列表中查看

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626153838893.png)

## 三、IdP 侧配置

重点需要在 IdP 的 OAuth2 应用中配置 Redirect URI：

```text
https://<SwanLab 外部访问地址>/api/auth/sso/oauth2/callback/<Provider 名称>
```

请确保该地址与 SwanLab 实际生成的地址完全一致，包括协议、域名、端口、子路径和 Provider 名称。
示例：

```text
https://swanlab.example.com/api/auth/sso/oauth2/callback/arui-oauth2
```

以 `keycloak` 中的配置为例：

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626154427440.png)

Client Secret 一般由 IDP 生成，需要复制并填写到 SwanLab 配置：
![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626154713024.png)
