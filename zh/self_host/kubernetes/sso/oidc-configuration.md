# 管理员配置 OIDC SSO 登录（以 Keycloak 为例）

本文说明如何在 SwanLab 中配置 OIDC SSO Provider，以及 IdP 侧需要填写的回调地址和字段映射。

## 一、功能简介

OIDC（OpenID Connect）是在 OAuth2 基础上增加身份层的标准协议。SwanLab 会通过 Issuer Discovery 获取授权端点、Token 端点和 JWKS，完成授权码流程后校验 ID Token，并从 ID Token claims 中映射第三方用户。

账号策略：

- 首次 OIDC 登录会自动创建 SwanLab 账号并绑定。
- 已绑定用户：认证成功后直接登录。

## 二、操作流程

1. 在 SwanLab 管理员控制面板点击创建 OIDC

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626172105680.png)

2. 填写基础配置

| 字段     | 是否必填 | 说明                                                                |
| -------- | -------- | ------------------------------------------------------------------- |
| 名称     | 是       | Provider 唯一标识，最多 25 个字符，只允许字母、数字、下划线、连字符 |
| 展示名称 | 是       | 登录按钮展示的名称，最多 100 个字符                                 |

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626172631299.png)

3. 填写 OIDC 配置

| 字段          | 是否必填 | 说明                                                                             |
| ------------- | -------- | -------------------------------------------------------------------------------- |
| Client ID     | 是       | IdP 上配置的 Client ID，一般在 IDP 上配置时手动填写                              |
| Client Secret | 是       | IdP 分配给 SwanLab 的 OAuth2 Client Secret，一般为 IDP 自动生成，需要从 IDP 获取 |
| 授权地址      | 是       | OIDC **Issuer URL**，SwanLab 会基于该地址做 Discovery，注意不是具体的接口地址    |
| Scopes        | 否       | 空格分隔的授权范围，留空时默认 `openid profile`，但建议手动填写                  |

> “授权地址”仅需填写 OIDC Issuer 地址，不需要和 OAuth2 一样具体到某个路径上，因为 OIDC 对路由进行了约束，所以直接使用标准协议进行路由发现，若企业 IDP 相关配置不符合标准，则需要自行调整

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626173104696.png)

4. 填写用户映射

| 字段         | 是否必填 | 说明                                                     |
| ------------ | -------- | -------------------------------------------------------- |
| 用户 ID 字段 | 是       | 第三方用户唯一 ID 对应的字段名                           |
| 用户名字段   | 是       | 第三方用户名对应的字段名，自动创建账号时将作为默认用户名 |

| 用户唯一 ID 推荐字段 | 用户名推荐字段               |
| -------------------- | ---------------------------- |
| `sub`                | `preferred_username`、`name` |

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626173544832.png)

5. 填写展示配置

图标 URL 推荐填写在线链接

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626173837111.png)

6. 启用 IDP

创建后即可在列表中查看当前已有 IDP，启用后即可进行测试、使用：

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626174020396.png)

7. 在登录列表中查看

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626174323178.png)

## 三、IdP 侧配置

在 IdP 的 OIDC 应用中配置 Redirect URI：

```text
https://<SwanLab 外部访问地址>/api/auth/sso/oidc/callback/<Provider 名称>
```

请确保该地址与 SwanLab 实际生成的地址完全一致，包括协议、域名、端口、子路径和 Provider 名称。
示例：

```text
https://swanlab.example.com/api/auth/sso/oidc/callback/arui-oidc
```

> 建议启用 Authorization Code Flow，并确保返回 ID Token。

以 keycloak 中的配置为例：

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626175104031.png)

Client Secret 一般由 IDP 生成，需要复制并填写到 SwanLab 配置：

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626175225603.png)

> SwanLab 后台对于 OIDC/SAML2 会缓存 IDP 信息，若 IDP 侧信息发生改变，SwanLab 不一定会及时同步相关更新，可以通过修改 SwanLab 中对应配置的任意字段刷新缓存。
