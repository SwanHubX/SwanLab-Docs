# 管理员配置 SAML2 SSO 登录（以 Keycloak、Authentik 为例）

本文说明如何在 SwanLab 中配置 SAML2 SSO Provider，以及 IdP 侧需要填写的 SP Metadata、ACS、EntityID 和属性映射。

## 一、功能简介

SAML2 SSO 适用于企业已有 SAML IdP 的场景。SwanLab 当前作为 SP（Service Provider）发起登录，请求会被签名；IdP 返回 SAMLResponse 后，SwanLab 在 ACS 中校验响应、解析 Assertion，并根据 Attribute 映射出第三方用户。

账号策略：

- 首次 SAML2 登录会自动创建 SwanLab 账号并绑定。
- 已绑定用户：认证成功后直接登录。

当前仅支持 SP-initiated 登录，不支持用户从 IdP 门户直接发起登录。

## 二、操作流程

1. 在 SwanLab 管理员控制面板点击创建 SAML2

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626175725388.png)

2. 填写基础配置

| 字段     | 是否必填 | 说明                                                                |
| -------- | -------- | ------------------------------------------------------------------- |
| 名称     | 是       | Provider 唯一标识，最多 25 个字符，只允许字母、数字、下划线、连字符 |
| 展示名称 | 是       | 登录按钮展示的名称，最多 100 个字符                                 |

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626175623167.png)

3. 填写 SAML2 配置

| 字段            | 是否必填 | 说明                                |
| --------------- | -------- | ----------------------------------- |
| SAML 元数据地址 | 是       | IdP Metadata XML 的 URL             |
| SAML 私钥内容   | 是       | SwanLab 作为 SP 使用的私钥 PEM 内容 |
| SAML 证书内容   | 是       | SwanLab 作为 SP 使用的证书 PEM 内容 |

> SwanLab 会签名 AuthnRequest，所以证书为必填项，配置 IdP 时也需要在 IdP 上填写此处用到的证书。

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626180857336.png)

4. 填写用户映射

| 字段         | 是否必填 | 说明                                                     |
| ------------ | -------- | -------------------------------------------------------- |
| 用户 ID 字段 | 是       | 第三方用户唯一 ID 对应的字段名                           |
| 用户名字段   | 是       | 第三方用户名对应的字段名，自动创建账号时将作为默认用户名 |

| 用户唯一 ID 推荐字段                              | 用户名推荐字段    |
| ------------------------------------------------- | ----------------- |
| `uid`、`employeeNumber`、`email` 等稳定 Attribute | `username`、`uid` |

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626181459970.png)

5. 填写展示配置

图标 URL 推荐填写在线链接

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626181535376.png)

6. 启用 IdP

创建后即可在列表中查看当前已有 IdP，启用后即可进行测试、使用：

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626181628925.png)

7. 在登录列表中查看

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626181753499.png)

## 三、SwanLab SP 地址

SwanLab 会为每个 SAML2 Provider 生成独立的 SP 地址。

SP Metadata 地址：

```text
https://<SwanLab 外部访问地址>/api/auth/sso/saml2/metadata/<Provider 名称>
```

ACS 地址：

```text
https://<SwanLab 外部访问地址>/api/auth/sso/saml2/acs/<Provider 名称>
```

EntityID：

```text
https://<SwanLab 外部访问地址>/api/auth/sso/saml2/metadata/<Provider 名称>
```

示例：

```text
SP Metadata: https://swanlab.example.com/api/auth/sso/saml2/metadata/corp-saml2
ACS:         https://swanlab.example.com/api/auth/sso/saml2/acs/corp-saml2
EntityID:    https://swanlab.example.com/api/auth/sso/saml2/metadata/corp-saml2
```

## 四、IdP 侧配置

在 IdP 中创建 SAML 应用时，可能需要配置：

| IdP 字段                     | SwanLab 填写值             |
| ---------------------------- | -------------------------- |
| Single sign-on URL / ACS URL | SwanLab ACS 地址           |
| Recipient URL                | SwanLab ACS 地址           |
| Destination URL              | SwanLab ACS 地址           |
| Audience URI / SP Entity ID  | SwanLab EntityID           |
| Name ID format               | 可使用默认或 `unspecified` |
| Response 签名                | 建议开启                   |
| Assertion 签名               | 建议开启                   |
| SP 发起登录                  | 必须允许                   |

SwanLab 会签名 AuthnRequest，IdP 需要接受签名请求。当前 SAML Redirect Binding 签名算法会根据私钥类型选择 RSA-SHA256 或 ECDSA-SHA256。

1. 以 keycloak 配置为例：

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626182359746.png)

其中 Client ID 即为 Issuer，通常需要配置为 SP Issuer 地址，对应 SwanLab 的 EntityID 地址：

```
https://<SwanLab 外部访问地址>/api/auth/sso/saml2/metadata/<Provider 名称>
```

同时需要在 keycloak 中配置之前上传 SwanLab 的证书，否则无法解析请求：

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626183046996.png)

2. 以 Authentik 配置为例：

其中明确要求填写：

- ACS URL：对应 SwanLab ACS 地址 `https://swanlab.example.com/api/auth/sso/saml2/acs/corp-saml2`
- 颁发者：对应 SwanLab EntityID：`https://swanlab.example.com/api/auth/sso/saml2/metadata/corp-saml2`
- Audience：对应 SwanLab EntityID 地址：`https://swanlab.example.com/api/auth/sso/saml2/metadata/corp-saml2`

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626182712335.png)

同时需要配置证书：

- 签名证书：Authentik 自签名证书
- 验证证书：SwanLab 配置过的证书

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626183647075.png)
