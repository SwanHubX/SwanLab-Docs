# SwanLab SSO 功能说明

本文说明 SwanLab 当前版本的 SSO 能力、适用场景、账号策略和用户登录流程。当前版本支持 OAuth2、OIDC 和 SAML2 三类协议。

## 一、功能简介

SSO（Single Sign-On，单点登录）允许用户使用企业已有身份提供商（IdP）的账号登录 SwanLab。管理员在 SwanLab 中配置 Provider 并启用后，登录页会展示对应的 SSO 登录入口。用户点击入口后，会跳转到企业 IdP 完成认证，再返回 SwanLab 完成登录、绑定或自动创建账号。

## 二、支持范围

| 协议   | 当前支持内容                                                               | 适用场景                                                 |
| ------ | -------------------------------------------------------------------------- | -------------------------------------------------------- |
| OAuth2 | Authorization Code、Access Token、Userinfo、字段映射                       | 通用 OAuth2 身份源、企业自建 OAuth2 服务                 |
| OIDC   | Discovery、Authorization Code、ID Token 校验、nonce 校验、claims 映射      | Keycloak、Okta、Authing、Azure AD 等标准 OIDC 服务       |
| SAML2  | SP 发起登录、签名 AuthnRequest、ACS、SP Metadata、AudienceRestriction 校验 | 企业 SAML IdP、ADFS、Keycloak SAML、Authentik、Okta SAML |

当前限制：

- SAML2 仅支持 SP-initiated 登录，不支持从 IdP 门户直接发起登录。

## 三、私有化账号策略

私有化环境下，SSO 登录后的账号处理策略如下：

| 场景               | 处理方式                                              |
| ------------------ | ----------------------------------------------------- |
| 已绑定第三方身份   | 认证成功后直接登录 SwanLab                            |
| 未绑定 OAuth2 身份 | 用户使用已有 SwanLab 账号的用户名和密码完成绑定后登录 |
| 未绑定 OIDC 身份   | SwanLab 自动创建账号并绑定                            |
| 未绑定 SAML2 身份  | SwanLab 自动创建账号并绑定                            |

### 3.1 自动创建

OIDC/SAML2 首次登录且第三方身份未绑定时，SwanLab 会自动创建账号：

1. 优先使用 IdP 返回并映射出的用户名。
2. 如果用户名合法且未被占用，页面会静默创建账号并登录。
3. 如果用户名为空、不合法或已被占用，页面会要求用户手动修改用户名。
4. 创建成功后，第三方身份会自动绑定到新账号。

用户名规则：

```text
长度：1 到 25 个字符
字符：字母、数字、下划线、连字符
```

自动创建还会检查私有化 License 用户席位。如果席位不足，创建会失败，需要管理员先释放或增加用户席位。

### 3.2 解绑策略

用户可以在账号设置中查看或管理自己的绑定关系：

- OAuth2 允许用户解绑。
- OIDC/SAML2 自动创建的账号暂不允许用户自行解绑。

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626142403293.png)

## 四、用户登录流程

1. 用户打开 SwanLab 登录页，点击企业配置的 SSO 登录入口。
2. 页面跳转到企业身份提供商，用户使用企业账号完成认证。
3. 认证成功后，用户回到 SwanLab。
4. SwanLab 根据当前账号状态完成登录、绑定已有账号，或自动创建账号。

已绑定用户会直接登录 SwanLab。如果绑定的 SwanLab 用户已被禁用，本次登录会被阻止。

## 五、管理员可用操作

管理员可以执行以下操作：

| 操作               | 说明                                                     |
| ------------------ | -------------------------------------------------------- |
| 创建 Provider      | 新增 OAuth2、OIDC 或 SAML2 Provider                      |
| 编辑 Provider      | 修改显示名称、协议字段、映射字段、Logo、排序值、启用状态 |
| 启用/禁用 Provider | 只有启用状态的 Provider 会出现在登录页                   |
| 测试 Provider      | 发起一次 SSO 认证测试，查看字段映射和原始用户信息        |
| 删除 Provider      | 删除 Provider，并级联删除该 Provider 下的绑定关系        |

Provider 测试不会登录、绑定或创建账号，只用于确认 IdP 配置和字段映射是否正确。

1. 进入管理员控制面板
   ![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626141447861.png)

2. 进入“身份验证”控制面板，选择协议并创建后可选择是否启用，后续可在“操作”中进行测试、编辑、删除
   ![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626141407249.png)

3. 已启用的 IdP 将在登录页面显示，点击后即可启动 SSO 认证
   ![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626140437886.png)

## 六、常见异常

| 现象                               | 常见原因                                                                        |
| ---------------------------------- | ------------------------------------------------------------------------------- |
| 登录页没有 SSO 入口                | Provider 未启用，或登录入口未展示                                               |
| 返回地址不匹配                     | IdP 中配置的 Redirect URI、ACS 或 EntityID 与 SwanLab 实际外部地址不一致        |
| 能完成 IdP 认证但 SwanLab 登录失败 | 用户字段映射错误，或第三方用户 ID 为空                                          |
| OIDC 登录失败                      | Issuer/Discovery 不可访问，scope 缺少 `openid`，ID Token 校验失败，nonce 不匹配 |
| SAML2 登录失败                     | Metadata 不可访问，证书/私钥不匹配，ACS/EntityID 错误，Assertion 缺少映射属性   |
| 自动创建账号失败                   | 用户名不合法、用户名冲突、License 席位不足                                      |
