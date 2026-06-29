# SwanLab SSO 前置操作说明

配置 SSO 前需要完成的环境、网络、权限和 IdP 准备工作。

## 一、确认外部访问地址

SSO 依赖浏览器在 SwanLab 与 IdP 之间跳转，因此 SwanLab 必须生成稳定、可被用户浏览器访问的外部 URL。

私有化部署建议在部署配置（[values.yaml](https://docs.swanlab.cn/self_host/kubernetes/configuration.html#%E5%85%A8%E5%B1%80%E9%85%8D%E7%BD%AE-global)）中显式设置：

```yaml
global:
  settings:
    host: https://swanlab.example.com
```

该地址应填写用户浏览器访问 SwanLab 的站点根地址，不需要手动追加任何路径：

| 用户访问地址                  | 推荐配置                      |
| ----------------------------- | ----------------------------- |
| `https://swanlab.example.com` | `https://swanlab.example.com` |

SwanLab 会基于该根地址生成以下协议地址：

```text
https://<SwanLab 外部地址>/api/auth/sso/...
```

如果未显式配置，系统会尝试通过请求信息推断外部地址。生产环境不建议使用默认推断，这可能会导致某些问题。

## 二、网络

### 2.1 HTTPS

建议所有 SSO 环境使用 HTTPS。很多 IdP 默认要求回调地址使用 HTTPS，浏览器 Cookie、安全跳转和企业网关策略也通常要求 HTTPS。
如果测试环境必须使用 HTTP，请先确认 IdP 是否允许 HTTP Redirect URI、ACS URL 或 Metadata URL。

### 2.2 连通性

SwanLab 服务端需要能访问 IdP 的相关地址：

| 协议   | SwanLab 服务端需要访问                                    |
| ------ | --------------------------------------------------------- |
| OAuth2 | Authorization Endpoint、Token Endpoint、Userinfo Endpoint |
| OIDC   | Issuer Discovery、Token Endpoint、JWKS 地址               |
| SAML2  | IdP Metadata URL、IdP SSO 地址                            |

如果私有化环境在内网，请确认 DNS、代理、防火墙、证书信任链都已配置完成。

当前 SSO 访问 IdP 的服务端请求超时时间为 10 秒。

## 三、服务器设置

OIDC 和 SAML2 都会校验令牌或断言的有效时间。请确保 SwanLab 服务端和 IdP 服务端时间同步，建议启用 NTP。
时间偏差过大可能导致：

- OIDC ID Token 过期或尚未生效。
- SAML Assertion 过期或尚未生效。
- SAML Audience/Conditions 校验失败。

## 四、准备用户字段

配置任何协议前，都需要先和 IdP 管理员确认以下字段：

| 字段              | 要求                              | 用途                                                          |
| ----------------- | --------------------------------- | ------------------------------------------------------------- |
| 第三方用户唯一 ID | 稳定、唯一、不可复用、不能为空    | 唯一标识某 IdP 中的第三方用户，重复则将导致绑定和自动创建失败 |
| 第三方用户名      | 建议稳定、符合 SwanLab 用户名规则 | 私有化 OIDC/SAML2 自动创建账号时作为默认用户名                |

不建议把姓名、昵称、部门名等可能重复或变化的字段作为用户唯一 ID。

推荐选择：

| 协议   | 用户唯一 ID 推荐字段                              | 用户名推荐字段                            |
| ------ | ------------------------------------------------- | ----------------------------------------- |
| OAuth2 | `id`、`uid`、`user_id`、`sub`                     | `username`、`login`、`preferred_username` |
| OIDC   | `sub`                                             | **`preferred_username`、`name`**          |
| SAML2  | `uid`、`employeeNumber`、`email` 等稳定 Attribute | `username`、`uid`                         |

> 注意：
>
> - 字段映射填写的是 IdP 返回数据中的字段名，不是某个用户的实际字段值。
> - 不建议使用邮箱作为用户名，特殊字符将导致校验失败

## 五、准备 Provider 基础信息

所有协议都需要填写以下基础字段：

| 序号 | 字段         | 是否必填 | 说明                                                                |
| ---- | ------------ | -------- | ------------------------------------------------------------------- |
| 1    | 名称         | 是       | Provider 唯一标识，最多 25 个字符，只允许字母、数字、下划线、连字符 |
| 2    | 展示名称     | 是       | 登录按钮展示的名称，最多 100 个                                     |
| 3    | 用户 ID 字段 | 是       | 第三方用户唯一 ID 对应的字段名                                      |
| 4    | 用户名字段   | 是       | 第三方用户名对应的字段名                                            |
| 5    | 图标 URL     | 否       | 登录入口展示的图标 URL                                              |
| 6    | 排序值       | 否       | 非负整数，数值越大越靠前                                            |

Provider 名称会出现在回调地址中。上线后修改名称，通常需要同步修改 IdP 中的回调地址或 SAML 配置。若无必要请保证名称始终不变。

### 5.1 OAuth2

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626144351999.png)

### 5.2 OIDC

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626144745051.png)

### 5.3 SAML2

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260626144834227.png)

## 六、配置变更后验证

如果修改了以下内容，建议在 SwanLab 管理后台重新保存 Provider，并再次使用测试功能确认配置是否生效：

- OIDC Issuer、Client、证书或 JWKS。
- SAML2 Metadata、证书、私钥、ACS、EntityID 或属性映射。
- SwanLab 外部访问地址、部署路径或 Provider 名称。

> SwanLab 后台对于 OIDC/SAML2 会缓存 IdP 信息，若 IdP 侧信息发生改变，SwanLab 不一定会及时同步相关更新，可以通过修改 SwanLab 中对应配置的任意字段刷新缓存。
