# 管理员配置 TRUSTED 登录与第三方平台接入

本文说明如何在 SwanLab 私有化环境中配置 TRUSTED Provider，以及第三方平台如何签发短期 JWT 并发起 SwanLab 登录。

## 一、功能简介

TRUSTED 是 SwanLab 为受信第三方平台提供的简化登录方式，适用于第三方平台没有 OAuth2、OIDC 或 SAML2 IdP 能力，但能够自行认证用户并使用非对称密钥签发 JWT 的场景。

TRUSTED 登录由第三方平台发起：

1. 用户先在第三方平台完成登录。
2. 第三方平台后端为当前用户签发短期、一次性的 JWT。
3. 第三方平台通过新窗口向 SwanLab Auth 提交 JWT。
4. SwanLab 使用 Provider 中配置的公钥验证 JWT，并读取第三方用户 ID 和用户名。
5. 已绑定的第三方用户直接登录；未绑定用户自动创建 SwanLab 账号、建立绑定关系并登录。

TRUSTED 与其他 SSO 协议存在以下区别：

- 只用于私有化环境。
- 只支持登录，不支持绑定或 Provider 测试。
- 不会出现在 SwanLab 公共登录方式列表中。
- 不调用 SSO Redirect 接口，由第三方平台直接向 TRUSTED callback 提交 JWT。
- 第三方平台不能指定 `state`、`action` 或 SwanLab 回调地址。

## 二、使用前准备

开始配置前，请确认：

- 已部署支持 TRUSTED 的私有化版本。
- 已按照[文档](https://docs.swanlab.cn/self_host/kubernetes/configuration.html#%E5%85%A8%E5%B1%80%E9%85%8D%E7%BD%AE-global)完成 `global.settings.host` 配置。
- 第三方平台后端可以安全保存 JWT 私钥。
- 第三方平台和 SwanLab 服务器均已进行时间同步。
- SwanLab 使用 HTTPS 对外提供服务。

### 2.1 配置 SwanLab 外部访问地址

TRUSTED 登录完成后，Auth 需要跳转到 SwanLab 的 `/sso` 页面。该地址必须由私有化部署配置确定，不能从第三方请求中读取。

在私有化部署脚本使用的 `values.yaml` 中配置 [`global.settings.host`](https://docs.swanlab.cn/self_host/kubernetes/configuration.html#%E5%85%A8%E5%B1%80%E9%85%8D%E7%BD%AE-global)：

```yaml
global:
  settings:
    host: https://swanlab.example.com
```

`global.settings.host` 应填写用户通过浏览器访问 SwanLab 时使用的网关外部 URL：

- 应包含 `http://` 或 `https://`，生产环境必须使用 HTTPS。
- 不要填写 `/api/auth`、`/sso` 或 TRUSTED callback 路径。
- 该配置不会自动创建或修改网关转发规则，应确保对应域名已经正确指向 SwanLab 网关。
- 修改后保证配置同步到相关服务。

示例：

```text
正确：https://swanlab.example.com
错误：https://swanlab.example.com/api/auth
错误：https://swanlab.example.com/sso
```

## 三、配置 TRUSTED Provider

### 3.1 生成签名密钥

TRUSTED 使用非对称签名：

- 第三方平台保存私钥，用于签发 JWT。
- SwanLab Provider 保存公钥，用于验证 JWT。

推荐使用 RSA 2048 位密钥和 `RS256` 算法。可以使用 OpenSSL 生成 PKCS#8 私钥和对应公钥：

```bash
openssl genpkey \
  -algorithm RSA \
  -pkeyopt rsa_keygen_bits:2048 \
  -out trusted-private.pem

openssl pkey \
  -in trusted-private.pem \
  -pubout \
  -out trusted-public.pem
```

生成后：

- `trusted-private.pem` 只能保存在第三方平台后端的密钥管理系统中。
- `trusted-public.pem` 填写到 SwanLab Provider 的 JWT 公钥字段。
- 不要通过代码仓库、前端环境变量、浏览器存储或日志保存私钥。

TRUSTED 同时支持以下算法：

- RSA：`RS256`、`RS384`、`RS512`
- ECDSA：`ES256`、`ES384`、`ES512`

不支持 `HS256` 等使用共享密钥的 HMAC 算法。

### 3.2 创建 Provider

1. 使用 SwanLab 管理员账号进入“身份验证”管理页面，选择创建 `TRUSTED` Provider。

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260720153228495.png)

2. 填写基础配置

| 字段     | 是否必填 | 说明                                                                                                                  |
| -------- | -------- | --------------------------------------------------------------------------------------------------------------------- |
| 名称     | 是       | Provider 唯一标识，最多 25 个字符，只允许字母、数字、下划线和连字符。该值会出现在 callback URL 中，创建后建议不要修改 |
| 展示名称 | 是       | Provider 的管理和登录流程展示名称，不会作为公共登录按钮显示，仅为管理使用                                             |

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260720153433757.png)

3. 填写 TRUSTED 配置

| 字段         | 是否必填 | 说明                                                                                           |
| ------------ | -------- | ---------------------------------------------------------------------------------------------- |
| 三方平台标识 | 是       | 第三方平台的稳定标识，对应 JWT 的 `iss`，两者必须完全一致，否则后续校验将不通过                |
| JWT 公钥     | 是       | 用于验证 JWT 签名的 RSA 或 ECDSA PEM 公钥，应包含完整的 `BEGIN PUBLIC KEY` 和 `END PUBLIC KEY` |

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260720154820031.png)

4. 配置用户字段映射

用户字段映射决定 SwanLab 从 JWT payload 的哪些字段中读取第三方用户身份。

| 字段         | 是否必填 | 推荐值     | 说明                                                                  |
| ------------ | -------- | ---------- | --------------------------------------------------------------------- |
| 用户 ID 字段 | 是       | `sub`      | JWT 中第三方用户稳定且唯一的 ID 对应字段                              |
| 用户名字段   | 是       | `username` | JWT 中第三方用户名对应字段，首次自动创建 SwanLab 账号时作为默认用户名 |

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260720155100572.png)

使用推荐配置时，JWT payload 至少应包含：

```json
{
  "sub": "external-user-123",
  "username": "alice"
}
```

字段映射可以使用其他名称。例如配置：

```text
用户 ID 字段：employee_id
用户名字段：login_name
```

则 JWT 中必须包含：

```json
{
  "employee_id": "employee-123",
  "login_name": "alice"
}
```

用户 ID 应满足以下要求：

- 在同一个 Provider 中永久唯一。
- 用户改名、换邮箱或调整组织后仍保持不变。
- 不要使用可能重复或变化的展示名称作为用户 ID。

若希望首次登录时无需用户修改用户名，JWT 中映射出的用户名还应：

- 只包含字母、数字、下划线和连字符。
- 不超过 25 个字符。
- 未被其他 SwanLab 用户占用。

如果用户名不合法或已经存在，SwanLab 会在新窗口中要求用户修改用户名后再创建账号。

5. 启用 Provider

创建 Provider 后，在 Provider 列表中将状态切换为“已启用”。

TRUSTED Provider 不会出现在 SwanLab 普通登录页面，也不会提供 Provider 测试按钮或登录入口 Logo 配置。排序值只影响管理列表中的 Provider 顺序。第三方平台必须通过 callback URL 主动发起登录。

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260720155318452.png)

## 四、第三方平台签发 JWT

### 4.1 JWT 必填内容

JWT protected header 示例：

```json
{
  "alg": "RS256",
  "typ": "JWT"
}
```

JWT payload 示例：

```json
{
  "iss": "partner-platform",
  "sub": "external-user-123",
  "username": "alice",
  "iat": 1784000000,
  "exp": 1784000060,
  "jti": "e39f79d8-faf5-4da8-8fc5-f9115d2e9568"
}
```

字段要求：

| Claim         | 是否必填 | 说明                                                                                |
| ------------- | -------- | ----------------------------------------------------------------------------------- |
| `iss`         | 是       | 必须与 SwanLab Provider 的“三方平台标识”完全一致                                    |
| `iat`         | 是       | JWT 签发时间，Unix 秒级时间戳，不能晚于请求到达 SwanLab 时的当前时间                |
| `exp`         | 是       | JWT 过期时间，Unix 秒级时间戳，必须晚于 `iat`，并且 `exp - iat <= 300` 秒           |
| `jti`         | 是       | JWT 唯一标识，必须是非空字符串，每次签发都应生成新值，推荐使用 UUIDv4，防止重放攻击 |
| 用户 ID Claim | 是       | Claim 名称由 Provider 的“用户 ID 字段”决定，值必须能稳定标识第三方用户              |
| 用户名 Claim  | 是       | Claim 名称由 Provider 的“用户名字段”决定，值不能为空                                |
| `nbf`         | 否       | 如果提供，不能晚于 SwanLab Auth 当前时间                                            |

推荐将 JWT 有效期设置为 60 秒。Auth 接受的最大签发有效期为 300 秒。

同一 Provider、三方平台标识和 `jti` 组合只能成功使用一次。重复提交同一个 JWT 会返回凭证已使用错误；请求失败后需要重新签发新的 JWT，不能重用旧 token。

### 4.2 在第三方后端签发 JWT

在第三方平台签发 JWT。以下示例使用 Node.js 和 `jose`：

```bash
pnpm add jose
```

```ts
import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import { importPKCS8, SignJWT } from "jose";

const algorithm = "RS256";
const issuer = "partner-platform";

const privateKeyPEM = await readFile("./trusted-private.pem", "utf8");
const privateKey = await importPKCS8(privateKeyPEM, algorithm);

export async function issueSwanLabToken(user: { id: string; username: string }) {
  const now = Math.floor(Date.now() / 1000);

  return new SignJWT({
    sub: user.id,
    username: user.username,
  })
    .setProtectedHeader({ alg: algorithm, typ: "JWT" })
    .setIssuer(issuer)
    .setIssuedAt(now)
    .setExpirationTime(now + 60)
    .setJti(randomUUID())
    .sign(privateKey);
}
```

如果在 SwanLab 上配置身份验证方时使用的用户信息映射字段不是 `sub` 和 `username`，应同步修改 `SignJWT` payload 中的字段名称。

第三方只有在确认当前第三方用户已经登录后才能签发 JWT。不要允许浏览器任意指定第三方用户 ID 或用户名后请求签发。

## 五、第三方平台发起登录

### 5.1 Callback 地址

TRUSTED callback 地址格式：

```text
https://<SwanLab 外部访问地址>/api/auth/sso/trusted/callback/<Provider 名称>
```

示例：

```text
https://swanlab.example.com/api/auth/sso/trusted/callback/partner-platform
```

请求要求：

- 请求方法必须为 `POST`。
- Content-Type 推荐使用 `application/x-www-form-urlencoded`。
- 表单字段名必须为 `token`。
- 整个表单请求体不能超过 16 KiB。
- token 不应放在 URL query 中。

### 5.2 使用隐藏表单打开 SwanLab

> 以场景“点击第三方平台上的按钮后，在新窗口打开 SwanLab 并自动登录”为例，给出如下示例方案。

第三方平台获取刚签发的 JWT 后，应先同步打开空白窗口，再通过隐藏表单向该窗口提交 token。先打开窗口可以避免异步请求完成后被浏览器当作弹窗拦截。

```ts
function submitToken(token: string, target: string) {
  const provider = "partner-platform";
  const form = document.createElement("form");
  const input = document.createElement("input");

  form.method = "POST";
  form.action = `https://swanlab.example.com/api/auth/sso/trusted/callback/${encodeURIComponent(provider)}`;
  form.enctype = "application/x-www-form-urlencoded";
  form.target = target;
  form.hidden = true;

  input.type = "hidden";
  input.name = "token";
  input.value = token;

  form.append(input);
  document.body.append(form);

  try {
    form.submit();
  } finally {
    form.remove();
  }
}

export async function startSwanLabLogin() {
  const target = `swanlab-trusted-${crypto.randomUUID()}`;
  // 必须使用 window.open 打开
  const popup = window.open("", target);

  if (!popup) throw new Error("浏览器阻止了新窗口，请允许本站打开弹窗");

  try {
    // 该接口属于第三方平台，由第三方校验当前登录用户并签发 JWT
    const response = await fetch("/api/swanlab/trusted-token", {
      method: "POST",
      credentials: "same-origin",
    });
    if (!response.ok) throw new Error("获取 SwanLab 登录凭证失败");

    const { token } = (await response.json()) as { token: string };
    submitToken(token, target);
  } catch (error) {
    popup.close();
    throw error;
  }
}
```

推荐的按钮处理流程：

1. 用户点击“打开 SwanLab”。
2. 前端立即调用 `window.open('', target)` 创建窗口。
3. 签发当前用户的短期 JWT。
4. 获取成功后使用隐藏表单向新窗口提交 JWT。
5. 获取或提交失败时关闭空白窗口，并在第三方平台显示错误。

> 注意！
> 必须使用 `window.open` 打开新窗口，因为 SwanLab 在登录失败时提供“关闭”按钮，点击后自动关闭由 `window.open` 打开的 SwanLab 窗口。

不要使用以下方式传递 token：

```text
https://swanlab.example.com/api/auth/sso/trusted/callback/partner-platform?token=<jwt>
```

URL 可能被浏览器历史记录、访问日志、代理日志和 Referer 保存。

### 5.3 登录后的账号行为

SwanLab 根据“Provider + 第三方用户 ID”维护第三方用户与 SwanLab 用户的绑定关系：

- 已存在绑定：直接创建 SwanLab 登录态并进入 SwanLab。
- 不存在绑定：使用映射出的用户名创建 SwanLab 用户（席位不足时将提示失败），建立绑定关系后登录。
- 用户名不合法或重复：在 SwanLab 新窗口中要求用户修改用户名。
- 已绑定 SwanLab 用户被禁用：拒绝登录并显示账号已禁用。

登录成功后，SwanLab 默认进入用户空间。业务错误会在新窗口中显示对应信息，用户可以关闭窗口后从第三方平台重新发起登录。

## 六、错误说明与排查

| 错误                     | 含义                                                               | 排查建议                                                                     |
| ------------------------ | ------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| 受信第三方登录配置不可用 | Provider 不存在、未启用、协议不是 TRUSTED 或 Auth 无法读取配置     | 检查 callback 中的 Provider 名称、Provider 状态以及 SwanLab Server/Auth 连接 |
| 登录凭证无效或已过期     | token 缺失、签名失败、算法不支持、`iss` 不匹配或时间字段不符合要求 | 重新签发 token；检查公私钥、算法、`iss`、`iat`、`exp` 和服务器时间           |
| 登录凭证已经使用         | 相同的 `jti` 已经成功提交                                          | 每次登录生成新的 `jti` 和 JWT，不要重试旧 token                              |
| 无法获取完整用户信息     | 用户字段映射对应的 Claim 缺失、为空或类型不支持                    | 检查 Provider 映射和 JWT payload，确保用户 ID、用户名均为非空字符串          |
| 登录服务暂时不可用       | Auth 无法写入 JWT 消费记录或 Redis 暂时不可用                      | 检查 Auth Redis 连接，恢复后重新签发 JWT                                     |

常见问题：

### 6.1 提交后返回 500

检查私有化部署的 `values.yaml` 是否配置了正确的 `global.settings.host`，以及修改后是否已经重新执行部署或升级脚本。该值必须与用户实际访问 SwanLab 的网关外部 URL 一致。

### 6.2 提示 token 无效

依次检查：

1. JWT `alg` 是否属于 SwanLab 支持的 RSA/ECDSA 算法。
2. Provider 公钥是否与第三方签名私钥配对。
3. Provider 三方平台标识是否与 JWT `iss` 完全一致。
4. `iat`、`exp` 是否为 Unix 秒级时间戳，而不是毫秒时间戳。
5. `exp` 是否晚于 `iat`，且两者相差不超过 300 秒。
6. 第三方服务器与 SwanLab Auth 的系统时间是否同步。
7. JWT 是否包含非空且未使用过的 `jti`。

### 6.3 用户首次登录时仍需填写用户名

检查 JWT 中映射出的用户名是否：

- 只包含字母、数字、下划线和连字符。
- 不超过 25 个字符。
- 未被其他 SwanLab 用户使用。

如果第三方用户名不能满足 SwanLab 用户名要求，可以在第三方平台签发 JWT 时生成一个符合要求且稳定的 SwanLab 用户名字段，并将 Provider 的“用户名字段”映射到该 Claim。

### 6.4 修改公钥后旧 token 无法登录

Provider 当前只保存一份 JWT 公钥。更新公钥后，使用旧私钥签发的 token 会立即失效。轮换密钥时应协调 Provider 配置更新时间和第三方平台签名密钥切换时间，并让旧 token 尽快过期。

## 七、安全注意事项

- TRUSTED 只应部署在明确受信的私有化环境中。
- 私钥必须保存在第三方平台后端或专用密钥管理系统中，不得发送到 SwanLab 或浏览器。
- JWT 是短期登录凭证，应按密码同等级别保护，不得记录完整 token。
- JWT 是 Bearer Credential，首个成功提交者会以 JWT 中的用户身份继续登录；只能在已认证用户明确点击进入 SwanLab 后即时签发。
- JWT 推荐 60 秒过期，最大签发区间为 300 秒。
- 每次签发必须生成新的 `jti`，JWT 被消费后不能重用。
- 第三方平台必须在确认当前用户已经登录后签发 JWT，不能信任浏览器提交的用户身份参数。
- 第三方平台的 JWT 签发接口应使用自身的会话鉴权、CSRF 防护和限流策略。
- 第三方平台不得把私钥、JWT 或用户敏感信息写入前端日志、分析平台或错误上报系统。
- SwanLab 和第三方平台必须启用 HTTPS，并保持系统时间同步。
- Provider 的用户 ID 映射应使用稳定、不可复用的第三方用户标识。
