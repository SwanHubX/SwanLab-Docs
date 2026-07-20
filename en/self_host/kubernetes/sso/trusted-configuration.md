# Configure TRUSTED Login and Third-Party Platform Integration as an Administrator

This document describes how to configure a TRUSTED Provider in a self-hosted SwanLab environment and how a third-party platform can issue short-lived JWTs to initiate SwanLab login.

## 1. Feature Introduction

TRUSTED is a simplified login method that SwanLab provides for trusted third-party platforms. It is intended for scenarios where a third-party platform does not provide OAuth2, OIDC, or SAML2 IdP capabilities but can authenticate users itself and sign JWTs with an asymmetric key.

TRUSTED login is initiated by the third-party platform:

1. The user logs in to the third-party platform.
2. The third-party platform's backend issues a short-lived, single-use JWT for the current user.
3. The third-party platform submits the JWT to SwanLab Auth in a new window.
4. SwanLab verifies the JWT with the public key configured in the Provider and reads the third-party user ID and username.
5. A third-party user with an existing binding is logged in directly. Otherwise, SwanLab automatically creates an account, establishes the binding, and logs the user in.

TRUSTED differs from other SSO protocols in the following ways:

- It is available only in self-hosted environments.
- It supports login only, not a separate account-binding flow or Provider testing.
- It does not appear in SwanLab's public list of login methods.
- It does not call the SSO Redirect endpoint. Instead, the third-party platform submits the JWT directly to the TRUSTED callback.
- The third-party platform cannot specify `state`, `action`, or the SwanLab callback URL.

## 2. Prerequisites

Before you begin, make sure that:

- You have deployed a self-hosted version that supports TRUSTED.
- You have configured `global.settings.host` according to the [documentation](https://docs.swanlab.cn/en/self_host/kubernetes/configuration.html#global-configuration-global).
- The third-party platform's backend can store the JWT private key securely.
- The clocks on the third-party platform and SwanLab servers are synchronized.
- SwanLab is served externally over HTTPS.

### 2.1 Configure the SwanLab External URL

After TRUSTED login succeeds, Auth redirects the user to SwanLab's `/sso` page. This address must come from the self-hosted deployment configuration and cannot be read from the third-party request.

Configure [`global.settings.host`](https://docs.swanlab.cn/en/self_host/kubernetes/configuration.html#global-configuration-global) in the `values.yaml` used by the self-hosted deployment scripts:

```yaml
global:
  settings:
    host: https://swanlab.example.com
```

Set `global.settings.host` to the external gateway URL that users enter in their browsers to access SwanLab:

- Include `http://` or `https://`. HTTPS is required in production.
- Do not include `/api/auth`, `/sso`, or the TRUSTED callback path.
- This setting does not automatically create or modify gateway forwarding rules. Make sure the corresponding domain already points to the SwanLab gateway.
- After changing the setting, make sure it is synchronized to the relevant services.

Examples:

```text
Correct: https://swanlab.example.com
Incorrect: https://swanlab.example.com/api/auth
Incorrect: https://swanlab.example.com/sso
```

## 3. Configure a TRUSTED Provider

### 3.1 Generate Signing Keys

TRUSTED uses asymmetric signing:

- The third-party platform stores the private key and uses it to sign JWTs.
- The SwanLab Provider stores the public key and uses it to verify JWTs.

We recommend using a 2048-bit RSA key with the `RS256` algorithm. You can use OpenSSL to generate a PKCS#8 private key and the corresponding public key:

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

After generating the keys:

- Store `trusted-private.pem` only in the third-party platform backend's key management system.
- Enter `trusted-public.pem` in the SwanLab Provider's JWT public key field.
- Do not store the private key in a code repository, frontend environment variable, browser storage, or logs.

TRUSTED also supports the following algorithms:

- RSA: `RS256`, `RS384`, and `RS512`
- ECDSA: `ES256`, `ES384`, and `ES512`

HMAC algorithms that use shared secrets, such as `HS256`, are not supported.

### 3.2 Create the Provider

1. Sign in with a SwanLab administrator account, open the Authentication management page, and select the option to create a `TRUSTED` Provider.

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260720153228495.png)

2. Fill in the basic configuration.

| Field        | Required | Description                                                                                                                                                                           |
| ------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Name         | Yes      | Unique Provider identifier. Up to 25 characters. Only letters, digits, underscores, and hyphens are allowed. This value appears in the callback URL; avoid changing it after creation |
| Display name | Yes      | Name shown in Provider management and the login flow. It is used only for administration and is not displayed as a public login button                                                |

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260720153433757.png)

3. Fill in the TRUSTED configuration.

| Field                           | Required | Description                                                                                                                             |
| ------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Third-party platform identifier | Yes      | Stable identifier for the third-party platform. It corresponds to the JWT `iss` claim, and the two values must match exactly            |
| JWT public key                  | Yes      | RSA or ECDSA PEM public key used to verify JWT signatures. It must include the complete `BEGIN PUBLIC KEY` and `END PUBLIC KEY` markers |

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260720154820031.png)

4. Configure user field mapping.

User field mapping determines which JWT payload fields SwanLab uses to read the third-party user's identity.

| Field          | Required | Recommended value | Description                                                                                                                       |
| -------------- | -------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| User ID field  | Yes      | `sub`             | JWT field that contains the third-party user's stable, unique ID                                                                  |
| Username field | Yes      | `username`        | JWT field that contains the third-party username, used as the default username when creating a SwanLab account for the first time |

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260720155100572.png)

With the recommended configuration, the JWT payload must contain at least:

```json
{
  "sub": "external-user-123",
  "username": "alice"
}
```

You can use other field names for the mapping. For example, with the following configuration:

```text
User ID field: employee_id
Username field: login_name
```

The JWT must contain:

```json
{
  "employee_id": "employee-123",
  "login_name": "alice"
}
```

The user ID must meet the following requirements:

- It must be permanently unique within the same Provider.
- It must remain unchanged if the user changes their name or email address, or moves to another organization.
- Do not use a display name that might be duplicated or changed as the user ID.

To avoid requiring users to change their usernames during their first login, the username mapped from the JWT must also:

- Contain only letters, digits, underscores, and hyphens.
- Be no longer than 25 characters.
- Not already be used by another SwanLab user.

If the username is invalid or already exists, SwanLab asks the user to change it in the new window before creating the account.

5. Enable the Provider.

After creating the Provider, switch its status to **Enabled** in the Provider list.

A TRUSTED Provider does not appear on the standard SwanLab login page and does not provide a Provider test button or login-entry logo configuration. Its sort value affects only the Provider order in the management list. The third-party platform must initiate login through the callback URL.

![](https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/images/20260720155318452.png)

## 4. Issue a JWT from the Third-Party Platform

### 4.1 Required JWT Content

Example JWT protected header:

```json
{
  "alg": "RS256",
  "typ": "JWT"
}
```

Example JWT payload:

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

Claim requirements:

| Claim          | Required | Description                                                                                                                              |
| -------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `iss`          | Yes      | Must exactly match the **Third-party platform identifier** configured in the SwanLab Provider                                            |
| `iat`          | Yes      | JWT issuance time as a Unix timestamp in seconds. It cannot be later than SwanLab's current time when the request arrives                |
| `exp`          | Yes      | JWT expiration time as a Unix timestamp in seconds. It must be later than `iat`, and `exp - iat` must be no greater than 300 seconds     |
| `jti`          | Yes      | Unique JWT identifier. It must be a non-empty string, and each JWT must use a new value. UUIDv4 is recommended to prevent replay attacks |
| User ID claim  | Yes      | Claim name is determined by the Provider's **User ID field**. Its value must identify the third-party user consistently                  |
| Username claim | Yes      | Claim name is determined by the Provider's **Username field**. Its value cannot be empty                                                 |
| `nbf`          | No       | If provided, it cannot be later than the current time on SwanLab Auth                                                                    |

We recommend setting the JWT lifetime to 60 seconds. The maximum issuance interval accepted by Auth is 300 seconds.

Each combination of Provider, third-party platform identifier, and `jti` can be used successfully only once. Submitting the same JWT again returns a credential-already-used error. If a request fails, issue a new JWT instead of reusing the old token.

### 4.2 Issue a JWT in the Third-Party Backend

Issue the JWT on the third-party platform. The following example uses Node.js and `jose`:

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

If the user information mapping fields configured for the Provider in SwanLab are not `sub` and `username`, update the field names in the `SignJWT` payload accordingly.

The third-party platform must issue a JWT only after confirming that the current third-party user is authenticated. Do not allow the browser to request a JWT by specifying an arbitrary third-party user ID or username.

## 5. Initiate Login from the Third-Party Platform

### 5.1 Callback URL

The TRUSTED callback URL uses the following format:

```text
https://<SwanLab external access address>/api/auth/sso/trusted/callback/<Provider name>
```

Example:

```text
https://swanlab.example.com/api/auth/sso/trusted/callback/partner-platform
```

Request requirements:

- The request method must be `POST`.
- The recommended Content-Type is `application/x-www-form-urlencoded`.
- The form field name must be `token`.
- The entire form request body must not exceed 16 KiB.
- Do not include the token in the URL query string.

### 5.2 Open SwanLab with a Hidden Form

> The following example demonstrates how to open SwanLab in a new window and log the user in automatically after they click a button on the third-party platform.

After obtaining a newly issued JWT, the third-party platform should synchronously open a blank window and then submit the token to that window through a hidden form. Opening the window first prevents the browser from treating it as a pop-up initiated after an asynchronous request and blocking it.

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
  // You must open the window with window.open.
  const popup = window.open("", target);

  if (!popup) throw new Error("The browser blocked the new window. Allow pop-ups for this site.");

  try {
    // This endpoint belongs to the third-party platform. It verifies the current user and issues a JWT.
    const response = await fetch("/api/swanlab/trusted-token", {
      method: "POST",
      credentials: "same-origin",
    });
    if (!response.ok) throw new Error("Failed to obtain SwanLab login credentials");

    const { token } = (await response.json()) as { token: string };
    submitToken(token, target);
  } catch (error) {
    popup.close();
    throw error;
  }
}
```

Recommended button flow:

1. The user clicks **Open SwanLab**.
2. The frontend immediately calls `window.open('', target)` to create a window.
3. The third-party platform issues a short-lived JWT for the current user.
4. After obtaining the JWT, the frontend submits it to the new window through a hidden form.
5. If token issuance or submission fails, the frontend closes the blank window and displays an error on the third-party platform.

> **Note**
>
> You must use `window.open` to open the new window. When login fails, SwanLab displays a **Close** button that automatically closes a SwanLab window opened with `window.open`.

Do not pass the token in the following way:

```text
https://swanlab.example.com/api/auth/sso/trusted/callback/partner-platform?token=<jwt>
```

URLs may be saved in browser history, access logs, proxy logs, and Referer headers.

### 5.3 Account Behavior After Login

SwanLab maintains the binding between the third-party user and SwanLab user based on the combination of **Provider + third-party user ID**:

- Existing binding: SwanLab creates a login session and takes the user directly into SwanLab.
- No existing binding: SwanLab creates a user with the mapped username, subject to seat availability, establishes the binding, and logs the user in.
- Invalid or duplicate username: SwanLab asks the user to change the username in the new window.
- Bound SwanLab user is disabled: SwanLab rejects the login and displays an account-disabled message.

After a successful login, SwanLab opens the user's workspace by default. Business errors are displayed in the new window. The user can close the window and initiate login again from the third-party platform.

## 6. Errors and Troubleshooting

| Error                                               | Meaning                                                                                                                             | Troubleshooting                                                                                                  |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Trusted third-party login configuration unavailable | The Provider does not exist, is disabled, does not use the TRUSTED protocol, or Auth cannot read its configuration                  | Check the Provider name in the callback, the Provider status, and the connection between SwanLab Server and Auth |
| Login credential invalid or expired                 | The token is missing, signature verification failed, the algorithm is unsupported, `iss` does not match, or time claims are invalid | Issue a new token and check the key pair, algorithm, `iss`, `iat`, `exp`, and server clocks                      |
| Login credential already used                       | The same `jti` has already been submitted successfully                                                                              | Generate a new `jti` and JWT for every login. Do not retry with the old token                                    |
| Unable to obtain complete user information          | The claims configured in the user field mapping are missing, empty, or use unsupported types                                        | Check the Provider mapping and JWT payload, and make sure the user ID and username are both non-empty strings    |
| Login service temporarily unavailable               | Auth cannot write the JWT consumption record, or Redis is temporarily unavailable                                                   | Check the Auth Redis connection. After service is restored, issue a new JWT                                      |

Common issues:

### 6.1 The Callback Returns 500

Check whether the self-hosted deployment's `values.yaml` contains the correct `global.settings.host` value and whether you reran the deployment or upgrade script after changing it. The value must match the external SwanLab gateway URL that users actually access.

### 6.2 The Token Is Reported as Invalid

Check the following items in order:

1. Verify that the JWT `alg` is one of the RSA or ECDSA algorithms supported by SwanLab.
2. Verify that the Provider public key matches the third-party signing private key.
3. Verify that the Provider's third-party platform identifier exactly matches the JWT `iss`.
4. Verify that `iat` and `exp` are Unix timestamps in seconds, not milliseconds.
5. Verify that `exp` is later than `iat` and no more than 300 seconds after it.
6. Verify that the clocks on the third-party server and SwanLab Auth are synchronized.
7. Verify that the JWT contains a non-empty `jti` that has not been used before.

### 6.3 The User Is Still Asked for a Username During First Login

Check whether the username mapped from the JWT:

- Contains only letters, digits, underscores, and hyphens.
- Is no longer than 25 characters.
- Is not already used by another SwanLab user.

If the third-party username cannot meet SwanLab's username requirements, the third-party platform can generate a stable, SwanLab-compatible username field when issuing the JWT and map the Provider's **Username field** to that claim.

### 6.4 Old Tokens Cannot Log In After the Public Key Is Changed

The Provider stores only one JWT public key. After the public key is updated, tokens signed with the old private key become invalid immediately. When rotating keys, coordinate the Provider configuration update with the third-party platform's signing-key switch, and allow old tokens to expire as soon as possible.

## 7. Security Considerations

- Deploy TRUSTED only in explicitly trusted, self-hosted environments.
- Store the private key only in the third-party platform backend or a dedicated key management system. Never send it to SwanLab or the browser.
- A JWT is a short-lived login credential. Protect it like a password and never log the complete token.
- A JWT is a bearer credential. The first party to submit it successfully continues the login as the user identified in the JWT. Issue it only after an authenticated user explicitly chooses to enter SwanLab.
- A 60-second JWT lifetime is recommended. The maximum issuance interval is 300 seconds.
- Generate a new `jti` for every JWT. A JWT cannot be reused after it is consumed.
- The third-party platform must confirm that the current user is authenticated before issuing a JWT. Do not trust user identity parameters submitted by the browser.
- Protect the third-party platform's JWT issuance endpoint with its own session authentication, CSRF protection, and rate limiting.
- Do not write private keys, JWTs, or sensitive user information to frontend logs, analytics platforms, or error-reporting systems.
- Enable HTTPS on both SwanLab and the third-party platform, and keep their system clocks synchronized.
- Map the Provider user ID to a stable, non-reusable third-party user identifier.
