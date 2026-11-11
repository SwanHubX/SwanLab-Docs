FROM swanlab-registry-vpc.cn-hangzhou.cr.aliyuncs.com/builder/node:20-bookworm AS builder
WORKDIR /app

# Override with --build-arg NPM_REGISTRY=https://registry.npmjs.org/ when
# building from networks where the China mirror is undesirable.
ARG NPM_REGISTRY=https://registry.npmmirror.com/

# pnpm 版本与 package.json 的 packageManager 字段保持一致。
COPY package.json pnpm-lock.yaml ./

# Install pnpm via the configured registry (the builder image ships npm only).
# npm_config_registry applies to both the pnpm bootstrap install and
# `pnpm install` itself: the pnpm v9 lockfile stores no tarball URLs for
# registry packages, so resolution follows the configured registry. Integrity
# hashes in the lockfile still pin every tarball, so mirror-served content
# is verified.
RUN npm config set registry "${NPM_REGISTRY}" && \
    npm i -g pnpm@12.15.0

ENV npm_config_registry=${NPM_REGISTRY}

# pnpm install --frozen-lockfile installs strictly per pnpm-lock.yaml
# (reproducible, skips resolver work). The BuildKit cache mount persists the
# pnpm store across builds.
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile

COPY . .

ARG UMAMI_WEBSITE_ID
ENV UMAMI_WEBSITE_ID=${UMAMI_WEBSITE_ID}

# VitePress SSR rendering plus build-time plugin scans of this doc set exceed
# Node's default ~2GB heap. Raise it.
ENV NODE_OPTIONS=--max-old-space-size=4096

RUN npm run docs:build

FROM swanlab-registry-vpc.cn-hangzhou.cr.aliyuncs.com/builder/nginx:alpine AS prod
COPY --from=builder /app/.vitepress/dist /usr/share/nginx/html
COPY --from=builder /app/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
