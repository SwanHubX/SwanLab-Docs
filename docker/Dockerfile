# syntax=docker/dockerfile:1.4

FROM swanlab-registry-vpc.cn-hangzhou.cr.aliyuncs.com/builder/node:20-bookworm AS builder
WORKDIR /app

# Override with --build-arg NPM_REGISTRY=https://registry.npmjs.org/ when
# building from networks where the China mirror is undesirable.
ARG NPM_REGISTRY=https://registry.npmmirror.com/

COPY package.json package-lock.json ./

# npm ci installs strictly per package-lock.json (reproducible, skips resolver
# work); its replace-registry-host=npmjs default rewrites registry.npmjs.org
# URLs inside the lockfile to ${NPM_REGISTRY}, so no lockfile edit is needed.
# The BuildKit cache mount persists downloaded tarballs across builds.
RUN --mount=type=cache,target=/root/.npm \
    npm config set registry "${NPM_REGISTRY}" && \
    npm ci

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
