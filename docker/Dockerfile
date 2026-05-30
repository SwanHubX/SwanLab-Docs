FROM swanlab-registry-vpc.cn-hangzhou.cr.aliyuncs.com/builder/node:20-bookworkm AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm install
COPY . .

ARG UMAMI_WEBSITE_ID
ENV UMAMI_WEBSITE_ID=${UMAMI_WEBSITE_ID}

RUN npm run docs:build

FROM swanlab-registry-vpc.cn-hangzhou.cr.aliyuncs.com/builder/nginx:alpine AS prod
COPY --from=builder /app/.vitepress/dist /usr/share/nginx/html
COPY --from=builder /app/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
