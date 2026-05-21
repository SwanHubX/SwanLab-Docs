FROM node:20.19.0
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
RUN echo "依赖安装完成..."
COPY . .

# 注入环境变量到构建阶段
ARG UMAMI_WEBSITE_ID
ENV UMAMI_WEBSITE_ID=${UMAMI_WEBSITE_ID}

RUN echo '开始build'
RUN npm run docs:build
RUN echo '---build 完成---'

FROM nginx:alpine

RUN echo '拷贝dist到 nginx目录'
COPY --from=0 /app/.vitepress/dist /usr/share/nginx/html
COPY --from=0 /app/nginx.conf /etc/nginx/conf.d/default.conf
