FROM node:20.9.0
WORKDIR /app
COPY package.json ./
RUN npm install -g pnpm && pnpm install
RUN echo "依赖安装完成..."
COPY . .

RUN echo '开始build'
RUN pnpm run docs:build
RUN echo '---build 完成---'

FROM nginx:latest

RUN echo '拷贝dist到 nginx目录'
COPY --from=0 /app/.vitepress/dist /usr/share/nginx/html
COPY --from=0 /app/nginx.conf /etc/nginx/conf.d/default.conf
