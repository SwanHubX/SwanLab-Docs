# swanlab-docs

本仓库托管了SwanLab的[官方文档](https://docs.swanlab.cn)，基于[vitepress](https://vitepress.dev/zh/guide/getting-started)。



### 如何为文档做贡献

很简单！只需要增添或修改Markdown文件，提交他们，创建一个PR就可以。



### 本地开发流程

1. 克隆本仓库

```bash
git clone https://github.com/SwanHubX/SwanLab-Docs
```



2. 安装环境

```bash
npm add -D vitepress
```



3. 本地开发，在项目根目录运行：

```bash
npm run docs:dev
```

4. 打包与预览

```bash
npm run docs:build
npm run docs:preview
```