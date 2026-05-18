# swanlab-docs

This repository regroups documentation and information that is hosted on the SwanLab website.

You can access the SwanLab documentation at [Document](https://docs.swanlab.cn). Powered by [vitepress](https://vitepress.dev/zh/guide/getting-started).


### How to contribute to the docs

It's very simple! Just clone the project, add or modify Markdown files, commit them, and then create a PR.

### Previewing locally 

**1. Clone this repository**

```bash
git clone https://github.com/SwanHubX/SwanLab-Docs
cd SwanLab-Docs
```

**2. Install**

You need to install nodejs and npm in advance. For detailed instructions, please refer to the [Node.js tutorial](https://nodejs.org/en/download/package-manager).

Use the following commands to install other dependencies:

```bash
npm add -D vitepress
npm install
```

**3. Run**

If you are performing local development or previewing documents, you can run the following in the project root directory:

```bash
npm run docs:dev
```

If a complete compilation and packaging is required, use the following command:

```bash
npm run docs:build
npm run docs:preview
```