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

You need to install Node.js and pnpm in advance. For detailed instructions, please refer to the [pnpm tutorial](https://pnpm.io/installation).

Use the following commands to install other dependencies:

```bash
pnpm install
```

**3. Run**

If you are performing local development or previewing documents, you can run the following in the project root directory:

```bash
pnpm docs:dev
```

If a complete compilation and packaging is required, use the following command:

```bash
pnpm docs:build
pnpm docs:preview
```
