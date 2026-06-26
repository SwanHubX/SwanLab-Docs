# 使用 Skill 自动分析实验

- 🔨 [Skill 仓库](https://github.com/SwanHubX/SwanLab-Skill)

:::tip
SwanLab Skill功能需要 SDK 版本 >= 0.8 才能使用，请确保你安装的 SDK 版本满足要求。
:::

SwanLab-Skill 可以让 Claude Code、Codex、OpenCode 等 AI Agent 工具自动记录实验数据并查询分析实验结果。可通过以下方式安装：

:::code-group

```bash [npm]
npx skills add SwanHubX/SwanLab-Skill -y -g
```

```bash [bun]
bunx skills add SwanHubX/SwanLab-Skill -y -g
```

:::

> `npx skills` 是一个用于在 AI Agent CLI 中安装 Skill 的工具，使用 `-g` 参数可全局安装。详见 [skills 文档](https://github.com/vercel-labs/skills)。

你也可以通过 SkillHub 或 ModelScope 安装：

- 🐧 SkillHub: [swanlab-skill](https://skillhub.cn/skills/swanlab-skill)
- 🤖 ModelScope: [swanlab-skill](https://www.modelscope.cn/skills/SwanLab/swanlab-skill)

> ⚠️ 注意: 使用前请确保已通过 `swanlab login` 或 `SWANLAB_API_KEY` 环境变量保存 API Key。
