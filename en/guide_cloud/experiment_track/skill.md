# Using Skill in AI Agents

- 🔨 [Skill Repository](https://github.com/SwanHubX/SwanLab-Skill)

:::tip
SwanLab Skill requires SDK version >= 0.8. Make sure your SDK meets this requirement.
:::

SwanLab-Skill enables AI agents like Claude Code , Codex and OpenCode to automatically log experiment data and query/analyze experiment results on the SwanLab platform.

## Installation

Install with a single command:

:::code-group

```bash [npm]
npx skills add SwanHubX/SwanLab-Skill -y -g
```

```bash [bun]
bunx skills add SwanHubX/SwanLab-Skill -y -g
```

:::

> `npx skills` is a utility for installing skills into AI agent CLIs. Use `-g` for a global install. See the [skills docs](https://github.com/vercel-labs/skills) for details.

You can also install via SkillHub or ModelScope:

- 🐧 SkillHub: [swanlab-skill](https://skillhub.cn/skills/swanlab-skill)
- 🤖 ModelScope: [swanlab-skill](https://www.modelscope.cn/skills/SwanLab/swanlab-skill)

> ⚠️ Note: Before using, make sure you have saved your API Key via `swanlab login` or the `SWANLAB_API_KEY` environment variable.
