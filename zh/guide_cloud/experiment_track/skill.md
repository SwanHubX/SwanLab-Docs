# 使用 Skill 自动分析实验

- 🔨 [Skill 仓库](https://github.com/SwanHubX/SwanLab-Skill)

:::tip
SwanLab Skill 功能需要 SDK 版本 >= 0.8 才能使用；其中 `swanlab api` 的部分命令（如 `run series`）需要 SDK >= 0.9.0。请确保你安装的 SDK 版本满足要求。
:::

SwanLab-Skill 可以让 Claude Code、Codex、OpenCode 等 AI Agent 工具自动记录实验数据并查询分析实验结果。它覆盖两类用法：

- 用 Python SDK（`swanlab.init` / `swanlab.log` / `swanlab.finish` + 多媒体记录）写训练跟踪代码
- 用 `swanlab api` CLI 查询实验指标、日志、摘要与媒体

Agent 会先按任务路由读取对应的能力说明，再生成代码或执行查询，避免误用接口。

## 安装

### 通过 Agent 安装（推荐）

如果你正在使用 Claude Code、Codex 等 coding agent，直接将下面这段话发给它即可自动完成安装：

```text
Fetch the installation guide and follow it: https://raw.githubusercontent.com/SwanHubX/SwanLab-Skill/main/README.md
```

### 手动安装

推荐使用全局安装方式：

:::code-group

```bash [npm]
npx skills add SwanHubX/SwanLab-Skill -y -g
```

```bash [bun]
bunx skills add SwanHubX/SwanLab-Skill -y -g
```

:::

> `npx skills` 是一个用于在 AI Agent CLI 中安装 Skill 的工具，使用 `-g` 参数可全局安装（安装到用户目录下的 `.agents/skills`，同一个用户下的多个 Agent CLI 可复用）。详见 [skills 文档](https://github.com/vercel-labs/skills)。

你也可以通过 SkillHub 或 ModelScope 安装：

- 🐧 SkillHub: [swanlab-skill](https://skillhub.cn/skills/swanlab-skill)
- 🤖 ModelScope: [swanlab-skill](https://www.modelscope.cn/skills/SwanLab/swanlab-skill)

后续如果 SwanLab Skill 有更新，执行：

```bash
npx skills update swanlab-skill -g -y
```

### 登录 SwanLab

使用前请确保已保存 API Key：

```bash
pip install swanlab
swanlab login          # paste your API key from https://swanlab.cn
swanlab ping           # (optional) check connectivity
swanlab verify         # (optional) validate credentials
```

## 能力

| 能力                                                              | 方式                             |
| ----------------------------------------------------------------- | -------------------------------- |
| 训练 / 微调中记录指标与媒体等对象资源文件（image / audio / text） | Python SDK                       |
| 查看某个 run 的指标、日志、摘要、列与媒体                         | `swanlab api run ...` CLI        |
| 按 config 或 summary 条件筛选实验                                 | `swanlab api run filter` CLI     |
| 管理项目、私有化用户等资源                                        | `swanlab api project / user` CLI |
| 画**单个**实验的标量指标折线图                                    | 辅助脚本 `plot_metrics.py`       |
| **对比**多个实验的同一指标（归一化 + 排名）                       | 辅助脚本 `runs_benchmark.py`     |
