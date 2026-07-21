# Using Skill in AI Agents

- 🔨 [Skill Repository](https://github.com/SwanHubX/SwanLab-Skill)

:::tip
SwanLab Skill requires SDK version >= 0.8; some `swanlab api` commands (e.g. `run series`) require SDK >= 0.9.0. Make sure your SDK meets these requirements.
:::

SwanLab-Skill enables AI agents like Claude Code, Codex and OpenCode to automatically log experiment data and query/analyze experiment results on the SwanLab platform. It covers two use cases:

- Writing training tracking code with the Python SDK (`swanlab.init` / `swanlab.log` / `swanlab.finish` + multimedia logging)
- Querying experiment metrics, logs, summaries and media with the `swanlab api` CLI

The agent first reads the corresponding capability reference based on the task, then generates code or runs queries, avoiding API misuse.

## Installation

### Install via Agent (Recommended)

If you are using a coding agent like Claude Code or Codex, simply send it the following message to complete the installation automatically:

```text
Fetch the installation guide and follow it: https://raw.githubusercontent.com/SwanHubX/SwanLab-Skill/main/README.md
```

### Manual Installation

Global installation is recommended:

:::code-group

```bash [npm]
npx skills add SwanHubX/SwanLab-Skill -y -g
```

```bash [bun]
bunx skills add SwanHubX/SwanLab-Skill -y -g
```

:::

> `npx skills` is a utility for installing skills into AI agent CLIs. Use `-g` for a global install (installed into `.agents/skills` under your home directory, reusable across multiple agent CLIs for the same user). See the [skills docs](https://github.com/vercel-labs/skills) for details.

You can also install via SkillHub or ModelScope:

- 🐧 SkillHub: [swanlab-skill](https://skillhub.cn/skills/swanlab-skill)
- 🤖 ModelScope: [swanlab-skill](https://www.modelscope.cn/skills/SwanLab/swanlab-skill)

To update SwanLab Skill later, run:

```bash
npx skills update swanlab-skill -g -y
```

### Log in to SwanLab

Before using, make sure your API Key is saved:

```bash
pip install swanlab
swanlab login          # paste your API key from https://swanlab.cn
swanlab ping           # (optional) check connectivity
swanlab verify         # (optional) validate credentials
```

## Capabilities

| Capability                                                                      | Method                            |
| ------------------------------------------------------------------------------- | --------------------------------- |
| Log metrics and media objects (image / audio / text) during training/finetuning | Python SDK                        |
| View a run's metrics, logs, summary, columns and media                          | `swanlab api run ...` CLI         |
| Filter experiments by config or summary conditions                              | `swanlab api run filter` CLI      |
| Manage projects, self-hosted users and other resources                          | `swanlab api project / user` CLI  |
| Plot scalar metric curves for a **single** run                                  | Helper script `plot_metrics.py`   |
| **Compare** the same metric across multiple runs (normalized + ranking)         | Helper script `runs_benchmark.py` |
