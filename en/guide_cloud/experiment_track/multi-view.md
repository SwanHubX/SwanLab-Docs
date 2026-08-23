# Multi-View

Create multiple independent dashboards within a single project. Each view saves its own filters, grouping, and chart layouts — without affecting the others.

Multi-View is built on the new dashboard architecture. **How to upgrade**: open the `project` you want to upgrade and click **"Upgrade to New Dashboard"** in the **top-right corner**. New projects created via the web or the new SDK automatically use the Multi-View dashboard — no manual upgrade needed.

**📊 Upgrading an Existing Project to the New Dashboard**:
<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260720174505317.png"/>

::: warning Before You Upgrade

- ⚠️ Note:
  - The upgrade duration depends on the number of experiments and charts in the project — please be patient
  - Experiments in progress cannot be upgraded
  - **The upgrade cannot be rolled back to the old dashboard** — please confirm carefully before proceeding!

**💥 Breaking Changes**:

- After upgrading to the new dashboard, you must upgrade the SDK to `v0.9.0+` to create experiments, resume experiments, and log metrics
- Projects not yet upgraded can still log metrics and create experiments with SDK versions earlier than `v0.9.0`; however, SDK `v0.9.0+` cannot log metrics to or create experiments in projects on the old dashboard
- The old dashboard will no longer be maintained after **October 1, 2026** — we recommend upgrading as soon as possible
  :::

[[toc]]

## What is Multi-View?

When tuning models, you might want to watch every wiggle of the loss curve, compare final metrics across a dozen experiments, and curate a clean set of result charts for a report — all at the same time. Cramming every chart into a single dashboard gets messy fast.

Multi-View solves this: **one project, multiple dashboards**. Create views for different analysis scenarios — each view remembers its own chart configuration, so switching views means switching perspectives in one click.

<img src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/images/20260820114444509.png"/>

**What Multi-View helps you do:**

- **Cover different analysis needs**: create dedicated views such as "Experiment Monitoring", "Model Comparison", and "Result Reproduction"
- **Switch analysis scenarios instantly**: no need to repeatedly adjust filters and layouts — just switch views
- **Avoid configuration conflicts**: changes in the current view never disturb what you've already organized in other views
- **Improve team collaboration**: fewer conflicts when multiple people adjust dashboards, keeping everyone's analysis environment stable

## Default Views and Custom Views

When creating a view, choose between two types:

- **Default view**: automatically renders charts for all logged metrics — ideal for quick validation and phases where you want a full picture of your data
- **Custom view**: starts from a blank canvas where you hand-pick and configure the metric charts you care about — ideal when you have too many metrics, or need to focus on core comparisons during hyperparameter tuning

<video controls src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/videos/01-demo.mp4"></video>

::: warning Note
A newly created custom view is a blank canvas — charts need to be added manually.
:::

## Personal Views and Public Views

Views support two visibility scopes — choose based on personal use or team collaboration:

- **Personal view**: visible only to you — organize it freely to match your own workflow
- **Public view**: visible to all project collaborators — ideal as a shared team dashboard

<video controls src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/videos/02-pub-pri.mp4"></video>

## Managing Views

A project can have multiple views, with convenient management operations:

- **Reorder**: drag to sort, keeping frequently used views within easy reach
- **Rename**: rename anytime so each view's purpose is clear at a glance
- **Pin**: pin your most important views to the front
- **Duplicate**: quickly create a copy of an existing view and fine-tune from there

<video controls src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/videos/03-self-def.mp4"></video>

## Chart Layout Sync

Within the same view, the layouts of **multi-experiment comparison charts** and **single-experiment charts** are bidirectionally synced — any adjustment to charts or groups is mirrored between the two. "Edit once, apply everywhere", keeping the experience consistent across every dimension.

<video controls src="https://swanlab-docs-1301372061.cos.ap-beijing.myqcloud.com/assets/videos/04-sync-layout.mp4"></video>
