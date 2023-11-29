---
title: 01-neoforge-NeoForge 20.2 for Minecraft 1.20.2
date: 2023-11-25 09:31:21
tags:
- 我的世界
- Neoforge更新汉化
cover: https://view.moezx.cc/images/2022/05/31/06d9f9250d4e9b0cc72aae9625d53a4b.png
---

# NeoForge 20.2 for Minecraft 1.20.2 （NeoForge20.2支持我的世界1.20.2）

NeoForge for Minecraft 1.20.2 的第一个测试版现已发布。请尝试一下，使用它，使用它进行开发，并向我们提供反馈！我们还不稳定，因此预计未来几周会发生一些重大变化。

对于玩家来说，您可以直接从 https://neoforged.net/ 获取最新的安装程序。

对于模组作者来说，这篇博文的其余部分适合您。

## Versioning （版本）

NeoForge 版本今后将使用以下版本格式： `<minecraft_minor>.<minecraft_patch>.<number>(-beta)` 。这意味着 Minecraft 1.20.2 的所有版本都将格式化为 `20.2.*` 。 `-beta` 标签表示该版本不稳定，并且可能会在“number”版本之间发生重大更改。

## Toolchain Updates（ 工具链更新）

我们现在到处都在使用 MojMaps。这极大地简化了工具链、简化了 mod 构建、允许调试正在运行的 modpack 以及许多其他好处。

Gradle 插件（现在称为 NeoGradle）进行了大幅重写，为 mod 和 NeoForge 的开发人员提供了大幅加速，并带来了许多可用性优势。要迁移您的构建脚本，请查看更新的 MDK。

我们预计新的工具链将更加易于使用。如果您需要帮助，Discord 服务器上的 `#modder-support-1.20` 就是您所寻找的。像往常一样，请报告您可能发现的任何错误，以便我们修复它。

## NeoForge Changes （NeoForge 的变化）

所有包都已更改为 `net.neoforged` 。我们还在 NeoForge 本身中重命名了一些类。以下是这些重命名的完整列表。

如果您的 mod 是用 Java 编写的，您可以使用此重新映射脚本自动应用类和包重命名。

NeoForge 的新 modid 是… `neoforge` 。这意味着您必须更新 `mods.toml` 文件，以及可能仍引用 `forge` modid 的任何资源。

一个值得注意的例外是标签，它目前仍然位于 `forge` 命名空间下。我们正在与 Fabric 协调，提供两个加载程序共享的新约定，但这仅从 Minecraft 1.21 开始可用。

## Event System Changes（ 事件系统变更）

我们对事件系统进行了多项改进。您可以在我们的专门博客文章中阅读有关它们的所有信息。

## Minecraft Changes （我的世界的变化）

Minecraft 1.20.2 本身带来了一些变化。我们将来将提供入门/指南。

## Coming Soon （即将推出）

在稳定 20.2 之前，我们希望在未来几周内做出一些改变。

我们将重新设计我们的能力和注册系统，以解决长期存在的问题。如果您使用它们，您的模组可能会损坏。如果您对讨论感兴趣，我们鼓励您加入我们的 Discord 服务器并查看 `#brainstorming` 论坛频道。

我们还将调整我们的网络挂钩和协议，以适应 Mojang 引入的新配置阶段。

这些是我们的大型短期计划，但我们也欢迎各种较小的贡献。 NeoForge 的重大变更窗口现已开放，我们预计它会保持开放一段时间！现在是开始处理 Pull 请求的好时机。

## 1.20.1 Plans （1.20.1 计划）

我们的首要任务是为未来做好准备，并在未来几年尽可能地改进 NeoForge。这只能在最新的分支上进行。

然而，如果时间允许，我们仍然会合并针对 1.20.1 分支的 PR，前提是它们也提交到 1.20.2 分支并被 1.20.2 分支接受。

## Final Notes （最后的注释）

感谢为移植提供帮助的团队成员。感谢社区耐心等待此版本。感谢所有选择加入我们这次冒险的人。我们期待继续为您改进 NeoForge。
