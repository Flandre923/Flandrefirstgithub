---
title: NeoForge 20.3 for Minecraft 1.20.3 and 1.20.4 翻译
date: 2024-01-31 22:31:35
tags:
- 我的世界
- neoforge
cover: https://view.moezx.cc/images/2022/02/24/bf618023cd5ddad3ceb00fc11517fe55.png
---

# NeoForge 20.3 for Minecraft 1.20.3 and 1.20.4 

## 原文

[The Capability rework - The NeoForged project](https://neoforged.net/news/20.3capability-rework/)

> 注意：渣译，如果有错误请指正

随着Minecraft 1.20.4热修复版本的最近发布，我们将不再支持1.20.3版本，并鼓励所有模组开发者更新到NeoForge 20.4。此处提到的未来计划没有变化，但目标版本将调整为20.4而不是20.3。
Minecraft 1.20.3版本的NeoForge的第一个测试版，NeoForge 20.3.1-beta现已发布！请尝试使用它，与之互动，开发，并给我们反馈！我们目前还不稳定，所以在接下来的几周内可能会遇到一些重大变化。
对于玩家来说，您可以直接从https://neoforged.net/获取最新的安装程序。
这篇博客的其余部分是针对模组开发者的。让我们来谈谈模组开发者应该注意的NeoForge最近的更新，这些更新与迁移到20.3有关。Minecraft 1.20.3本身也带来了一些技术上的变化，但这些内容将不会在本帖中介绍。

## Capability 重做

能力系统重做 20.3版本中最有影响力的变化是能力系统的重做。您可以在我们专门的博客文章中阅读所有相关信息。

## 值得注意的近期新增功能

让我们讨论一些已经在20.2版本系列中提供，当然也在所有20.3版本中，但您可能尚未听说过的变化。

### mods.toml内的访问转换器 

现在，模组可以通过在它们的mods.toml中声明来包含多个访问转换器文件：

```toml

[[accessTransformers]]
file="modid_base.at"

[[accessTransformers]]
file="modid_extra.at"
```

如果不存在此类条目，FML将像以前一样回退到META-INF/accesstransformer.cfg。我们将在可预见的未来继续接受这两种格式。
此外，ATs仍然需要被指定，以便NeoGradle在您的开发环境中应用它们：

```groovy
minecraft {
    accessTransformers {
        file('src/main/resources/modid_base.cfg')
        file('src/main/resources/modid_extra.cfg')
    }
}
```

### MixinExtras ships with NeoForge

LlamaLad7的MixinExtras现在随NeoForge一同提供，并且会自动启用。现在您的build.gradle中不再需要有关MixinExtras的任何配置。
MixinExtras是Mixin的补充库，旨在帮助模组开发者以更具表达性和兼容性的方式编写他们的Mixin。如果您在您的模组中使用Mixin，我们鼓励您阅读MixinExtras的[维基](https://github.com/LlamaLad7/MixinExtras/wiki)。
如果您想要用较新的版本覆盖捆绑的MixinExtras版本，您可以按照更新前使用JiJ（即“jar in jar”）常规MixinExtras的方式，来JiJ较新的版本。

### mods.toml 内的 Mixin 配置

现在可以在您的mods.toml文件中直接指定Mixin配置，例如：

```toml
[[mixins]]
config="modid_base.mixins.json"

[[mixins]]
config="modid_extra.mixins.json"
```

我们建议在您的模组中使用这种新格式。现在使用Mixin不再需要Gradle支持，这为 未来选择性地启用Mixin配置打开了大门。

## 很快来到

在我们能够稳定20.3版本之前，我们将发布对我们网络钩子和协议的全面检修，以适应Mojang最近引入的配置阶段。如果您对讨论感兴趣，我们鼓励您加入我们的Discord服务器，并查看#brainstorming论坛频道。
像往常一样，我们也欢迎各种小型贡献。20.3的破坏性变更窗口将在接下来的几周内保持开放。现在是开始处理您的Pull Requests的好时机。

### 1.20.2计划 

从几天前发布的20.2.86版本开始，NeoForge的1.20.2版本现在已稳定。由于社区采用有限，我们将只接受关键错误修复的向后移植，前提是它们首先被提交并接受到1.20.3分支。
