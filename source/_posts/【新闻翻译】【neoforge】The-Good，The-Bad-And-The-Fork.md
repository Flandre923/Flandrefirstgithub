---
title: 【新闻翻译】【neoforge】The Good，The Bad...And Fork
date: 2024-02-01 15:00:32
tags:
- 我的世界
- neoforge
cover: https://view.moezx.cc/images/2022/02/24/d12b859c2d57190aaf6fe882a610afcd.png
---


# 2023: The Good, The Bad... and The Fork

现在，我们即将步入2024年，作为NeoForged团队，是时候回顾过去6个月里发生的事情了。

## The Fork

让我们先来谈谈房间里的大象：我们究竟是谁？

NeoForge 是 Minecraft Forge 项目的分支。它于 2023 年 7 月 12 日（提前）正式宣布，原 Forge 团队中几乎所有人都加入了 NeoForged，只有一个明显的例外：LexManos。

导致分支的原因有很多，最明显的是团队（尤其是triage团队）与管理层之间的分歧。一些次要的原因包括有自由去使用以及重构大部分内部结构和相关的基础，从模组API到模组加载系统。这些变化与 Forge 有着明显的分歧，Forge 历来对某些领域的变更持严格立场，通常反对大规模的重构，因此从一个“更干净”的起点开始会更容易。

## The Good

我们的团队已增加 6 名成员，但我们仍在寻找更多成员，例如存储库的维护人员。 （有兴趣吗？请填写我们的申请表进行 [申请](https://links.neoforged.net/apply).)

自fork以来，我们一直致力于改进代码库的多个领域。例如，我们重写了 [Gradle 插件](https://github.com/NeoForged/NeoGradle)，现在默认提供对 [Parchment](https://github.com/neoforged/neogradle?tab=readme-ov-file#apply-parchment-mappings) 的支持。

因此，我们对 NeoForge API 进行了各种更改：

- 我们的第[一次重构](https://github.com/neoforged/Bus/milestone/1?closed=1)受到了 [EventBus](https://github.com/NeoForged/Bus)（我们的 EventBus 的分支）的影响（从好的意义上来说），改进了它的性能和内部结构，以及一些面向修改者的好处，例如防止（意外地）监听抽象类；
- 在 1.20.2 版本中，我们进行了第一次重大改革：注册系统得到了显著的改进（[#257](https://github.com/neoforged/NeoForge/pull/257)），简化了其内部结构，并与原版游戏的一致性得到了提升；
- 自 1.20.2 版本起，我们默认集成了 [MixinExtras](https://github.com/LlamaLad7/MixinExtras)（[#303](https://github.com/neoforged/NeoForge/pull/303)），以便模组制作者能够编写更兼容的混合代码；
- 在过渡到 1.20.3 版本时，我们重新设计了能力系统（[#73](https://github.com/neoforged/NeoForge/pull/73)），作为我们的第二次重大改革，将其分为数据附件和 API 提供者，并解决了旧系统的一些长期存在的问题（例如，旧系统不支持 Block 能力。是的，你现在可以使用你喜欢的管道模组从大锅提取流体了）；

1.20.4 版本也充满了变化：

- 增加了一个测试框架（[#291](https://github.com/neoforged/NeoForge/pull/291)），使得测试 Neo 功能的过程更加直接，并且与 Mojang 的 GameTest 系统很好地集成在一起，从而提高了平台的长期稳定性。这个框架将来会对模组开放；
- 引入了一种方式，让模组可以标记自己与其他模组的不兼容性（[#397](https://github.com/neoforged/NeoForge/pull/397)）；
- 对 /neoforged generate 这个区块预生成命令进行了各种改进，尤其是性能方面的改进（[#364](https://github.com/neoforged/NeoForge/pull/364)）。（对于不了解的人来说，服务器管理员可以使用这个命令来预先生成区块，这样游戏进行时就不会同时让所有玩家触发世界生成了。）我们要感谢 Jasmine 和 Gegy 允许我们使用他们的模组作为新命令的基础；
- 我们建立了一个 [Crowdin 项目](https://crowdin.neoforged.net/)，您可以在其中提交不同语言的翻译（并请求新语言进行翻译）；以及
- 我们的第三次重大改革：[网络重构。](https://github.com/neoforged/NeoForge/pull/277)

我们还在 ModLauncher 及其姐妹项目上进行了多项性能改进，包括减少启动时间的工作。

这些变化还伴随着一些主要的基础设施变化：

- 我们还引入了一个系统，用于将[PR发布到GitHub Packages](https://github.com/neoforged/NeoForge/pull/429)，这将使用户（和模组制作者）在变更正式合并到每个人使用的版本之前更容易进行测试。
- 我们的大多数（如果不是全部，也只有一个例外）项目现在都是用GitHub Actions构建的。
- 我们目前正在计划在不久的将来对我们的基础设施进行彻底检修，转向一个新的服务器设置。

在Discord服务器方面，我们放宽了关于核心模组和旧版本的规定；你现在可以自由地讨论并获得这些内容的支持。此外，讨论其他加载器的内容不再被禁止——甚至受到鼓励，只要它是富有成效的。毕竟，NeoForge 并不存在于真空中，所有的API都有其客观的缺点和优点。

## 采用

尽管现在得出结论还为时尚早（因为1.20.1实际上是1.20生命周期中模组包的事实目标），但我们看到越来越多的模组从1.20.4开始使用NeoForge。
我们也很高兴地报告，[CurseForge](https://www.curseforge.com/download/app)，[Modrinth](https://modrinth.com/app)，[Prism](https://prismlauncher.org/) 和 [FTB](https://www.feed-the-beast.com/) 都在他们的启动器中添加了对NeoForge的支持，对此我们深表感激！

## 来看看一些统计数据 - 以经典的Wrapped风格

NeoForged显然还没有存在满一年，但这是我们自2023年公开离开以来173天的一些统计数据：

- 我们平均每个月提供约一个太字节（1000千兆字节）的Maven工件，仅在12月就有1000万次请求。
- 已经合并了超过200个PR。这意味着平均每天都有一个PR被合并！
- 已经解决了超过95个问题，标记为已完成。这意味着每两天就有一个问题被解决！

## The bad

然而，世界并不总是粉红色和光明的。我们犯过错误，我们对此承担责任。

我们为Discord服务器的突然、混乱和令人困惑的重新命名道歉。由于一些不值得深入讨论（也不相关于此帖）的原因，我们不得不在我们预期之前数月就公开，所以我们措手不及。这是一团糟，如果我们能够回到过去并做得更好，我们会这么做。

我们还为没有提供我们承诺的稳定的1.20.1环境道歉。在1.20.2发布后，我们忽视了1.20.1，并且至今仍然如此。在1.20.1上的开发努力用在Forge上会比用在NeoForge上更有价值。

文档（或者说文档的缺乏）一直是Forge的一个问题，我们也面临着整个工具链和API文档的迫切缺乏。在过去的几个月里，我们进行了一些更改，同时让我们的现有文档逐渐过时。在2024年，我们将关注改善这个敏感但重要的项目领域。

我们不得不以这样或那样的方式找到我们的步伐，但最终，我们很抱歉让大家看到我们的争吵和争论。我们希望在来年从我们的错误中学习和改进。

## So… what’s next?

我们对2024年的计划不多，但一些更紧迫的事项包括：

- 随着所有重构工作的完成，可以预期在2024年的前两周内，甚至更早，将发布一个稳定的1.20.4版本；
- 对FML的重构——我们试图简化的复杂性怪兽——已经拖延了很久。你可以在[FML Clean-up Discord线程](https://discord.com/channels/313125603924639766/1187879036815417456)中跟进进度并提供你的想法；
- 我们正在与Mumfrey合作，希望让Mixin摆脱长达两年的停滞状态。如果这一努力没有成功，我们将对其他替代方案持开放态度；
- 我们正在努力改进NeoGradle的缓存，以减少构建时间；
- 预计1.21将通过TelepathicGrunt的努力通过统一的[PR](https://github.com/neoforged/NeoForge/pull/135)统一NeoForge和Fabric之间的标签命名空间；
- 如我们之前提到的：文档，文档，更多的文档！
- 对传输（IItemHandler、IFluidHandler、IEnergyStorage）能力的某些潜在变化也正在[Transfer rework线程](https://discord.com/channels/313125603924639766/1183818213134446742)中讨论；
- 用基于Java的核心模组替换JavaScript核心模组也正在考虑中，可以在[Coremod changes线程](https://discord.com/channels/313125603924639766/1105595318197825557/threads/1155582283839983658)中查看。

一如既往，我们感谢您的投入，如果您能在我们的[Discord服务器](https://discord.neoforged.net/)或[GitHub讨论区](https://github.com/neoforged/NeoForge/discussions)提供反馈或想法，我们将不胜感激。

# …and thanks for all the fish

现在，是你们都期待已久的有趣部分：致谢！

如果没有团队成员与我们一同离开Forge，NeoForge是不可能实现的，为此我们感谢他们。
我们感谢在过去6个月为我们的众多项目做出贡献的所有贡献者，我们衷心感谢您的支持——我们的成就是多亏了社区反馈和那些给予我们犯错并反弹机会的人，让我们变得越来越强大。
祝大家2024年快乐，一如既往地，快乐的移植！ 🎉
