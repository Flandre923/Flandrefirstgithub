---
title: 02-neoforge-Event system changes in NeoForge 20.2
date: 2023-11-25 09:54:49
tags:
- 我的世界
- neoforge
cover: https://view.moezx.cc/images/2022/02/24/ddda99f39eb9f46d4d727a910703e431.png
---

# Event system changes in NeoForge 20.2(NeoForge 20.2 中的事件系统更改)

## Introduction 介绍

在过去的几周里，我们一直致力于更新我们的event system。这篇博文将详细介绍所做的所有更改，作为更新到 NeoForge 20.2 的模组制作者的迁移指南。

请注意，这篇文章不涵盖特定事件，而是对事件机制本身所做的更改。

## Key changes 主要变化

### Package change 包变化

根包从 `net.minecraftforge.eventbus` 更改为 `net.neoforged.bus` 。因此，API 现在位于 `net.neoforged.bus.api` 中。

迁移示例：

```
- import net.minecraftforge.eventbus.api.EventBus;
+ import net.neoforged.bus.api.EventBus;
```

如果您错过了它，可以使用一个重新映射脚本来应用所有包更改。[Renaming script for the class renames introduced in the 20.2 NeoForge release (github.com)](https://gist.github.com/Technici4n/facbcdf18ce1a556b76e6027180c32ce)

### Cancellable event changes (可取消的活动变更)

可取消事件现在应该实现 `ICancellableEvent` 而不是使用 `@Cancelable` 注释：

```diff
- @Cancelable
- public class MyEvent extends Event {
+ public class MyEvent extends Event implements ICancellableEvent {
      // Your event code
  }
```

使用 `setCanceled(true)` 取消事件，使用 `isCanceled()` 检查事件是否被取消。这并没有改变。

`post` 现在返回已发布的事件，而不是事件是否被取消。您可以对结果调用 `isCanceled()` 来实现之前的行为：

```diff
- if (NeoForge.EVENT_BUS.post(new MyEvent())) {
+ if (NeoForge.EVENT_BUS.post(new MyEvent()).isCanceled()) {
      // Do something if the event was canceled
  }
```

### 更新了 `@SubscribeEvent` 语义

我们更改了有关当对象或类 `register` 被添加到事件总线时如何检测 `@SubscribeEvent` 方法的一些详细信息：

将对象或类注册到新行为的新行为如下：

- 目标对象中的所有 `@SubscribeEvent` 方法（无论可见性如何）均已注册。不再有 `@SubscribeEvent` 方法的继承，现在可以使用私有方法。
- 注册类的超类或超接口不允许有任何使用 `@SubscribeEvent` 声明的方法。这可以防止开发人员认为继承有效的错误。
- 如果任何 `@SubscribeEvent` 方法的静态性不匹配，则会抛出错误： `Class` 的注册必须是 `static` ，并且对象的注册必须是非 `static` 。这可以防止 `static` 被遗忘或不必要的错误。
- 如果不存在 `@SubscribeEvent` 方法，则会抛出错误。这可以防止忘记 `@SubscribeEvent` 注释。

### `abstract` 事件无法再监听

无法再监听 `abstract` 事件。这应该有助于防止开发人员意外监听超类的错误，例如监听 `SomeEvent` 而不是 `SomeEvent.Pre` 。 `abstract` 事件的所有超类现在都必须是抽象的。

将许多 NeoForge 事件处理为 `abstract` 来防止开发人员犯错误。

### 更新了 mod 总线语义

Forge 总线将不再允许监听器实现 `IModBusEvent` 的事件。这应该可以防止订阅错误的事件总线。

此外，通过 `ModLoader` 在 mod 总线上分派的事件（例如所有 NeoForge 注册事件）现在尊重不同总线之间的事件优先级。 （例如，使用 `EventPriority.LOW` 注册的侦听器将始终在使用 `EventPriority.NORMAL` 注册的其他 mods 的侦听器之前运行。）

我们为 `IEventBus#addListener` 的 lambda 注册添加了一些方便的重载。例如，现在可以进行以下操作：

```java
bus.addListener(SomeEvent.class, event -> {
    // Listener code here.
});
```

### Generic events 已弃用并删除

Generic events 已经是删除的废弃状态，并将在将来删除 [1](https://neoforged.net/news/20.2eventbus-changes/#fn:1) 。我们鼓励模组制作者不再使用他们。 NeoForge 仍然仅将它们用于 `AttachCapabilitiesEvent` 。我们将在功能重做中解决这个问题。

### Event results 正在逐步淘汰

目前，仅弃用删除 `@Event.HasResult` 注释。我们最终将删除 `getResult()` 和 `setResult(result)` 方法，但是 NeoForge 中的许多事件仍然依赖于它们。

如果您对某些事件使用此注释，我们鼓励您改用自定义 `enum` 类型，因为它们对于 API 用户来说更清晰。

如果您仅使用 `getResult` 和 `setResult` 方法，则无需执行任何操作。

## Other changes 其他变化

- 删除子类转换器：以前，事件子类的无参数构造函数必须是 `public` 。现在情况已不再如此 - 如果您愿意，您现在可以创建这样的构造函数 `protected` 、包私有或 `private` 。

- `Event#getPhase` 和 `Event#setPhase` 已删除。
- 现在，在调用 `Event#setResult` 时会检查 `@Event.HasResult` 。 `Event#hasResult` 和 `Event#getResult` 现已最终确定。
- `EventListenerHelper` 已从 API 中删除。
- `EventListener` 使用 `toString` 进行人类可读的描述（ `listenerName` 已删除）。
- 性能改进：自动删除不可取消事件的 `isCanceled` 检查。
- `IEventListener` 被重命名为 `EventListener` 并出于性能原因更改为抽象类。
- ModLauncher 挂钩被删除，大大简化了事件总线的实现。
-  `IEventBusInvokeDispatcher` 已被删除。



