---
title: 【新闻翻译】Neoforge-网络重构
date: 2024-02-01 12:23:02
tags:
- Minecraft
- 我的世界
- java
cover: https://view.moezx.cc/images/2022/02/24/8a20d0b70c60d53715bfbbf32f464784.png
---

# 引言

## 重构网络

欢迎阅读NeoForge重做网络的博客文章，这些更改在NeoForge 20.4.70-beta及更高版本中可用。文章将描述NeoForge的更改，以启用由Fabric的维护者Modmuss50设计的配置网络协议，并进行了一些小的修改。我将在下面分享更多相关信息。

## SimpleChannel和EventChannel

Forge有两种不同的方法来实现自定义网络通道。一种简单的基于注册的方法称为SimpleChannel，而一种为每个接收到的数据包触发事件的系统称为EventChannel。为了简化API并使与新数据包布局的交互更加容易，决定将这两种实现重构成一个单一的系统，结合两者的优点。

# 实现

## 新的网络有效载荷处理

该系统基于 Mojang 使用的 CustomPacketPayload 定义来表示自定义数据包的内容。在内部，他们将主要用于调试的自定义有效载荷注册到一个映射表中。我们需要扩展这个映射表，以便模组制作者可以发送和接收 CustomPacketPayload 的自定义实现。这是重写工作的大部分内容。模组制作者可以通过在 RegisterPayloadHandlerEvent 期间注册新的 CustomPacketPayload 实现来引入它们。

## 注册器

任何模组制作者都可以为他们想要的任何命名空间请求一个注册器；然而，建议一个模组只为其命名空间请求注册器。一旦从事件中获取了注册器，就可以使用两种不同的选项进行配置：versioned(String version) 以配置调用后注册的所有有效载荷的版本，以及 optional() 将调用后注册的所有有效载荷标记为不需要接收端。注册器配置的示例可以在这里找到:
```java
@SubscribeEvent
public static void register(final RegisterPayloadHandlerEvent event) {
    final IPayloadRegistrar registrar = event.registrar("my_mod")
            .versioned("1.2.3")
            .optional();
}
```

> 注意： 
注册器是一个半不可变对象；在对注册器实例调用 versioned(String version) 或 optional() 时，将创建一个具有所需配置的新实例。

> 警告: 
一旦事件的范围被离开，注册器就会失效。如果在事件处理范围之外注册有效载荷处理程序，将导致系统不知道这些有效载荷，并且不会通过连接发送它们。此外，尝试在事件范围之外注册它们将触发一个异常。

注册器提供了六个不同的端点，三对两个，用于注册新的有效载荷。一对用于游戏阶段的连接，一对用于特殊配置子阶段，还有一对方法适用于两者

> 信息：
无法注册应该在连接的登录阶段发送的自定义有效载荷。因此，新代码没有提供实现这一功能的基础设施。

对于每个阶段的注册方法对，有两种变体（因此，共有六种方法）。一种为连接的两端注册相同的处理程序，另一种接受一个消费者，允许配置单方面处理或差异化处理的负载。

配置阶段方法的签名示例如下：

```java
<T extends CustomPacketPayload> IPayloadRegistrar configuration(ResourceLocation id, FriendlyByteBuf.Reader<T> reader, IConfigurationPayloadHandler<T> handler);

<T extends CustomPacketPayload> IPayloadRegistrar configuration(ResourceLocation id, FriendlyByteBuf.Reader<T> reader, Consumer<IDirectionAwarePayloadHandlerBuilder<T, IConfigurationPayloadHandler<T>>> handler);
```

对于游戏阶段，存在类似的注册方法。对于应该在与配置阶段同时发送的有效载荷，也存在一对方法。然而，在这里，处理程序是处理回调的常见超类型，它具有两个不同类型可用信息的减少的超集。

## 有效载荷区分 

在考虑这个系统时，你可能会问，系统是如何将不同的有效载荷类型彼此区分开的。在调用有效载荷的写入方法之前，系统会向连接写入一个区分符ID。类似地，在客户端，首先读取区分符，然后查找一个读取器，以便读取有效载荷的剩余部分。
写入时区分符的值是从 CustomPacketPayload#id() 方法中获取的，且不能为 null。从连接中读取的 id 值与注册器作为第一个参数给出的值进行比较，以找到读取器。因此，注册器和有效载荷实例的 id() 方法接收相同的资源位置至关重要。

> 提示:
我们建议您在 public static final ResourceLocation ID = new ResourceLocation(“mod_id”, “payload_id”) 字段中存储您的 ID，并在两个地方引用它。

> 警告:
由于 ID 被用作区分符，您使用一个唯一的值非常重要，尤其是对于每种有效载荷类型的路径。如果您尝试两次注册相同的 ID，注册器将抛出异常。如果您尝试注册一个与注册器所在命名空间不同的 ID，注册器将抛出异常。您可以自由地为其他命名空间请求注册器，而不仅仅是您自己的命名空间。

### 有效载荷读取

有效载荷读取是通过实现 FriendlyByteBuf.Reader 函数式接口的 vanilla 方法进行的。在注册有效载荷类型时，如上所述，您需要传递这个接口的一个实现，以便当带有这种有效载荷的自定义有效载荷数据包到达接收端时，系统可以创建一个有效载荷的新实例。


> 由于我们建议您的有效载荷实现使用 Java 的记录（record）而不是类，我们也建议您创建一个自定义构造函数，以便从缓冲区读取记录字段。这个构造函数可以作为方法引用传递给读取器实现。因此，如果 SimplePayload(String something) {} 是您的正常记录，那么在 SimplePayload 记录中添加 SimplePayload(FriendlyByteBuf buf) { this(buf.readUtf()); } 作为构造函数，将允许您在注册器请求 FriendlyByteBuf.Reader 的实现时，将其作为方法引用 SimplePayload::new 传递给它。

> 没有关于读取或写入回调将在哪个线程上调用的保证。因此，需要注意的是，如果相同的包同时由许多连接处理，该方法可能会在许多线程上并行调用。

### 有效载荷写入

CustomPacketPayload 接口包含一个方法：write(FriendlyByteBuf)。当需要将您的有效载荷写入网络连接时，会调用这个方法。关于哪个线程调用写入器，没有任何保证。

> 与有效载荷读取一样，关于写入回调将在哪个线程上调用，也没有任何保证。因此，需要注意的是，如果相同的包同时发送到许多连接，该方法可能会在许多线程上并行调用。

> 只有当通过连接发送时，才会读取和写入有效载荷。这意味着单玩家世界的主机（即使是暴露给局域网）在内存中有数据包和有效载荷的传输。这意味着对于这些有效载荷，不会调用写入方法，也不会调用读取器。只有处理程序会被调用！

### 有效载荷处理

一旦有效载荷被写入、传输并读取，就会调用有效载荷处理程序。这个处理程序再次使用有效载荷的 id 查找，然后以接收端的上下文调用。每个处理程序接收两个参数：有效载荷实例和上下文。

> 有效载荷在网络线程上处理，因此可以与正在处理的同一类型的其他有效载荷并行发生。如果您需要确保有效载荷在主线程上按顺序处理，请参阅上下文中的 workHandler() 方法下可用的 ISynchronizedWorkHandler。

### 上下文

上下文包含信息、回调函数和入口点，用于访问周围的网络系统、主线程，以及处理其他数据包或完成配置任务。

### 回复处理器（ReplyHandler）

回复处理器可以用来快速将有效载荷发送回发送者。例如，它可用于发送查询数据包的答案，或发送确认已接收并处理了有效载荷的信息。您仍然需要注册返回的有效载荷。

### 数据包处理器（PacketHandler）

如果您实现了一个数据包分割机制，无论是基于完整的 vanilla 数据包，还是基于自定义数据包有效载荷，IPacketHandler 接口都允许您访问处理管道的起点，使您能够立即处理其他有效载荷。

> 这不会传输有效载荷，它纯粹允许接收端在处理您的有效载荷时处理内存中构建的其他数据包或有效载荷。

数据包处理器还提供了一个 disconnect(Component) 方法，允许您终止连接，并向用户显示给定的组件作为断开连接的原因。

### 工作处理器（WorkHandler）

工作处理器允许您在接收端的主线程上调度工作。如果逻辑接收端是客户端，这可能是 Minecraft 类实例，如果是服务器端，则可能是 MinecraftServer 实例。

### 数据包流（packetFlow）

上下文将通过数据包流指示接收端当前的状态。如果是服务器流向，那么处理程序当前正在服务器上下文中被调用。如果是客户端流向，那么处理程序当前正在客户端上下文中被调用。

### 连接协议（ConnectionProtocol）

当前活动的连接协议在您的有效载荷中包含数据包的原始字节时很有用，它允许您的处理程序在将数据包或有效载荷传递给 IPacketHandler 处理之前解码内部数据包或有效载荷。

### ChannelHandlerContext

当前正在处理有效载荷的 Netty 通道处理上下文也被作为上下文提供。这个上下文可以用来通过 ConnectionUtils 检索底层的原始连接，或者用来处理内部数据包和有效载荷的原始字节。

### Player
还提供了一个包含玩家的 Optional 对象。如果处理程序在服务器端被调用，那么这个玩家就是发送有效载荷的玩家。如果处理程序在客户端被调用，那么这是本地玩家（如果可用）。

### Level
还提供了一个包含玩家所在level的 Optional 对象。

### 任务完成处理器（TaskCompletedHandler）
这是一个特殊的环境变量，仅在配置阶段对有效载荷可用，它指示特定的配置任务已经完成，可以开始下一个任务。

### 未来对上下文的添加

我们完全意识到，作为模组制作者，您可能需要处理数据包的所有信息。通常，扩展接口和记录非常简单。它们特别设计成允许通过简单的 PR（Pull Request）在未来添加内容，所以请毫不犹豫地创建一个快速的 PR 来添加您需要的数据到上下文中。


## 数据包发送

与通道注册机制的重构相辅相成，我们添加了新的工具和系统，使您能够更轻松地将自定义有效载荷发送到不同的目标

## 数据包分发器（PacketDistributor）

这个包装类现在有能力单独处理自定义有效载荷。由于其实例和目标是不可变的，所以它们可以被传递。扩展类上的几个方法将接受这些实例，以方便有效载荷的轻松传递。


## 扩展对象

我们扩展了几个 vanilla 类型，使它们也能接受有效载荷，而不仅仅是数据包。例如区块部分、监听器、实体和玩家。

## Netty 信息
我们存储了许多与连接相关的信息，例如在连接对象本身上协商的有效载荷类型。因此，我们添加了几个属性来存储这些信息。这些属性被视为内部 API，您自行承担使用风险。

# 客户端加入时的配置任务

## 任务（Tasks）
Vanilla 现在提供了一种集中式的方法来执行玩家加入时需要执行的任务和作业。在这些任务完成之前，不会实例化玩家或将其添加到世界中。
在正常情况下，这些任务是实现 ConfigurationTask 接口的实例，该接口有一个方法：start(Consumer<Packet<?>>)。然而，在我们的情况下，这并不理想。模组制作者实际上不应该接触原始数据包来执行他们的任务，而应该只处理有效载荷。因此，决定让模组制作者实现 NeoForge 的 ICustomConfigurationTask 接口。这为 ConfigurationTask 签名提供了一个包装器，并通过实现 run(Consumer) 来允许发送有效载荷而不是数据包。然后，给定的消费者将自动将有效载荷转换为数据包并发送给正在配置的客户端。

> 实际上，ICustomConfigurationTask 的一个实例也是 ConfigurationTask 的一个实例，因为一个扩展了另一个。但是，为了使其成为一个功能接口，start 方法默认是实现的。您不应该覆盖它。

## OnGameConfigurationEvent 事件

该事件用于收集所有应该运行的任务，并允许向监听器注册 ICustomConfigurationTask 实例。无法注册 Vanilla 的 ConfigurationTask 实例。

> 该事件在模组总线上触发，以保持依赖顺序。鉴于配置任务只能按注册顺序运行，您可以安全地假设，在您的配置任务运行之前，您依赖的配置任务已经运行。

## NeoForge 数据包更改：

将配置同步、注册表同步和层级注册表同步移到了配置阶段任务。


## 数据包捆绑处理

在 1.19.4 版本中，Mojang 引入了数据包捆绑系统，这是一个核心组件，允许数据包一起处理。我们预计模组制作者可能想要使用这个系统，因此我们对其进行了调整，使其能够接受自定义有效载荷。您会在 ServerGamePacketListener 类上找到一个 sendBundled(CustomPacketPayload… payloads) 方法。

> 数据包捆绑仅在网络协议的游戏阶段支持。它不能在协议的配置阶段使用。

## 使用自定义数据打开菜单

过去，NeoForge 支持从服务器端以附加数据打开 UI，通过 NetworkHooks.openScreen(…)。这个系统已经被移动，现在是服务器 ServerPlayer 扩展的一部分。您可以使用相同参数调用 openMenu 方法。


## 使用自定义数据生成实体
以前的网络实现允许通过 Entity 类中的可覆盖方法 getAddEntityPacket 生成自定义实体。模组制作者如果想支持在实体生成时，客户端处理自定义附加数据，可以覆盖这个方法，并使用 NetworkHooks.getEntitySpawningPacket(…) 从方法中返回一个数据包。

这个系统现在已经被重构（因为 Mojang 使用捆绑包生成和处理实体数据包）。这个新框架的核心是 Entity 类上的方法 sendPairingData(ServerPlayer, Consumer)。现在有兩種方法来配置这个功能。

## 使用自定义有效载荷

通过覆盖该方法并调用包含自定义有效载荷的消费者，您可以确保您的有效载荷在生成数据包之后立即处理。您可以自由地做任何您想做的事情，但是我们建议您至少传递实体 id，就像 Vanilla 一样，以便在您的数据包到达时检索实体实例。

## 使用 IEntityWithComplexSpawn 接口

在您的实体上实现这个接口会强制您实现两个方法：writeSpawnData 和 readSpawnData。这些方法分别在生成实体捆绑包时和实体已经生成后被调用。

## 在 EntityType 中移除自定义实体创建代码

现在无法在客户端使用上述提到的实体生成数据包代码来创建不同的实体类。这是因为有效载荷机制的重构，以及对 Vanilla 生成捆绑包的依赖。

# NeoForge 数据包分割器

NeoForge 数据包分割器现在可以用于任何数据包（除了分割数据包本身）。

# 模组列表传输
目前，新协议不支持在网络上发送模组列表，并且只在工作通道和注册表内容基础上运行。然而，我们确实打算与 Fabric 团队合作，以扩展协议，使得服务器能够了解客户端安装了哪些类型的模组。我们认为这是必要的，例如，允许服务器所有者阻止允许作弊的模组，如 XRay 模组等。这将在进一步审查我们的立场和实现可能性后，在另一个 PR 中添加。

# 更多文档
更多关于使用网络功能的文档可以在文档的 [Networking](https://docs.neoforged.net/docs/networking/)部分找到。



