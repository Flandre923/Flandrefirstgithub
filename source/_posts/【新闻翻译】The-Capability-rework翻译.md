---
title: The Capability rework翻译
date: 2024-01-31 22:51:06
tags:
- 我的世界
- Java
- neoforge
cover: https://view.moezx.cc/images/2019/10/22/idea1-bg.jpg
---

# 介绍

我们最初的[20.3版本]([NeoForge 20.3 for Minecraft 1.20.3 and 1.20.4 - The NeoForged project](https://neoforged.net/news/20.3release/))对功能系统进行了根本性的重新设计，目标是修复经过多年使用后在上一次迭代中发现的所有问题。

最重要的是，现在有两种不同的系统来取代以前所谓的“capabilities”：

- **Data attachments**允许将任意数据添加到block entities, chunks, entities, and item stacks.
- **Capabilities** 允许从blocks, entities, and item stacks中查询行为实例。

# Data attachments 

附加系统允许mods将任意数据对象附加到 block entities, chunks, entities, and stacks.

要使用该系统，您需要注册 `AttachmentType` 。附件类型包含：

- 一个默认值提供者，用于在第一次访问数据时创建实例，或者比较有数据的堆栈和没有数据的堆栈;

- 可选的序列化程序（如果附件应该持久化）;
- attachment的附加配置选项，例如 `copyOnDeath` 标志。

有几种方法可以提供附件序列化器：直接实现 `IAttachmentSerializer` ，实现 `INBTSerializable` 并使用静态 `AttachmentSerializer.serializable()` 方法来创建构建器，或者为构建器提供编解码器。(后一个选项不推荐用于itemstack，因为相对较慢）。

在任何情况下，我们建议使用 `DeferredRegister` 进行注册：

```java
// 创建用于attachment types的DeferredRegister
private static final DeferredRegister<AttachmentType<?>> ATTACHMENT_TYPES = DeferredRegister.create(NeoForgeRegistries.Keys.ATTACHMENT_TYPES, MOD_ID);

// 通过INBTSerializable进行序列化
private static final Supplier<AttachmentType<ItemStackHandler>> HANDLER = ATTACHMENT_TYPES.register(
        "handler", () -> AttachmentType.serializable(() -> new ItemStackHandler(1)).build());
// 通过编解码器进行序列化
private static final Supplier<AttachmentType<Integer>> MANA = ATTACHMENT_TYPES.register(
        "mana", () -> AttachmentType.builder(() -> 0).serialize(Codec.INT).build());
// 无序列化
private static final Supplier<AttachmentType<SomeCache>> SOME_CACHE = ATTACHMENT_TYPES.register(
        "some_cache", () -> AttachmentType.builder(() -> new SomeCache()).build()
);

//  别忘了将DeferredRegister注册到您的模组总线
ATTACHMENT_TYPES.register(modBus);
```

一旦注册了attachment type，就可以在任何holder object上使用它。如果没有数据，则调用 `getData` 将附加一个新的默认实例。

```java
// 如果已经存在，则获取ItemStackHandler，否则附加一个新的
ItemStackHandler stackHandler = stack.getData(HANDLER);
// 如果可用，获取当前玩家的mana，否则附加0
int playerMana = player.getData(MANA);
// And so on...
```

如果不需要附加默认实例，可以添加 `hasData` 检查：

```java
// 在进行任何操作之前，检查堆栈是否具有HANDLER附件
if (stack.hasData(HANDLER)) {
    ItemStackHandler stackHandler = stack.getData(HANDLER);
    //使用stack.getData(HANDLER)进行一些操作。
}
```

也可以使用 `setData` 更新数据：

```java
// Increment mana by 10.
player.setData(MANA, player.getData(MANA) + 10);
```

通常，block entities和chunks 在被修改时需要标记为脏（使用 `setChanged` 和 `setUnsaved(true)` ）。这是自动完成的调用 `setData` ：

```java
chunk.setData(MANA, chunk.getData(MANA) + 10); // will call setUnsaved automatically

```

但是如果你修改了从 `getData` 获得的一些数据，那么你必须显式地将 block entities 和chunks 标记为脏：

```java
var mana = chunk.getData(MUTABLE_MANA);
mana.set(10);
chunk.setUnsaved(true); // must be done manually because we did not use setData
```

在我们继续讨论capabilities之前，关于data attachment 系统，这里有几点需要注意：

- **Level attachments**已删除：请改用SavedData。

- 可序列化item stack attachments 现在始终与客户端同步。
- 当玩家从终点传送回来时，Entity attachments会被复制。（以前不是这样的）。
- 在构建器中设置了 `copyOnDeath` 的Entity attachments将在玩家死亡（以及怪物转换）时自动复制。

### attachments方面今后的工作

我们计划在未来几周内对附件系统进行以下改进：

- 配方JSON中的**Attachments** ：就像我们在配方结果中添加对count和NBT的支持一样，我们将在配方结果JSON中添加对指定data attachments的支持。
- 同步数据**attachments**：目前，所有可序列化的 item stack attachments 都自动从逻辑服务器同步到逻辑客户端。我们将在未来研究块 block entity, chunk, 和entity attachments 的选择同步。
- 自定义复制处理程序：目前，所有data attachments都是通过序列化到NBT，然后重新序列化新副本来复制的。这是一个很好的默认值，但我们希望允许moders提供自己的副本实现以获得更好的性能。

我们也欢迎其他建议，请不要犹豫，与我们联系！

# Capabilities 

Capabilities 旨在将block, entity or item stack 的功能与操作方式区分开来。如果您想知道capabilities 是否适合某项工作，请问自己以下问题：

1. 我是否只关心一个block, entity or item stack可以做什么，而不关心它是如何做的？
2. 是什么，行为，只适用于某些 blocks, entities, or item stacks，但不是所有的？
3. 该行为的实现方式是否依赖于特定的block, entity or item stack?

 下面是一些很好的能力使用的例子：

1. “我想数一数某个实体中有多少物品，但我不知道该实体如何存储它们。”- 是的，使用 `IItemHandler` 功能。
2. “我想把能量填充到某个物品堆里，但是不知道这个物品堆是怎么储存能量的。”- 是的，使用 `IEnergyStorage` 功能。
3. “我想为玩家当前瞄准的任何区块应用一些颜色，但我不知道该区块将如何转换。- 是的NeoForge不提供颜色块的功能，但你可以自己实现一个。

下面是一个不鼓励使用的功能的示例：

- “我想检查一个实体是否在我的机器范围内。”- 不，使用helper方法代替。

NeoForge支持blocks, entities, and item stacks的功能。

Capabilities 允许使用一些分派逻辑查找一些API的实现。NeoForge中实现了以下几种功能：

- `BlockCapability` ：blocks 和block entities的功能;行为取决于特定的 `Block` 。
- `EntityCapability` ：实体的功能：行为依赖于特定的 `EntityType` 。
- `ItemCapability` ：item stacks的功能：行为取决于特定的 `Item` 。

### Creating capabilities 

NeoForge已经定义了通用功能，我们建议与其他mod兼容。举例来说：

```java
// 标准物品处理程序方块能力
Capabilities.ItemHandler.BLOCK
// 标准物品处理程序物品能力
Capabilities.ItemHandler.ITEM

// 查看Capabilities类以获取完整列表。
```

如果这些还不够，您可以创建自己的功能。创建一个功能是一个单一的函数调用，结果对象应该存储在 `static final` 字段中。必须提供以下参数：

- 能力的名称。多次创建具有相同名称的功能将始终返回相同的对象。不同名称的功能是完全独立的，可以用于不同的目的。
- 正在查询的行为类型。这是 `T` 类型参数。
- 查询中其他上下文的类型。这是 `C` 类型参数。

例如，以下是如何声明side-aware block `IItemHandler` 的能力：

```java
public static final BlockCapability<IItemHandler, @Nullable Direction> ITEM_HANDLER_BLOCK =
    BlockCapability.create(
        // Provide a name to uniquely identify the capability.
        new ResourceLocation("mymod", "item_handler"),
        // Provide the queried type. Here, we want to look up `IItemHandler` instances.
        IItemHandler.class,
        // Provide the context type. We will allow the query to receive an extra `Direction side` parameter.
        Direction.class);
```

`@Nullable Direction` 对于块是如此常见，以至于有一个专用的helper：

```java
public static final BlockCapability<IItemHandler, @Nullable Direction> ITEM_HANDLER_BLOCK =
    BlockCapability.createSided(
        // Provide a name to uniquely identify the capability.
        new ResourceLocation("mymod", "item_handler"),
        // Provide the queried type. Here, we want to look up `IItemHandler` instances.
        IItemHandler.class);
```

对于entities 和item stacks，类似的方法分别存在于 `EntityCapability` 和 `ItemCapability` 中。

### Querying capabilities 

一旦我们在静态字段中有了 `BlockCapability` 、 `EntityCapability` 或 `ItemCapability` 对象，我们就可以查询一个能力。

Entities 和item stacks 基本上与以前的API相同，但返回类型为 `@Nullable T` 而不是 `LazyOptional<T>` 。只需使用capability对象和上下文调用 `getCapability` 即可：

```java
var object = entity.getCapability(CAP, context);
if (object != null) {
    // Use object
}
```

```java
var object = stack.getCapability(CAP, context);
if (object != null) {
    // Use object
}
```

Block capabilities 的使用方式不同，以适应没有 block entities的blocks 提供的capabilities 。在 `level` 上执行查询：

```java
var object = level.getCapability(CAP, pos, context);
if (object != null) {
    // Use object
}
```

如果 block entity 和/或block state 是已知的，则可以在查询时将它们传递给保存：

```java
var object = level.getCapability(CAP, pos, blockState, blockEntity, context);
if (object != null) {
    // Use object
}
```

为了给予一个更具体的例子，下面是如何从 `Direction.NORTH` 端查询块的 `IItemHandler` 能力：

```java
IItemHandler handler = level.getCapability(Capabilities.ItemHandler.BLOCK, pos, Direction.NORTH);
if (handler != null) {
    // Use the handler for some item-related operation.
}
```

### Block capability caching 

为了高效查询和自动缓存，请使用 `BlockCapabilityCache` 而不是直接调用 `level.getCapability` 。这是旧的 `LazyOptional` 失效系统的更强大的替代品。

查找功能时，系统将在后台执行以下步骤：

- 获取block entity和block state（如果未提供）
- 获取已注册的功能提供程序。(More在下面）。
- 迭代提供者并询问他们是否可以提供功能。
- 其中一个提供者将返回一个功能实例，可能会分配一个新对象。 

这个实现是相当高效的，但是对于频繁执行的查询，例如每个游戏时间点，这些步骤可能会占用大量的服务器时间。BlockCapabilityCache系统为在给定位置频繁查询的功能提供了显著的加速。

通常， `BlockCapabilityCache` 将被创建一次，然后存储在执行频繁功能查询的对象的字段中。何时存储该高速缓存取决于您。该高速缓存必须具有查询级别、位置和查询上下文的能力。	\

```java
// Declare the field:
private BlockCapabilityCache<IItemHandler, @Nullable Direction> capCache;

// Later, for example in `onLoad` for a block entity:
this.capCache = BlockCapabilityCache.create(
        Capabilities.ItemHandler.BLOCK, // capability to cache
        level, // level
        pos, // target position
        Direction.NORTH // context
);
```

然后使用 `getCapability()` 查询该高速缓存：

```java
IItemHandler handler = this.capCache.getCapability();
if (handler != null) {
    // Use the handler for some item-related operation.
}
```

**该高速缓存由垃圾收集器自动清除，无需注销。**

也可以在功能对象更改时接收通知！这包括功能更改（ `oldHandler != newHandler` ）、变得不可用（ `null` ）或再次可用（不再是 `null` ）。

然后需要使用两个附加参数创建该高速缓存：

- 一种有效性检查，用于确定该高速缓存是否仍然有效。在作为块实体字段的最简单用法中， `() -> !this.isRemoved()` 就可以了。
- 一个失效侦听器，当功能更改时调用。这是您可以对功能更改、删除或出现做出反应的地方。

```vjava
// With optional invalidation listener:
this.capCache = BlockCapabilityCache.create(
        Capabilities.ItemHandler.BLOCK, // capability to cache
        level, // level
        pos, // target position
        Direction.NORTH, // context
        () -> !this.isRemoved(), // validity check (because the cache might outlive the object it belongs to)
        () -> onCapInvalidate() // invalidation listener
);
```

为了让这个系统工作，每当一个功能改变、出现或消失时，modder必须调用 `level.invalidateCapabilities(pos)` 。

```java
// whenever a capability changes, appears, or disappears:
level.invalidateCapabilities(pos);
```

​	NeoForge已经处理了一些常见的情况，比如块加载/卸载和块实体创建/删除，但其他情况需要由modder显式处理。例如，在以下情况下，modders必须使能力无效：

- 如果功能提供 block entity 的配置发生变化。
- 如果放置了功能提供block（没有 block entity ）或更改了state，则通过覆盖 `onPlace` 。
- 如果删除了功能提供block（没有block entity），则通过覆盖 `onRemove` 。

有关普通块的示例，请参阅 `ComposterBlock.java` 文件。

有关更多信息，请参阅 `IBlockCapabilityProvider` 的javadoc。

### Registering capabilities 

能力提供者是最终提供能力的人。能力提供者是一个函数，它可以返回一个能力实例，或者如果它不能提供能力，则返回 `null` 。供应商具体针对：

- 他们提供的特定能力，以及
- 它们提供的block instance, block entity type, entity type,或item 实例。

需要在 `RegisterCapabilitiesEvent` 中注册。

区块提供程序已注册到 `registerBlock` 。举例来说：

```java
private static void registerCapabilities(RegisterCapabilitiesEvent event) {
    event.registerBlock(
        Capabilities.ItemHandler.BLOCK, // capability to register for
        (level, pos, state, be, side) -> <return the IItemHandler>,
        // blocks to register for
        MY_ITEM_HANDLER_BLOCK,
        MY_OTHER_ITEM_HANDLER_BLOCK);
}
```

一般来说，注册将特定于某些 block entity types，因此也提供了 `registerBlockEntity` helper方法：

```java
   event.registerBlockEntity(
        Capabilities.ItemHandler.BLOCK, // capability to register for
        MY_BLOCK_ENTITY_TYPE, // block entity type to register for
        (myBlockEntity, side) -> <return the IItemHandler for myBlockEntity and side>)
```

实体注册类似，使用 `registerEntity` ：

```java
event.registerEntity(
    Capabilities.ItemHandler.ENTITY, // capability to register for
    MY_ENTITY_TYPE, // entity type to register for
    (myEntity, context) -> <return the IItemHandler for myEntity>);
```

item注册也是类似的。请注意，提供程序接收stack：

```java
event.registerItem(
    Capabilities.ItemHandler.ITEM, // capability to register for
    (itemStack, context) -> <return the IItemHandler for the itemStack>,
    // items to register for
    MY_ITEM,
    MY_OTHER_ITEM);
```

如果出于某种原因，您需要为所有块、实体或项注册一个提供程序，则需要重新配置相应的注册表并为每个对象注册提供程序。

例如，NeoForge使用此系统为所有桶注册流体处理器功能：

```java
// For reference, you can find this code in the `CapabilityHooks` class.
for (Item item : BuiltInRegistries.ITEM) {
    if (item.getClass() == BucketItem.class) {
        event.registerItem(Capabilities.FluidHandler.ITEM, (stack, ctx) -> new FluidBucketWrapper(stack), item);
    }
}
```

提供者被要求按照他们注册的顺序提供功能。如果你想在NeoForge已经为你的某个对象注册的提供程序之前运行，请以更高的优先级注册你的 `RegisterCapabilitiesEvent` 处理程序。举例来说：

```java
modBus.addListener(RegisterCapabilitiesEvent.class, event -> {
    event.registerItem(
        Capabilities.FluidHandler.ITEM,
        (stack, ctx) -> new MyCustomFluidBucketWrapper(stack),
        // blocks to register for
        MY_CUSTOM_BUCKET);
}, EventPriority.HIGH); // use HIGH priority to register before NeoForge!
```

请参阅 `CapabilityHooks` 以获取NeoForge自己注册的providers 列表。

### Entities, IItemHandler and Direction 

*如果不使用item处理程序实体功能，则可以跳过此部分。*

实体上的item处理程序现在有两种功能：

- `Capabilities.ItemHandler.ENTITY` ：暴露某个实体的完整**inventory** 。
- `Capabilities.ItemHandler.ENTITY_AUTOMATION` ：暴露自动化可访问的**inventory**。料斗和滴管添加以支持这一能力。

下面是旧系统的迁移指南，旧系统使用单一功能，并使用 `Direction` 参数进行区分：

#### Minecart and chest inventories 

如果您希望支持自动化感知inventories：

| Old Syntax 旧语法           | New Syntax 新语法                                            |
| --------------------------- | ------------------------------------------------------------ |
| `entity.getCapability(...)` | `entity.getCapability(Capabilities.ItemHandler.ENTITY_AUTOMATION)` |

否则：

| Old Syntax 旧语法           | New Syntax 新语法                                       |
| --------------------------- | ------------------------------------------------------- |
| `entity.getCapability(...)` | `entity.getCapability(Capabilities.ItemHandler.ENTITY)` |

#### 马匹 inventory

| Old Syntax 旧语法               | New Syntax 新语法                                      |
| ------------------------------- | ------------------------------------------------------ |
| `horse.getCapability(..., ...)` | `horse.getCapability(Capabilities.ItemHandler.ENTITY)` |

#### Living entities

| Old Syntax 旧语法                                     | New Syntax 新语法                                            |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| `entity.getCapability(..., any vertical direction)`   | `new EntityHandsInvWrapper(livingEntity)`                    |
| `entity.getCapability(..., any horizontal direction)` | `new EntityArmorInvWrapper(livingEntity)`                    |
| `entity.getCapability(..., null)`                     | `livingEntity.getCapability(Capabilities.ItemHandler.ENTITY)` |

#### Players 

| Old Syntax 旧语法                                     | New Syntax 新语法                                            |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| `player.getCapability(..., any vertical direction)`   | `new PlayerMainInvWrapper(player.getInventory())`            |
| `player.getCapability(..., any horizontal direction)` | `new CombinedInvWrapper(new PlayerArmorInvWrapper(player.getInventory()), new PlayerOffhandInvWrapper(player.getInventory()))` |
| `player.getCapability(..., null)`                     | `player.getCapability(Capabilities.ItemHandler.ENTITY)`      |

### 未来能力计划

Composters 现在支持item处理程序功能。然而，坩埚仍然不支持流体处理器能力。这将在未来几周内得到解决，使用块流体处理器能力的mods将与开箱即用的坩埚一起工作。

我们已经广泛地审查和测试了这一能力改革。尽管如此，我们希望在发布后发现问题。请不要犹豫与我们联系，无论是在Discord还是GitHub上！

就到这里吧，移植愉快！
