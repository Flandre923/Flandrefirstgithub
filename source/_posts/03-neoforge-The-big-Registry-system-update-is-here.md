---
title: 03-neoforge-The big Registry system update is here
date: 2023-11-25 10:11:13
tags:
- 我的世界
- noeforge
cover: https://view.moezx.cc/images/2022/02/24/d44e7008f180acb26efa4b9ec7aebe90.png
---

# 注册表系统大更新来了

## Introduction 介绍

NeoForge 20.2.59-beta 版刚刚发布了注册表系统的重大更新！我们的主要目标是尽可能简化代码，并使其与原版 Minecraft 中的注册表系统保持一致。

这篇博文将介绍所做的最重要的更改，作为模组制作者的迁移指南。

这次重做是我们最初的 20.2 版本发布后进行的三项重大重做中的第一个。未来几周将进行彻底检修的另外两个系统是**capabilities**和 **networking**.。一旦这些也发布，我们的目标是发布 20.2 稳定版本。

## Using the registries 使用注册表

此前，NeoForge 注册表系统与原版完全分开。现在，我们使用原生的现有注册表系统，并添加了一些与 mod 支持相关的内容。

这意味着 `IForgeRegistry` 被替换为普通的 `Registry` 。 NeoForge 通过 `IRegistryExtension` 向普通 `Registry` 类型添加了一些方法。以下是更改后的方法的概述：

| `IForgeRegistry`               | `Registry`                                                   |
| ------------------------------ | ------------------------------------------------------------ |
| `getValue(ResourceLocation)`   | `get(ResourceLocation)`                                      |
| `getKeys()`                    | `keySet()`                                                   |
| `getValues().stream()`         | `stream()`                                                   |
| `getHolder(T)Optional<Holder>` | `wrapAsHolder(T)Holder?`                                     |
| `tags()`                       | Use `getTag(TagKey)` and the HolderSet API 使用 `getTag(TagKey)` 和 HolderSet API |

现在应该通过 `BuiltInRegistries` 访问 Minecraft 本身定义的注册表：

```diff
- ForgeRegistries.ITEMS.getValue(new ResourceLocation("minecraft:diamond"));
+ BuiltInRegistries.ITEM.get(new ResourceLocation("minecraft:diamond"));
```

NeoForge 定义的注册表可以通过 `NeoForgeRegistries` 访问。它们不再包裹在 `Supplier` 中，可以直接使用：

```diff
- ForgeRegistries.FLUID_TYPES.get().getValue(new ResourceLocation("mymod:fancyfluid"));
+ NeoForgeRegistries.FLUID_TYPES.get(new ResourceLocation("mymod:fancyfluid"));
```

## Registration 登记

注册仍然发生在 `RegisterEvent` 中，我们仍然建议模组作者使用 `DeferredRegister` 来抽象事件。

`RegistryObject` 被 `DeferredHolder` 取代，它实现了 vanilla 的 `Holder` 接口。将对象注册到延迟寄存器时，我们推荐两个选项：

- 如果您不需要任何 `Holder` 函数，则可以使用 `Supplier` 作为字段类型。
- 否则，请使用 `DeferredHolder` 和两个通用参数（一个用于注册表，一个用于您的对象类型）。

这是一个例子：

```diff
  private static final DeferredRegister<Enchantment> ENCHANTMENTS = DeferredRegister.create(Registries.ENCHANTMENT, "mymod");

- public static final RegistryObject<Enchantment> MAGIC =
-         ENCHANTMENTS.register("magic", () -> new MagicEnchantment(/* create enchantment */));
 // Supplier only:
+ public static final Supplier<MagicEnchantment> MAGIC =
+         ENCHANTMENTS.register("magic", () -> new MagicEnchantment(/* create enchantment */));
  // Access to both Holder and the exact object type:
+ public static final DeferredHolder<Enchantment, MagicEnchantment> MAGIC =
+         ENCHANTMENTS.register("magic", () -> new MagicEnchantment(/* create enchantment */));
```

NeoForge 还为实现 `ItemLike` 的项目和块提供 `DeferredHolder` 和 `DeferredRegister` 专门化。例如：

```java
// Make sure you use the special DeferredRegister.Blocks and DeferredRegister.Items types,
// NOT DeferredRegister<Block> or DeferredRegister<Item>!
private static final DeferredRegister.Blocks BLOCKS = DeferredRegister.createBlocks("mymod");
private static final DeferredRegister.Items ITEMS = DeferredRegister.createItems("mymod");

// If you are registering blocks or items directly, use a normal `register` call:
public static final DeferredBlock<MyBlock> MY_BLOCK = BLOCKS.register("my_block", () -> new MyBlock(/* create block */));
public static final DeferredItem<MyItem> MY_ITEM = ITEMS.register("my_item", () -> new MyItem(/* create item */));

// There are also a few extra helper functions.
// `registerBlock` to directly register a `new Block` from some block properties:
public static final DeferredBlock<Block> MY_SIMPLE_BLOCK =
        BLOCKS.registerBlock("simple_block", BlockBehaviour.Properties.of().mapColor(MapColor.STONE));
// `registerItem` to directly register a `new Item` from some item properties:
public static final DeferredItem<Item> MY_SIMPLE_ITEM =
        ITEMS.registerItem("simple_item", new Item.Properties().stacksTo(1));
// `registerBlockItem` to directly register a `new BlockItem` for a block:
public static final DeferredItem<BlockItem> MY_BLOCK_ITEM =
        ITEMS.registerBlockItem(MY_BLOCK);
```

像往常一样，不要忘记将 mod 总线传递给 `DeferredRegister` ：

```java
@Mod("mymod")
public class MyMod {
    // In case you missed it, mod constructors can now receive a number of optional arguments,
    // including the mod's event bus. Unrelated to registries, but still pretty cool. ;)
    public MyMod(IEventBus modEventBus) {
        ENCHANTMENTS.register(modEventBus);
        BLOCKS.register(modEventBus);
        ITEMS.register(modEventBus);
    }
}
```

## Custom registries 自定义注册表

自定义注册表是使用 `RegistryBuilder` 创建的，并且必须注册到 `NewRegistryEvent` 。它们现在可以保存在静态字段中，就像 `BuiltInRegistries` 或 `NeoForgeRegistries` 中的注册表一样。

这是一个注册示例，使用 `DeferredRegister` 提供的辅助方法：

```java
// Create a registry key - we don't have a registry yet so give the key to DeferredRegister.
public static final ResourceKey<Registry<Custom>> CUSTOM_REGISTRY_KEY =
        ResourceKey.createRegistryKey(new ResourceLocation("mymod:custom"));
// Create the DeferredRegister with our registry key.
private static final DeferredRegister<Custom> CUSTOMS =
        DeferredRegister.create(CUSTOM_REGISTRY_KEY, "mymod");

// We can register objects as usual...
public static final Holder<Custom> CUSTOM_OBJECT =
        CUSTOMS.register("custom_object", () -> new Custom());

// And here is how to create the registry!
public static final Registry<Custom> CUSTOM_REGISTRY =
        CUSTOMS.makeRegistry(builder -> /* use builder to configure registry if needed */);

// Remember to register CUSTOMS in the mod constructor!
```

另一种方法是直接使用 `RegistryBuilder` 创建注册表，并手动注册它：

```java
// We still need a registry key.
public static final ResourceKey<Registry<Custom>> CUSTOM_REGISTRY_KEY =
        ResourceKey.createRegistryKey(new ResourceLocation("mymod:custom"));
// Create the registry directly...
public static final Registry<Custom> CUSTOM_REGISTRY = new RegistryBuilder<>(CUSTOM_REGISTRY_KEY)
    // configure the builder if you want, for example with .sync(true)
    // then build the registry
    .build();

// Remember to tell NeoForge about your registry! For example:
modEventBus.addListener(NewRegistryEvent.class, event -> event.register(CUSTOM_REGISTRY));
```

## That’s it! 就是这样！

像往常一样，如果您有任何问题，请在 `#modder-support-1.20` 频道中的 Discord 服务器上提问。
