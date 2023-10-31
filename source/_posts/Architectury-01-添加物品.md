---
title: Architectury-01-添加物品
date: 2023-10-28 19:50:25
tags:
-  我的世界
-  模组教程
cover: https://w.wallhaven.cc/full/x6/wallhaven-x6kkwo.jpg
---

# 添加方块

![image-20231028204614315](https://s2.loli.net/2023/10/28/9Y1U3PzImcJWT8x.png)

## 添加方块
```java
package net.tutorialmod.item;

import dev.architectury.registry.registries.DeferredRegister;
import dev.architectury.registry.registries.RegistrySupplier;
import net.minecraft.core.registries.Registries;
import net.minecraft.world.item.Item;
import net.tutorialmod.TutorialMod;

public class ModItem {


    public static final DeferredRegister<Item> ITEMS = DeferredRegister.create(TutorialMod.MOD_ID, Registries.ITEM);
    public static final RegistrySupplier<Item> EXAMPLE_ITEM = ITEMS.register("zircon", () ->
            new Item(new Item.Properties().arch$tab(ModCreativeTab.EXAMPLE_TAB)));


    public static void register(){
        ITEMS.register();
    }

}

```

## 添加创造物品栏

```java
package net.tutorialmod.item;

import dev.architectury.registry.CreativeTabRegistry;
import dev.architectury.registry.registries.DeferredRegister;
import dev.architectury.registry.registries.RegistrySupplier;
import net.minecraft.core.registries.Registries;
import net.minecraft.network.chat.Component;
import net.minecraft.world.item.CreativeModeTab;
import net.minecraft.world.item.ItemStack;
import net.tutorialmod.TutorialMod;

public class ModCreativeTab {
    public static final DeferredRegister<CreativeModeTab> TABS = DeferredRegister.create(TutorialMod.MOD_ID, Registries.CREATIVE_MODE_TAB);


    public static final RegistrySupplier<CreativeModeTab> EXAMPLE_TAB = TABS.register("example_tab", () ->
            CreativeTabRegistry.create(Component.translatable("itemGroup." + TutorialMod.MOD_ID + ".example_tab"),
                    () -> new ItemStack(ModItem.EXAMPLE_ITEM.get())));

    public static void register(){
        TABS.register();
    }


}

```



## 创建语言文件

```json

{
  "item.tutorialmod.zircon": "Zircon",
  "itemGroup.tutorialmod.example_tab": "TutoriablTab"
}
```



## 创建models-item



```java

{
  "parent": "minecraft:item/generated",
  "textures": {
    "layer0": "tutorialmod:item/zircon"
  }
}
```



## 创建贴图



![image-20231028204651340](https://s2.loli.net/2023/10/28/IH5BTa4WK9s6tCF.png)

## 进入游戏测试

![image-20231028204958179](https://s2.loli.net/2023/10/28/u1m32KvjzM9NDgb.png)
