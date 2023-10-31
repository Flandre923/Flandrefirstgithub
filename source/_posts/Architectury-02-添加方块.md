---
title: Architectury-02-添加方块
date: 2023-10-28 20:51:04
tags:
-  我的世界
-  模组教程
cover: https://w.wallhaven.cc/full/3l/wallhaven-3l7769.png
---



# 添加方块

![image-20231028212405539](https://s2.loli.net/2023/10/28/wqLaZK1WGzPrnVF.png)

## 添加方块

```java
package net.tutorialmod.block;

import dev.architectury.registry.registries.DeferredRegister;
import dev.architectury.registry.registries.RegistrySupplier;
import net.minecraft.core.registries.Registries;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.Item;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockBehaviour;
import net.tutorialmod.TutorialMod;
import net.tutorialmod.item.ModCreativeTab;
import net.tutorialmod.item.ModItem;

import java.util.function.Supplier;

public class ModBlock {

    public static final DeferredRegister<Block> BLOCKS = DeferredRegister.create(TutorialMod.MOD_ID, Registries.BLOCK);

    public static final RegistrySupplier<Block> ZIRCON_BLOCK = registerBlock("zircon_block",()->new Block(BlockBehaviour.Properties.copy(Blocks.STONE)));
    private static <T extends Block> RegistrySupplier<T> registerBlock(String name, Supplier<T> block) {
        RegistrySupplier<T> toReturn = BLOCKS.register(name, block);
        registerBlockItem(name, toReturn);
        return toReturn;

    }

    private static <T extends Block> RegistrySupplier<Item> registerBlockItem(String name, RegistrySupplier<T> block) {
        return ModItem.ITEMS.register(name, () -> new BlockItem(block.get(), new Item.Properties().arch$tab(ModCreativeTab.EXAMPLE_TAB)));
    }


    public static void register(){
        BLOCKS.register();
    }



}

```



## 修改TutorialMod类

```java
package net.tutorialmod;

import com.google.common.base.Suppliers;
import dev.architectury.registry.CreativeTabRegistry;
import dev.architectury.registry.registries.DeferredRegister;
import dev.architectury.registry.registries.RegistrarManager;
import dev.architectury.registry.registries.RegistrySupplier;
import net.minecraft.core.registries.Registries;
import net.minecraft.network.chat.Component;
import net.minecraft.world.item.CreativeModeTab;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.tutorialmod.block.ModBlock;
import net.tutorialmod.item.ModCreativeTab;
import net.tutorialmod.item.ModItem;

import java.util.function.Supplier;

public class TutorialMod {
    public static final String MOD_ID = "tutorialmod";
    // We can use this if we don't want to use DeferredRegister
    public static final Supplier<RegistrarManager> REGISTRIES = Suppliers.memoize(() -> RegistrarManager.get(MOD_ID));

    public static void init() { // 这里线注册方块，在注册Item，不可以反过来
        ModCreativeTab.register();
        ModBlock.register();
        ModItem.register();

        System.out.println(TutorialModExpectPlatform.getConfigDirectory().toAbsolutePath().normalize().toString());
    }
}

```

 

## 添加blockstate

```java

{
  "variants": {
    "": {
      "model": "tutorialmod:block/zircon_block"
    }
  }
}
```



## 添加model

```java
{
  "parent": "minecraft:block/cube_all",
  "textures": {
    "all": "tutorialmod:block/zircon_block"
  }
}
```



## 添加item model

```java

{
  "parent": "tutorialmod:block/zircon_block"
}
```



## 添加texture

## 添加lang

```java

{
  "item.tutorialmod.zircon": "Zircon",
  "itemGroup.tutorialmod.example_tab": "TutoriablTab",
  "block.tutorialmod.zircon_block": "Zircon Block"
}
```

