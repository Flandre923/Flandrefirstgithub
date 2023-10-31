---
title: Architectury-08-添加画
date: 2023-10-29 17:04:16
tags:
- 我的世界
- 模组
- Java
cover: https://w.wallhaven.cc/full/l8/wallhaven-l8d5m2.jpg
---

# 添加画

![image-20231029172543488](https://s2.loli.net/2023/10/29/eLszt2JiVmAb9dR.png)

![image-20231029172640107](https://s2.loli.net/2023/10/29/XT6xNVYCc7jUlwq.png)

## 添加画

```java
package net.tutorialmod.painting;

import dev.architectury.registry.registries.DeferredRegister;
import dev.architectury.registry.registries.RegistrySupplier;
import net.minecraft.core.registries.Registries;
import net.minecraft.world.entity.decoration.PaintingVariant;
import net.minecraft.world.level.block.Block;
import net.tutorialmod.TutorialMod;

public class ModPainting {
    public static final DeferredRegister<PaintingVariant> PAINTING_VARIANTS = DeferredRegister.create(TutorialMod.MOD_ID, Registries.PAINTING_VARIANT);
    public static final RegistrySupplier<PaintingVariant> PLANT = PAINTING_VARIANTS.register("plant",
            () -> new PaintingVariant(16, 16));
    public static final RegistrySupplier<PaintingVariant> WANDERER = PAINTING_VARIANTS.register("wanderer",
            () -> new PaintingVariant(16, 32));
    public static final RegistrySupplier<PaintingVariant> SUNSET = PAINTING_VARIANTS.register("sunset",
            () -> new PaintingVariant(32, 16));

    public static void register(){
        PAINTING_VARIANTS.register();
    }

}

```



main类

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
import net.tutorialmod.painting.ModPainting;

import java.util.function.Supplier;

public class TutorialMod {
    public static final String MOD_ID = "tutorialmod";
    // We can use this if we don't want to use DeferredRegister
    public static final Supplier<RegistrarManager> REGISTRIES = Suppliers.memoize(() -> RegistrarManager.get(MOD_ID));

    public static void init() {
        ModCreativeTab.register();
        ModBlock.register();
        ModItem.register();
        ModPainting.register();

        System.out.println(TutorialModExpectPlatform.getConfigDirectory().toAbsolutePath().normalize().toString());
    }
}

```



## 添加tags

```json
{
  "values": [
    "tutorialmod:wanderer",
    "tutorialmod:plant",
    "tutorialmod:sunset"
  ]
}
```



## 添加贴图

![image-20231029172554282](https://s2.loli.net/2023/10/29/EvGjHSUJ1V2ZYrn.png)

