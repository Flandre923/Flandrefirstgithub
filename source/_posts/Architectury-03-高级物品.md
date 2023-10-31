---
title: Architectury-03-高级物品
date: 2023-10-28 21:55:06
tags:
- 我的世界
- 模组
cover: https://w.wallhaven.cc/full/5g/wallhaven-5gppp7.jpg
---



# 添加高级物品

![image-20231028215814333](https://s2.loli.net/2023/10/28/R28V6rfXhaKlyGs.png)

## 添加自定义物品

```java

package net.tutorialmod.item.custom;

import net.minecraft.network.chat.Component;
import net.minecraft.util.RandomSource;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.InteractionResultHolder;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.Level;

public class EightBallItem extends Item {
    public EightBallItem(Properties properties) {
        super(properties);
    }

    @Override
    public InteractionResultHolder<ItemStack> use(Level level, Player player, InteractionHand interactionHand) {

        if(!level.isClientSide() && interactionHand == InteractionHand.MAIN_HAND) {
            outputRandomNumber(player);
            player.getCooldowns().addCooldown(this, 20);
        }
        return InteractionResultHolder.success(new ItemStack(this,1));
    }

    private void outputRandomNumber(Player player) {
        player.sendSystemMessage(Component.literal("Your Number is " + getRandomNumber()));
    }

    private int getRandomNumber() {
        return RandomSource.createNewThreadLocalInstance().nextInt(10);
    }

}

```



## 注册

```java

package net.tutorialmod.item;

import dev.architectury.registry.registries.DeferredRegister;
import dev.architectury.registry.registries.RegistrySupplier;
import net.minecraft.core.registries.Registries;
import net.minecraft.world.item.EggItem;
import net.minecraft.world.item.Item;
import net.tutorialmod.TutorialMod;
import net.tutorialmod.item.custom.EightBallItem;

public class ModItem {


    public static final DeferredRegister<Item> ITEMS = DeferredRegister.create(TutorialMod.MOD_ID, Registries.ITEM);
    // 皓石
    public static final RegistrySupplier<Item> EXAMPLE_ITEM = ITEMS.register("zircon", () ->
            new Item(new Item.Properties().arch$tab(ModCreativeTab.EXAMPLE_TAB)));
    // 高级物品
    public static final RegistrySupplier<Item> EIGHT_BALL_ITEM = ITEMS.register("eight_ball",
            () -> new EightBallItem(new Item.Properties().arch$tab(ModCreativeTab.EXAMPLE_TAB).stacksTo(1)));

    public static void register(){
        ITEMS.register();
    }

}

```



## 添加model

```java
{
  "parent": "minecraft:item/generated",
  "textures": {
    "layer0": "tutorialmod:item/eight_ball"
  }
}
```



## 添加 lang

```java
略
```



## 添加textures

略
