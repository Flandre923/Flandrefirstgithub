---
title: Architectury-05-添加合成表和LootTable以及方块挖掘正确工具的挖掘等级
date: 2023-10-29 10:12:21
tags:
- 我的世界
- 模组
cover: https://w.wallhaven.cc/full/kx/wallhaven-kxjdp1.jpg
---

# 添加合成表Recipes，掉落物品表LootTable，方块挖掘等级，方块正确挖掘工具

先添加了几个方块在ModBlock和ModItem下面

![image-20231029110107898](https://s2.loli.net/2023/10/29/AOgi2lBfhboz3yW.png)

添加对应方块的物品和贴图

![image-20231029110147533](https://s2.loli.net/2023/10/29/bLjucy4IoSOaFKw.png)

添加对应的方块正确挖掘工具，挖掘等级，掉落物表，合成表

![image-20231029110221433](https://s2.loli.net/2023/10/29/ZNgmxABOy1uSola.png)

## 添加方块

```java

package net.tutorialmod.block;

import dev.architectury.registry.registries.DeferredRegister;
import dev.architectury.registry.registries.RegistrySupplier;
import net.minecraft.core.registries.Registries;
import net.minecraft.util.valueproviders.UniformInt;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.Item;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.DropExperienceBlock;
import net.minecraft.world.level.block.state.BlockBehaviour;
import net.tutorialmod.TutorialMod;
import net.tutorialmod.block.custom.JumpyBlock;
import net.tutorialmod.item.ModCreativeTab;
import net.tutorialmod.item.ModItem;

import java.util.function.Supplier;

public class ModBlock {

    public static final DeferredRegister<Block> BLOCKS = DeferredRegister.create(TutorialMod.MOD_ID, Registries.BLOCK);

//    public static final RegistrySupplier<Block> ZIRCON_BLOCK = BLOCKS.register("zircon_block",()->new Block(BlockBehaviour.Properties.copy(Blocks.STONE)));
    public static final RegistrySupplier<Block> ZIRCON_BLOCK = registerBlock("zircon_block",()->new Block(BlockBehaviour.Properties.copy(Blocks.STONE)));
    public static final RegistrySupplier<Block> JUMPY_BLOCK = registerBlock("jumpy_block",()-> new JumpyBlock(BlockBehaviour.Properties.copy(Blocks.STONE)));
    //ore
    public static final RegistrySupplier<Block> ZIRCON_ORE = registerBlock("zircon_ore",
            () -> new DropExperienceBlock(BlockBehaviour.Properties.copy(Blocks.STONE)
                    .strength(6f).requiresCorrectToolForDrops(),
                    UniformInt.of(3, 7)));
    public static final RegistrySupplier<Block> DEEPSLATE_ZIRCON_ORE = registerBlock("deepslate_zircon_ore",
            () -> new DropExperienceBlock(BlockBehaviour.Properties.copy(Blocks.STONE)
                    .strength(6f).requiresCorrectToolForDrops(),
                    UniformInt.of(3, 7)));
    public static final RegistrySupplier<Block> ENDSTONE_ZIRCON_ORE = registerBlock("endstone_zircon_ore",
            () -> new DropExperienceBlock(BlockBehaviour.Properties.copy(Blocks.STONE)
                    .strength(6f).requiresCorrectToolForDrops(),
                    UniformInt.of(3, 7)));
    public static final RegistrySupplier<Block> NETHERRACK_ZIRCON_ORE = registerBlock("netherrack_zircon_ore",
            () -> new DropExperienceBlock(BlockBehaviour.Properties.copy(Blocks.STONE)
                    .strength(6f).requiresCorrectToolForDrops(),
                    UniformInt.of(3, 7)));

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
package net.tutorialmod.block;

import dev.architectury.registry.registries.DeferredRegister;
import dev.architectury.registry.registries.RegistrySupplier;
import net.minecraft.core.registries.Registries;
import net.minecraft.util.valueproviders.UniformInt;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.Item;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.DropExperienceBlock;
import net.minecraft.world.level.block.state.BlockBehaviour;
import net.tutorialmod.TutorialMod;
import net.tutorialmod.block.custom.JumpyBlock;
import net.tutorialmod.item.ModCreativeTab;
import net.tutorialmod.item.ModItem;

import java.util.function.Supplier;

public class ModBlock {

    public static final DeferredRegister<Block> BLOCKS = DeferredRegister.create(TutorialMod.MOD_ID, Registries.BLOCK);

//    public static final RegistrySupplier<Block> ZIRCON_BLOCK = BLOCKS.register("zircon_block",()->new Block(BlockBehaviour.Properties.copy(Blocks.STONE)));
    public static final RegistrySupplier<Block> ZIRCON_BLOCK = registerBlock("zircon_block",()->new Block(BlockBehaviour.Properties.copy(Blocks.STONE)));
    public static final RegistrySupplier<Block> JUMPY_BLOCK = registerBlock("jumpy_block",()-> new JumpyBlock(BlockBehaviour.Properties.copy(Blocks.STONE)));
    //ore
    public static final RegistrySupplier<Block> ZIRCON_ORE = registerBlock("zircon_ore",
            () -> new DropExperienceBlock(BlockBehaviour.Properties.copy(Blocks.STONE)
                    .strength(6f).requiresCorrectToolForDrops(),
                    UniformInt.of(3, 7)));
    public static final RegistrySupplier<Block> DEEPSLATE_ZIRCON_ORE = registerBlock("deepslate_zircon_ore",
            () -> new DropExperienceBlock(BlockBehaviour.Properties.copy(Blocks.STONE)
                    .strength(6f).requiresCorrectToolForDrops(),
                    UniformInt.of(3, 7)));
    public static final RegistrySupplier<Block> ENDSTONE_ZIRCON_ORE = registerBlock("endstone_zircon_ore",
            () -> new DropExperienceBlock(BlockBehaviour.Properties.copy(Blocks.STONE)
                    .strength(6f).requiresCorrectToolForDrops(),
                    UniformInt.of(3, 7)));
    public static final RegistrySupplier<Block> NETHERRACK_ZIRCON_ORE = registerBlock("netherrack_zircon_ore",
            () -> new DropExperienceBlock(BlockBehaviour.Properties.copy(Blocks.STONE)
                    .strength(6f).requiresCorrectToolForDrops(),
                    UniformInt.of(3, 7)));

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

## 添加物品

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
    //
    public static final RegistrySupplier<Item> RAW_ZIRCON = ITEMS.register("raw_zircon",
            () -> new Item(new Item.Properties().arch$tab(ModCreativeTab.EXAMPLE_TAB)));

    public static void register(){
        ITEMS.register();
    }

}

```

## 添加方块state和方块物品以及贴图等略过

## 添加正确的挖掘工具

pickaxe.json  (在mineable下面，其他的工具同理例如斧头axe.json)

相关的工具可以在我的世界包下找到，详细可以去看视频。

```json

{
  "values": [
    "tutorialmod:zircon_block",
    "tutorialmod:zircon_ore",
    "tutorialmod:deepslate_zircon_ore"
  ]
}
```

## 掉落物表

这里json的意思就不详细介绍了，可以去看往期视频，或者去找对应wiki有介绍

wiki搜索minecraftwiki就可以了。

deepslate_zircon_ore.json

```json
{
  "type": "minecraft:block",
  "pools": [
    {
      "bonus_rolls": 0.0,
      "entries": [
        {
          "type": "minecraft:alternatives",
          "children": [
            {
              "type": "minecraft:item",
              "conditions": [
                {
                  "condition": "minecraft:match_tool",
                  "predicate": {
                    "enchantments": [
                      {
                        "enchantment": "minecraft:silk_touch",
                        "levels": {
                          "min": 1
                        }
                      }
                    ]
                  }
                }
              ],
              "name": "tutorialmod:zircon_ore"
            },
            {
              "type": "minecraft:item",
              "functions": [
                {
                  "add": false,
                  "count": {
                    "type": "minecraft:uniform",
                    "max": 9.0,
                    "min": 2.0
                  },
                  "function": "minecraft:set_count"
                },
                {
                  "enchantment": "minecraft:fortune",
                  "formula": "minecraft:uniform_bonus_count",
                  "function": "minecraft:apply_bonus",
                  "parameters": {
                    "bonusMultiplier": 1
                  }
                },
                {
                  "function": "minecraft:explosion_decay"
                }
              ],
              "name": "tutorialmod:raw_zircon"
            }
          ]
        }
      ],
      "rolls": 1.0
    }
  ],
  "random_sequence": "tutorialmod:blocks/zircon_ore"
}
```

zircon_block.json

```json
{
  "type": "minecraft:block",
  "pools": [
    {
      "bonus_rolls": 0.0,
      "conditions": [
        {
          "condition": "minecraft:survives_explosion"
        }
      ],
      "entries": [
        {
          "type": "minecraft:item",
          "name": "tutorialmod:zircon_block"
        }
      ],
      "rolls": 1.0
    }
  ],
  "random_sequence": "tutorialmod:blocks/zircon_block"
}
```

zircon_ore.json

```json
{
  "type": "minecraft:block",
  "pools": [
    {
      "bonus_rolls": 0.0,
      "entries": [
        {
          "type": "minecraft:alternatives",
          "children": [
            {
              "type": "minecraft:item",
              "conditions": [
                {
                  "condition": "minecraft:match_tool",
                  "predicate": {
                    "enchantments": [
                      {
                        "enchantment": "minecraft:silk_touch",
                        "levels": {
                          "min": 1
                        }
                      }
                    ]
                  }
                }
              ],
              "name": "tutorialmod:zircon_ore"
            },
            {
              "type": "minecraft:item",
              "functions": [
                {
                  "add": false,
                  "count": {
                    "type": "minecraft:uniform",
                    "max": 9.0,
                    "min": 2.0
                  },
                  "function": "minecraft:set_count"
                },
                {
                  "enchantment": "minecraft:fortune",
                  "formula": "minecraft:uniform_bonus_count",
                  "function": "minecraft:apply_bonus",
                  "parameters": {
                    "bonusMultiplier": 1
                  }
                },
                {
                  "function": "minecraft:explosion_decay"
                }
              ],
              "name": "tutorialmod:raw_zircon"
            }
          ]
        }
      ],
      "rolls": 1.0
    }
  ],
  "random_sequence": "tutorialmod:blocks/zircon_ore"
}
```

## 添加合成表

zircon.json

使用9个zircon合成一个block

```json

{
  "type": "minecraft:crafting_shapeless",
  "category": "misc",
  "group": "tutorialmod",
  "ingredients": [
    {
      "item": "tutorialmod:zircon_block"
    }
  ],
  "result": {
    "count": 9,
    "item": "tutorialmod:zircon"
  }
}
```

zircon_block.json

使用block合成9个zircon

```json

{
  "type": "minecraft:crafting_shaped",
  "category": "misc",
  "group": "tutorialmod",
  "key": {
    "#": {
      "item": "tutorialmod:zircon"
    }
  },
  "pattern": [
    "###",
    "###",
    "###"
  ],
  "result": {
    "item": "tutorialmod:zircon_block"
  },
  "show_notification": true
}
```

zircon_from_sme****.json

raw 熔炉 烧

```json
{
  "type": "minecraft:smelting",
  "category": "misc",
  "cookingtime": 200,
  "experience": 0.7,
  "group": "tutorialmod",
  "ingredient": {
    "item": "tutorialmod:raw_zircon"
  },
  "result": "tutorialmod:zircon"
}
```

zircon_from_bl**.json

raw 高炉 烧

```json
{
  "type": "minecraft:blasting",
  "category": "misc",
  "cookingtime": 100,
  "experience": 0.7,
  "group": "tutorialmod",
  "ingredient": {
    "item": "tutorialmod:raw_zircon"
  },
  "result": "tutorialmod:zircon"
}
```

具体字段含义查询wiki，不过看英语大概也能看懂把。

