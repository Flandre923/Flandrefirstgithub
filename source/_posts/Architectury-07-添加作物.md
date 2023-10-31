---
title: Architectury-07-添加作物
date: 2023-10-29 16:33:19
tags:
- 我的世界
- 模组
- java
- 教程
cover: https://w.wallhaven.cc/full/d6/wallhaven-d6v5qm.jpg
---

# 添加作物

![image-20231029165918514](https://s2.loli.net/2023/10/29/nREwbAS2OBrNJhc.png)



![image-20231029165957820](https://s2.loli.net/2023/10/29/4lGTQFCU3JWouwI.png)

![](https://s2.loli.net/2023/10/29/SjVHUfM84Fy9OWh.png)

![image-20231029170008472](https://s2.loli.net/2023/10/29/NTMytBC3ulbKspw.png)

![image-20231029170019289](https://s2.loli.net/2023/10/29/Dqp7zUBsE8rul1O.png)

## 添加作物方块

```java

package net.tutorialmod.block.custom;

import net.minecraft.world.level.ItemLike;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.CropBlock;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.StateDefinition;
import net.minecraft.world.level.block.state.properties.IntegerProperty;
import net.tutorialmod.item.ModItem;

public class BlueberryCropBlock extends CropBlock {
    public static final IntegerProperty AGE = IntegerProperty.create("age",0,6);
    public BlueberryCropBlock(Properties properties) {
        super(properties);
    }

    @Override
    protected ItemLike getBaseSeedId() {
        return ModItem.BLUEBERRY_SEEDS.get();
    }

    @Override
    protected IntegerProperty getAgeProperty() {
        return AGE;
    }

    @Override
    public int getMaxAge() {
        return 6;
    }

    @Override
    protected void createBlockStateDefinition(StateDefinition.Builder<Block, BlockState> builder) {
        builder.add(AGE);
    }


}

```



## 添加作物种子和果实

```java
   //
    public static final RegistrySupplier<Item> BLUEBERRY_SEEDS = ITEMS.register("blueberry_seeds",
            () -> new ItemNameBlockItem(ModBlock.BLUEBERRY_CROP.get(),
                    new Item.Properties().arch$tab(ModCreativeTab.EXAMPLE_TAB)));

    public static final RegistrySupplier<Item> BLUEBERRY = ITEMS.register("blueberry",
            () -> new Item(new Item.Properties().arch$tab(ModCreativeTab.EXAMPLE_TAB)
                    .food(new FoodProperties.Builder().nutrition(2).saturationMod(2f).build())));
```



## 注册作物方块



```java

    public static final RegistrySupplier<Block> BLUEBERRY_CROP = BLOCKS.register("blueberry_crop",
            () -> new BlueberryCropBlock(BlockBehaviour.Properties.copy(Blocks.WHEAT)));

```



## 添加blockstate

```json
{
  "variants": {
    "age=0": {
      "model": "tutorialmod:block/blueberry_stage0"
    },
    "age=1": {
      "model": "tutorialmod:block/blueberry_stage1"
    },
    "age=2": {
      "model": "tutorialmod:block/blueberry_stage2"
    },
    "age=3": {
      "model": "tutorialmod:block/blueberry_stage3"
    },
    "age=4": {
      "model": "tutorialmod:block/blueberry_stage4"
    },
    "age=5": {
      "model": "tutorialmod:block/blueberry_stage5"
    },
    "age=6": {
      "model": "tutorialmod:block/blueberry_stage6"
    }
  }
}
```



## 添加方块model

这里示例一个，其他一样只是改了序号

```json
{
  "parent": "minecraft:block/crop",
  "render_type": "minecraft:cutout",
  "textures": {
    "crop": "tutorialmod:block/blueberry_stage0"
  }
}
```



## 添加物品model

```json
{
  "parent": "minecraft:item/generated",
  "textures": {
    "layer0": "tutorialmod:item/blueberry"
  }
}
```

```json
{
  "parent": "minecraft:item/generated",
  "textures": {
    "layer0": "tutorialmod:item/blueberry_seeds"
  }
}
```



## 添加贴图

略，可在github中下载。

## 添加loottable掉落物品表

每个字段具体含义请查阅wiki，或者翻往期视频，或者使用data generater

```json
{
  "type": "minecraft:block",
  "functions": [
    {
      "function": "minecraft:explosion_decay"
    }
  ],
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
                  "block": "tutorialmod:blueberry_crop",
                  "condition": "minecraft:block_state_property",
                  "properties": {
                    "age": "6"
                  }
                }
              ],
              "name": "tutorialmod:blueberry"
            },
            {
              "type": "minecraft:item",
              "name": "tutorialmod:blueberry_seeds"
            }
          ]
        }
      ],
      "rolls": 1.0
    },
    {
      "bonus_rolls": 0.0,
      "conditions": [
        {
          "block": "tutorialmod:blueberry_crop",
          "condition": "minecraft:block_state_property",
          "properties": {
            "age": "6"
          }
        }
      ],
      "entries": [
        {
          "type": "minecraft:item",
          "functions": [
            {
              "enchantment": "minecraft:fortune",
              "formula": "minecraft:binomial_with_bonus_count",
              "function": "minecraft:apply_bonus",
              "parameters": {
                "extra": 3,
                "probability": 0.5714286
              }
            }
          ],
          "name": "tutorialmod:blueberry_seeds"
        }
      ],
      "rolls": 1.0
    }
  ],
  "random_sequence": "tutorialmod:blocks/blueberry_crop"
}
```



## 补充，fabric的渲染如何调整为cutout

![image-20231029173431149](https://s2.loli.net/2023/10/29/yfZWNVOoHzB3rcM.png)

![image-20231029173439330](https://s2.loli.net/2023/10/29/lLi21k9zHKUgoVv.png)

```json
  "entrypoints": {
    "main": [
      "net.tutorialmod.fabric.TutorialModFabric"
    ],
    "client": [
      "net.tutorialmod.fabric.TutorialModFabricClient"
    ]
```

```java
package net.tutorialmod.fabric;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.blockrenderlayer.v1.BlockRenderLayerMap;
import net.minecraft.client.renderer.RenderType;
import net.tutorialmod.block.ModBlock;

public class TutorialModFabricClient implements ClientModInitializer {
    @Override
    public void onInitializeClient() {
        BlockRenderLayerMap.INSTANCE.putBlock(ModBlock.BLUEBERRY_CROP.get(), RenderType.cutout());
    }
}

```

