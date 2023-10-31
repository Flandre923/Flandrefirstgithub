---
title: Architectury-06-添加多状态方块
date: 2023-10-29 16:13:45
tags:
- 我的世界
- 模组
- 教程
- java
cover: https://w.wallhaven.cc/full/d6/wallhaven-d6d963.jpg
---



# 添加多状态的方块

![image-20231029162325525](https://s2.loli.net/2023/10/29/lp8zAY4c3G5fRqS.png)

![image-20231029162337992](https://s2.loli.net/2023/10/29/MjznaZoV3GFNYQq.png)

## 添加方块

```java

package net.tutorialmod.block.custom;

import net.minecraft.core.BlockPos;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.StateDefinition;
import net.minecraft.world.level.block.state.properties.BooleanProperty;
import net.minecraft.world.phys.BlockHitResult;

public class ZirconLampBlock extends Block {

    public static final BooleanProperty LIT = BooleanProperty.create("lit");
    public ZirconLampBlock(Properties properties) {
        super(properties);
    }

    @Override
    public InteractionResult use(BlockState blockState, Level level, BlockPos blockPos, Player player, InteractionHand interactionHand, BlockHitResult blockHitResult) {
        if(!level.isClientSide() && interactionHand == InteractionHand.MAIN_HAND) {
            level.setBlock(blockPos, blockState.cycle(LIT),3);
        }
        return InteractionResult.SUCCESS;
    }

    @Override
    protected void createBlockStateDefinition(StateDefinition.Builder<Block, BlockState> builder) {
        builder.add(LIT);
    }

}

```



## 注册方块

```java

    //lamp
    public static final RegistrySupplier<Block> ZIRCON_LAMP = registerBlock("zircon_lamp",
            () -> new ZirconLampBlock(BlockBehaviour.Properties.copy(Blocks.STONE)
                    .strength(6f).requiresCorrectToolForDrops()
                    .lightLevel(state -> state.getValue(ZirconLampBlock.LIT) ? 15 : 0)));

```



## 添加blockstate

```json

{
  "variants": {
    "lit=false": {
      "model": "tutorialmod:block/zircon_lamp_off"
    },
    "lit=true": {
      "model": "tutorialmod:block/zircon_lamp_on"
    }
  }
}
```



## 添加方块model

```json
{
  "parent": "minecraft:block/cube_all",
  "textures": {
    "all": "tutorialmod:block/zircon_lamp_off"
  }
}
```



```json

{
  "parent": "minecraft:block/cube_all",
  "textures": {
    "all": "tutorialmod:block/zircon_lamp_on"
  }
}
```



## 添加物品model

```json
{
  "parent": "tutorialmod:block/zircon_lamp_off"
}
```



## 添加贴图

略

## 添加语言文件

略

