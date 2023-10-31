---
title: Architectury-04-高级方块
date: 2023-10-28 22:10:40
tags:
- 我的世界
- 模组
- Java
- 教程
cover: https://w.wallhaven.cc/full/rr/wallhaven-rreeg1.jpg
---

# 添加一个高级方块

![image-20231029150618990](https://s2.loli.net/2023/10/29/ct5BRLgM9qGo1fe.png)

## 添加方块代码

```java

package net.tutorialmod.block.custom;

import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.effect.MobEffectInstance;
import net.minecraft.world.effect.MobEffects;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.BlockHitResult;

public class JumpyBlock extends Block {
    public JumpyBlock(Properties properties) {
        super(properties);
    }

    @Override
    public InteractionResult use(BlockState blockState, Level level, BlockPos blockPos, Player player, InteractionHand interactionHand, BlockHitResult blockHitResult) {
        player.sendSystemMessage(Component.literal("Right Clicked this!"));

        return InteractionResult.SUCCESS;
    }


    @Override
    public void stepOn(Level level, BlockPos blockPos, BlockState blockState, Entity entity) {
        if(entity instanceof LivingEntity entity1){
            entity1.addEffect(new MobEffectInstance(new MobEffectInstance(MobEffects.JUMP,200)));
        }
        super.stepOn(level, blockPos, blockState, entity);
    }
}

```



## 注册方块

```java

    public static final RegistrySupplier<Block> JUMPY_BLOCK = registerBlock("jumpy_block",()-> new JumpyBlock(BlockBehaviour.Properties.copy(Blocks.STONE)));

```



## 添加blockstate

```jsno

{
  "variants": {
    "": {
      "model": "tutorialmod:block/jumpy_block"
    }
  }
}
```



## 添加方块model

```json
{
  "parent": "minecraft:block/cube_all",
  "textures": {
    "all": "tutorialmod:block/jumpy_block"
  }
}
```



## 添加方块item

```json
{
  "parent": "tutorialmod:block/jumpy_block"
}

```



## 添加方块贴图

略

