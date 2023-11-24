---
title: LootParams相关解析
date: 2023-11-12 20:44:04
tags:
- 我的世界
- 源码
cover: https://view.moezx.cc/images/2022/02/24/573064e65b87c86a4ddc518fc422c910.png
---

# LootParams类

```java
package net.minecraft.world.level.storage.loot;

import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.function.Consumer;
import javax.annotation.Nullable;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.storage.loot.parameters.LootContextParam;
import net.minecraft.world.level.storage.loot.parameters.LootContextParamSet;

public class LootParams {
   private final ServerLevel level; // 服务器世界对象
   private final Map<LootContextParam<?>, Object> params; // ,固定参数map
   private final Map<ResourceLocation, LootParams.DynamicDrop> dynamicDrops; //动态掉落map
   private final float luck; //,幸运值

   public LootParams(
      ServerLevel p_287766_, Map<LootContextParam<?>, Object> p_287705_, Map<ResourceLocation, LootParams.DynamicDrop> p_287642_, float p_287671_
   ) {
      this.level = p_287766_;
      this.params = p_287705_;
      this.dynamicDrops = p_287642_;
      this.luck = p_287671_;
   }

   public ServerLevel getLevel() {
      return this.level;
   }
//判断固定参数map是否包含指定参数
   public boolean hasParam(LootContextParam<?> p_287749_) {
      return this.params.containsKey(p_287749_);
   }
//从固定参数map获取指定参数,如果不存在抛出异常
   public <T> T getParameter(LootContextParam<T> p_287670_) {
      T t = (T)this.params.get(p_287670_);
      if (t == null) {
         throw new NoSuchElementException(p_287670_.getName().toString());
      } else {
         return t;
      }
   }
//,从固定参数map获取指定参数,如果不存在返回null
   @Nullable
   public <T> T getOptionalParameter(LootContextParam<T> p_287644_) {
      return (T)this.params.get(p_287644_);
   }
//从固定参数map获取指定参数,如果不存在返回null
   @Nullable
   public <T> T getParamOrNull(LootContextParam<T> p_287769_) {
      return (T)this.params.get(p_287769_);
   }
//addDynamicDrops方法,向dynamicDrops动态掉落map中添加指定资源位置的动态掉落

   public void addDynamicDrops(ResourceLocation p_287768_, Consumer<ItemStack> p_287711_) {
      LootParams.DynamicDrop lootparams$dynamicdrop = this.dynamicDrops.get(p_287768_);
      if (lootparams$dynamicdrop != null) {
         lootparams$dynamicdrop.add(p_287711_);
      }
   }
//返回luck幸运值属性
   public float getLuck() {
      return this.luck;
   }
//内部静态类Builder
   public static class Builder {
      private final ServerLevel level;//服务器世界对象
      private final Map<LootContextParam<?>, Object> params = Maps.newIdentityHashMap();//固定参数map,初始化为空map
      private final Map<ResourceLocation, LootParams.DynamicDrop> dynamicDrops = Maps.newHashMap();//动态掉落map,初始化为空map
      private float luck;//幸运值
//类构造函数,参数为服务器世界对象,初始化level属性
      public Builder(ServerLevel p_287594_) {
         this.level = p_287594_;
      }
//方法,返回level属性
      public ServerLevel getLevel() {
         return this.level;
      }
//withParameter方法,向固定参数map中put指定参数
      public <T> LootParams.Builder withParameter(LootContextParam<T> p_287706_, T p_287606_) {
         this.params.put(p_287706_, p_287606_);
         return this;
      }
//可选地向固定参数map中put指定参数,如果值为空则remove
      public <T> LootParams.Builder withOptionalParameter(LootContextParam<T> p_287680_, @Nullable T p_287630_) {
         if (p_287630_ == null) {
            this.params.remove(p_287680_);
         } else {
            this.params.put(p_287680_, p_287630_);
         }

         return this;
      }
//getParameter方法,从固定参数map获取指定参数,如果不存在抛出异常
      public <T> T getParameter(LootContextParam<T> p_287646_) {
         T t = (T)this.params.get(p_287646_);
         if (t == null) {
            throw new NoSuchElementException(p_287646_.getName().toString());
         } else {
            return t;
         }
      }
//getOptionalParameter方法,从固定参数map获取指定参数,如果不存在返回null

      @Nullable
      public <T> T getOptionalParameter(LootContextParam<T> p_287759_) {
         return (T)this.params.get(p_287759_);
      }
//withDynamicDrop方法,向dynamicDrops动态掉落map中put指定资源位置的动态掉落,不允许重复
      public LootParams.Builder withDynamicDrop(ResourceLocation p_287734_, LootParams.DynamicDrop p_287724_) {
         LootParams.DynamicDrop lootparams$dynamicdrop = this.dynamicDrops.put(p_287734_, p_287724_);
         if (lootparams$dynamicdrop != null) {
            throw new IllegalStateException("Duplicated dynamic drop '" + this.dynamicDrops + "'");
         } else {
            return this;
         }
      }
//设置luck幸运值属性
      public LootParams.Builder withLuck(float p_287703_) {
         this.luck = p_287703_;
         return this;
      }
// create方法,通过LootContextParamSet验证参数,然后构造LootParams返回

       //create方法接受一个LootContextParamSet类型的参数，首先计算出params映射中的键与p_287701_的getAllowed方法返回的集合的差集，如果这个差集不为空，则抛出IllegalArgumentException异常，然后计算出p_287701_的getRequired方法返回的集合与params映射中的键的差集，如果这个差集不为空，则抛出IllegalArgumentException异常，否则返回一个新的LootParams对象
      public LootParams create(LootContextParamSet p_287701_) {
         Set<LootContextParam<?>> set = Sets.difference(this.params.keySet(), p_287701_.getAllowed());
         if (false && !set.isEmpty()) { // Forge: Allow mods to pass custom loot parameters (not part of the vanilla loot table) to the loot context.
            throw new IllegalArgumentException("Parameters not allowed in this parameter set: " + set);
         } else {
            Set<LootContextParam<?>> set1 = Sets.difference(p_287701_.getRequired(), this.params.keySet());
            if (!set1.isEmpty()) {
               throw new IllegalArgumentException("Missing required parameters: " + set1);
            } else {
               return new LootParams(this.level, this.params, this.dynamicDrops, this.luck);
            }
         }
      }
   }
//这个方法接受一个Consumer<ItemStack>类型的参数。
   @FunctionalInterface
   public interface DynamicDrop {
      void add(Consumer<ItemStack> p_287584_);
   }
}

```

# LootContextParam 类

主要目的是定义一个可以存储资源位置的参数类，这个类可以被用于表示游戏中的各种上下文参数，例如掉落物的位置，玩家的位置等等

```java
package net.minecraft.world.level.storage.loot.parameters;

import net.minecraft.resources.ResourceLocation;
//主要目的是定义一个可以存储资源位置的参数类，这个类可以被用于表示游戏中的各种上下文参数，例如掉落物的位置，玩家的位置等等
public class LootContextParam<T> {
    // 定义了一个私有的、不可变的ResourceLocation类型的成员变量name。这个变量在这个类的构造函数中被初始化，并且在类的其他方法中可以被访问。
   private final ResourceLocation name;
//LootContextParam类的构造函数，它接受一个ResourceLocation类型的参数p_81283_，并将其赋值给成员变量name。
   public LootContextParam(ResourceLocation p_81283_) {
      this.name = p_81283_;
   }
//一个公共的方法getName，它没有参数，返回值为ResourceLocation类型。这个方法返回成员变量name的值。
   public ResourceLocation getName() {
      return this.name;
   }
//一个覆盖了Object类的toString方法的方法。这个方法返回一个字符串，这个字符串是"<parameter "和this.name的值的组合。这个方法的目的是提供一个表示这个对象的字符串，当我们试图将这个对象转换为字符串时，就会调用这个方法。
   @Override
   public String toString() {
      return "<parameter " + this.name + ">";
   }
}

```



# LootContextParams类

定义了一个名为`LootContextParams`的类，这个类用于创建各种类型的掉落物上下文参数

```java
package net.minecraft.world.level.storage.loot.parameters;

import net.minecraft.resources.ResourceLocation;
import net.minecraft.world.damagesource.DamageSource;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.Vec3;
//
public class LootContextParams {
    //定义了一个公共的、静态的、不可变的LootContextParam<Entity>类型的成员变量THIS_ENTITY，它的值是通过调用create方法并传入字符串"this_entity"得到的。
   public static final LootContextParam<Entity> THIS_ENTITY = create("this_entity");
    //定义了一个公共的、静态的、不可变的LootContextParam<Player>类型的成员变量LAST_DAMAGE_PLAYER，它的值是通过调用create方法并传入字符串"last_damage_player"得到的。
   public static final LootContextParam<Player> LAST_DAMAGE_PLAYER = create("last_damage_player");
   public static final LootContextParam<DamageSource> DAMAGE_SOURCE = create("damage_source");
   public static final LootContextParam<Entity> KILLER_ENTITY = create("killer_entity");
   public static final LootContextParam<Entity> DIRECT_KILLER_ENTITY = create("direct_killer_entity");
   public static final LootContextParam<Vec3> ORIGIN = create("origin");
   public static final LootContextParam<BlockState> BLOCK_STATE = create("block_state");
   public static final LootContextParam<BlockEntity> BLOCK_ENTITY = create("block_entity");
   public static final LootContextParam<ItemStack> TOOL = create("tool");
   public static final LootContextParam<Float> EXPLOSION_RADIUS = create("explosion_radius");

   private static <T> LootContextParam<T> create(String p_81467_) {
      return new LootContextParam<>(new ResourceLocation(p_81467_));
   }
}

```



# block类部分代码

```java

   public static void dropResources(BlockState p_49951_, Level p_49952_, BlockPos p_49953_) {
       //检查p_49952_是否是ServerLevel类的实例
      if (p_49952_ instanceof ServerLevel) {
          //先调用getDrops方法并将p_49951_，(ServerLevel)p_49952_，p_49953_，以及null作为参数传入，然后对返回的列表进行遍历，对于列表中的每一个元素，调用popResource方法并将p_49952_，p_49953_，以及元素作为参数传入。popResource方法的目的是在世界中掉落一个物品。
         getDrops(p_49951_, (ServerLevel)p_49952_, p_49953_, null).forEach(p_152406_ -> popResource(p_49952_, p_49953_, p_152406_));
          //用了p_49951_的spawnAfterBreak方法，并将(ServerLevel)p_49952_，p_49953_，一个空的ItemStack，以及true作为参数传入。这个方法的目的是在方块被挖掘后生成一些实体，例如掉落的物品或者生成的实体。
         p_49951_.spawnAfterBreak((ServerLevel)p_49952_, p_49953_, ItemStack.EMPTY, true);
      }
   }

   public static void dropResources(BlockState p_49893_, LevelAccessor p_49894_, BlockPos p_49895_, @Nullable BlockEntity p_49896_) {
      if (p_49894_ instanceof ServerLevel) {
         getDrops(p_49893_, (ServerLevel)p_49894_, p_49895_, p_49896_).forEach(p_49859_ -> popResource((ServerLevel)p_49894_, p_49895_, p_49859_));
         p_49893_.spawnAfterBreak((ServerLevel)p_49894_, p_49895_, ItemStack.EMPTY, true);
      }
   }

   public static void dropResources(
      BlockState p_49882_, Level p_49883_, BlockPos p_49884_, @Nullable BlockEntity p_49885_, @Nullable Entity p_49886_, ItemStack p_49887_
   ) {
       // 掉落经验
      dropResources(p_49882_, p_49883_, p_49884_, p_49885_, p_49886_, p_49887_, true);
   }
   public static void dropResources(BlockState p_49882_, Level p_49883_, BlockPos p_49884_, @Nullable BlockEntity p_49885_, @Nullable Entity p_49886_, ItemStack p_49887_, boolean dropXp) {
      if (p_49883_ instanceof ServerLevel) {
         getDrops(p_49882_, (ServerLevel)p_49883_, p_49884_, p_49885_, p_49886_, p_49887_).forEach(p_49944_ -> popResource(p_49883_, p_49884_, p_49944_));
         p_49882_.spawnAfterBreak((ServerLevel)p_49883_, p_49884_, p_49887_, dropXp);
      }
   }
// 这个方法是在玩家挖掘方块时被调用的
   public void playerDestroy(Level p_49827_, Player p_49828_, BlockPos p_49829_, BlockState p_49830_, @Nullable BlockEntity p_49831_, ItemStack p_49832_) {
       //调用了Player的awardStat方法，并将Stats.BLOCK_MINED.get(this)作为参数传入。这个方法的目的是奖励玩家挖掘方块的行为
      p_49828_.awardStat(Stats.BLOCK_MINED.get(this));
       //调用了Player的causeFoodExhaustion方法，并将0.005F作为参数传入。这个方法的目的是使玩家因为挖掘方块而感到饥饿，0.005F是饥饿的程度。
      p_49828_.causeFoodExhaustion(0.005F);
      //Forge: Don't drop xp as part of the resources as it is handled by the patches in ServerPlayerGameMode#destroyBlock
       //这个方法的目的是在玩家挖掘方块时掉落资源，false表示不掉落经验值。注释中提到的Forge是一个用于修改和扩展Minecraft的模组，它在ServerPlayerGameMode#destroyBlock方法中处理了经验值的掉落。
      dropResources(p_49830_, p_49827_, p_49829_, p_49831_, p_49828_, p_49832_, false);
   }

//定义了一个公共的、静态的方法getDrops，它接受四个参数：一个BlockState类型的参数p_49870_，一个ServerLevel类型的参数p_49871_，一个BlockPos类型的参数p_49872_，以及一个可以为null的BlockEntity类型的参数p_49873_。这个方法返回一个ItemStack类型的列表。
   public static List<ItemStack> getDrops(BlockState p_49870_, ServerLevel p_49871_, BlockPos p_49872_, @Nullable BlockEntity p_49873_) {
      LootParams.Builder lootparams$builder = new LootParams.Builder(p_49871_)
         .withParameter(LootContextParams.ORIGIN, Vec3.atCenterOf(p_49872_))
         .withParameter(LootContextParams.TOOL, ItemStack.EMPTY)
         .withOptionalParameter(LootContextParams.BLOCK_ENTITY, p_49873_);
      //调用了blockstate的getDrops方法，并将lootparams$builder作为参数传入，然后返回这个方法的结果。
      return p_49870_.getDrops(lootparams$builder);
   }
//定义了一个公共的、静态的方法getDrops，它接受六个参数：一个BlockState类型的参数p_49875_，一个ServerLevel类型的参数p_49876_，一个BlockPos类型的参数p_49877_，一个可以为null的BlockEntity类型的参数p_49878_，一个可以为null的Entity类型的参数p_49879_，以及一个ItemStack类型的参数p_49880_。这个方法返回一个ItemStack类型的列表。
   public static List<ItemStack> getDrops(
      BlockState p_49875_, ServerLevel p_49876_, BlockPos p_49877_, @Nullable BlockEntity p_49878_, @Nullable Entity p_49879_, ItemStack p_49880_
   ) {
      LootParams.Builder lootparams$builder = new LootParams.Builder(p_49876_)
         .withParameter(LootContextParams.ORIGIN, Vec3.atCenterOf(p_49877_))
         .withParameter(LootContextParams.TOOL, p_49880_)
         .withOptionalParameter(LootContextParams.THIS_ENTITY, p_49879_)
         .withOptionalParameter(LootContextParams.BLOCK_ENTITY, p_49878_);
       //调用了blockstate的getDrops方法，并将lootparams$builder作为参数传入，然后返回这个方法的结果。
      return p_49875_.getDrops(lootparams$builder);
   }


```



# blockstate部分代码

```java
      public List<ItemStack> getDrops(LootParams.Builder p_287688_) {
          //调用了this.getBlock()的getDrops方法，并将this.asState()和p_287688_作为参数传入，然后返回这个方法的结果。这个方法的目的是获取方块的掉落物
         return this.getBlock().getDrops(this.asState(), p_287688_);
      }

//定义了一个公共的、已被弃用的方法getDrops，它接受两个参数：一个BlockState类型的参数p_287732_和一个LootParams.Builder类型的参数p_287596_，并返回一个ItemStack类型的列表
   @Deprecated
   public List<ItemStack> getDrops(BlockState p_287732_, LootParams.Builder p_287596_) {
       //调用了this的getLootTable方法，并将返回的结果赋值给ResourceLocation类型的变量resourcelocation。这个方法的目的是获取方块的掉落物表。
      ResourceLocation resourcelocation = this.getLootTable();
       //检查resourcelocation是否等于BuiltInLootTables.EMPTY。如果等于，则返回一个空的列表
      if (resourcelocation == BuiltInLootTables.EMPTY) {
         return Collections.emptyList();
      } else {
          //调用了p_287596_的withParameter方法，并将LootContextParams.BLOCK_STATE和p_287732_作为参数传入，然后调用create方法并将LootContextParamSets.BLOCK作为参数传入，最后将返回的结果赋值给LootParams类型的变量lootparams。这个方法的目的是创建一个新的LootParams对象，并设置其参数。
         LootParams lootparams = p_287596_.withParameter(LootContextParams.BLOCK_STATE, p_287732_).create(LootContextParamSets.BLOCK);
          //调用了lootparams的getLevel方法，并将返回的结果赋值给ServerLevel类型的变量serverlevel。这个方法的目的是获取LootParams对象的世界。
         ServerLevel serverlevel = lootparams.getLevel();
          //调用了serverlevel.getServer()的getLootData方法，然后调用getLootTable方法并将resourcelocation作为参数传入，最后将返回的结果赋值给LootTable类型的变量
         LootTable loottable = serverlevel.getServer().getLootData().getLootTable(resourcelocation);
          //码调用了loottable的getRandomItems方法，并将lootparams作为参数传入，然后返回这个方法的结果。这个方法的目的是获取随机的掉落物。
         return loottable.getRandomItems(lootparams);
      }
   }
```



# ServerPlayerGameMode类部分代码

```java
//这个方法是在玩家挖掘方块时被调用的
   public boolean destroyBlock(BlockPos p_9281_) {
       //调用了this.level的getBlockState方法，并将p_9281_作为参数传入，然后将返回的结果赋值给BlockState类型的变量blockstate。这个方法的目的是获取指定位置的方块状态。
      BlockState blockstate = this.level.getBlockState(p_9281_);
       //调用了net.neoforged.neoforge.common.CommonHooks.onBlockBreakEvent方法，并将level，gameModeForPlayer，player，p_9281_作为参数传入，然后将返回的结果赋值给整型变量exp。这个方法的目的是处理玩家挖掘方块时的事件。
      int exp = net.neoforged.neoforge.common.CommonHooks.onBlockBreakEvent(level, gameModeForPlayer, player, p_9281_);
       //检查exp是否等于-1。如果等于，则返回false。
      if (exp == -1) {
         return false;
      } else {
          //调用了this.level的getBlockEntity方法，并将p_9281_作为参数传入，然后将返回的结果赋值给BlockEntity类型的变量blockentity。这个方法的目的是获取指定位置的方块实体。
         BlockEntity blockentity = this.level.getBlockEntity(p_9281_);
          //调用了blockstate的getBlock方法，并将返回的结果赋值给Block类型的变量block。这个方法的目的是获取方块状态对应的方块。
         Block block = blockstate.getBlock();
          //检查block是否是GameMasterBlock类的实例，以及this.player是否可以使用GameMasterBlock。如果block是GameMasterBlock类的实例，但this.player不能使用GameMasterBlock，则调用this.level的sendBlockUpdated方法，并返回false。
         if (block instanceof GameMasterBlock && !this.player.canUseGameMasterBlocks()) {
            this.level.sendBlockUpdated(p_9281_, blockstate, blockstate, 3);
            return false;
             //调用了player.getMainHandItem()的onBlockStartBreak方法，并将p_9281_和player作为参数传入。如果该方法返回true，则返回false。
         } else if (player.getMainHandItem().onBlockStartBreak(p_9281_, player)) {
            return false;
             //
         } else if (this.player.blockActionRestricted(this.level, p_9281_, this.gameModeForPlayer)) {
            return false;
         } else {
             //查this是否是创造模式。如果是，则调用removeBlock方法并将p_9281_和false作为参数传入，然后返回true。
            if (this.isCreative()) {
               removeBlock(p_9281_, false);
               return true;
            } else {
                //调用了this.player的getMainHandItem方法，并将返回的结果赋值给ItemStack类型的变量itemstack。这个方法的目的是获取玩家主手中的物品。
               ItemStack itemstack = this.player.getMainHandItem();
               ItemStack itemstack1 = itemstack.copy();
                //调用了blockstate的canHarvestBlock方法，并将this.level，p_9281_，和this.player作为参数传入，然后将返回的结果赋值给布尔类型的变量flag1。这个方法的目的是检查玩家是否可以采集这个方块。
               boolean flag1 = blockstate.canHarvestBlock(this.level, p_9281_, this.player); // previously player.hasCorrectToolForDrops(blockstate)
                //调用了itemstack的mineBlock方法，并将this.level，blockstate，p_9281_，和this.player作为参数传入。这个方法的目的是在挖掘方块时使用物品。
               itemstack.mineBlock(this.level, blockstate, p_9281_, this.player);
               if (itemstack.isEmpty() && !itemstack1.isEmpty())
                   //用net.neoforged.neoforge.event.EventHooks.onPlayerDestroyItem方法，并将this.player，itemstack1，和InteractionHand.MAIN_HAND作为参数传入。这个方法的目的是在玩家销毁物品时触发事件。
                  net.neoforged.neoforge.event.EventHooks.onPlayerDestroyItem(this.player, itemstack1, InteractionHand.MAIN_HAND);
                //调用了removeBlock方法并将p_9281_和flag1作为参数传入，然后将返回的结果赋值给布尔类型的变量flag。这个方法的目的是移除一个方块。
               boolean flag = removeBlock(p_9281_, flag1);

               if (flag && flag1) {
                  block.playerDestroy(this.level, this.player, p_9281_, blockstate, blockentity, itemstack1);
               }
//检查flag和flag1是否都为true。如果是，则调用block的playerDestroy方法，并将this.level，this.player，p_9281_，blockstate，blockentity，和itemstack1作为参数传入。这个方法的目的是处理玩家挖掘方块时的行为。
               if (flag && exp > 0)
                  blockstate.getBlock().popExperience(level, p_9281_, exp);

               return true;
            }
         }
      }
   }
```

