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
public class LootParams {
  ServerLevel ; // 服务器世界对象
     Map<LootContextParam<?>, Object> ; // ,固定参数map
     Map<ResourceLocation, LootParams.DynamicDrop> dynamicDrops; //动态掉落map
      luck; //,幸运值
        
//从固定参数map获取指定参数,如果不存在抛出异常
getParameter(LootContextParam<T> ) {
   }
    
//,从固定参数map获取指定参数,如果不存在返回null
getOptionalParameter(LootContextParam<T> ) {
   }
    
    
//从固定参数map获取指定参数,如果不存在返回null
getParamOrNull( ) {
   }
    
//addDynamicDrops方法,向dynamicDrops动态掉落map中添加指定资源位置的动态掉落
    addDynamicDrops() {

   }
    
//返回luck幸运值属性
   float getLuck() {
   }
//内部静态类Builder
   public static class Builder {
       ServerLevel level;//服务器世界对象
      Map<LootContextParam<?>, Object> params //固定参数map,初始化为空map
      Map<ResourceLocation, LootParams.DynamicDrop> dynamicDrops //动态掉落map,初始化为空map
      private float luck;//幸运值
//类构造函数,参数为服务器世界对象,初始化level属性
      public Builder(ServerLevel ) {
      }
//方法,返回level属性
      public ServerLevel getLevel() {
      }
//withParameter方法,向固定参数map中put指定参数
       <T> LootParams.Builder withParameter(LootContextParam<T> ,) {
      }
//可选地向固定参数map中put指定参数,如果值为空则remove
       <T> LootParams.Builder withOptionalParameter(LootContextParam<T>, @Nullable T ) {

      }
//getParameter方法,从固定参数map获取指定参数,如果不存在抛出异常
       <T> T getParameter(LootContextParam<T> ) {

      }
//getOptionalParameter方法,从固定参数map获取指定参数,如果不存在返回null

      @Nullable
       <T> T getOptionalParameter(LootContextParam<T> ) {
      }
//withDynamicDrop方法,向dynamicDrops动态掉落map中put指定资源位置的动态掉落,不允许重复
       LootParams.Builder withDynamicDrop(ResourceLocation , LootParams.DynamicDrop ) {

      }
//设置luck幸运值属性
      public LootParams.Builder withLuck(float ) {
      }
// create方法,通过LootContextParamSet验证参数,然后构造LootParams返回

       //create方法接受一个LootContextParamSet类型的参数，首先计算出params映射中的键与p_287701_的getAllowed方法返回的集合的差集，如果这个差集不为空，则抛出IllegalArgumentException异常，然后计算出p_287701_的getRequired方法返回的集合与params映射中的键的差集，如果这个差集不为空，则抛出IllegalArgumentException异常，否则返回一个新的LootParams对象
      public LootParams create(LootContextParamSet ) {

      }
   }
//这个方法接受一个Consumer<ItemStack>类型的参数。
   @FunctionalInterface
   public interface DynamicDrop {
      void add(Consumer<ItemStack>);
   }
}

```

# LootContextParam 类

主要目的是定义一个可以存储资源位置的参数类，这个类可以被用于表示游戏中的各种上下文参数，例如掉落物的位置，玩家的位置等等

```java

//主要目的是定义一个可以存储资源位置的参数类，这个类可以被用于表示游戏中的各种上下文参数，例如掉落物的位置，玩家的位置等等
  LootContextParam<T> {
    // 定义了一个私有的、不可变的ResourceLocation类型的成员变量name。这个变量在这个类的构造函数中被初始化，并且在类的其他方法中可以被访问。
      name;
//LootContextParam类的构造函数，它接受一个ResourceLocation类型的参数p_81283_，并将其赋值给成员变量name。
    LootContextParam( ) {
   }
//返回值为ResourceLocation类型。这个方法返回成员变量name的值。
    ResourceLocation getName() {
   }
//这个方法的目的是提供一个表示这个对象的字符串，当我们试图将这个对象转换为字符串时，就会调用这个方法。
    String toString() {
   }
}

```



# LootContextParams类

定义了一个名为`LootContextParams`的类，这个类用于创建各种类型的掉落物上下文参数

```java

public class LootContextParams {
"last_damage_player"
"damage_source"
"killer_entity"
"direct_killer_entity"
"origin"
"block_state"
"block_entity"
"tool"
"explosion_radius"

}

```



# block类部分代码

```java

      dropResources() {
       //是否是ServerLevel类的实例
          //先调用getDrops方法并，然后对返回的列表进行遍历，popResource方法的目的是在世界中掉落一个物品。
          //用了pspawnAfterBreak方法。这个方法的目的是在方块被挖掘后生成一些实体，例如掉落的物品或者生成的实体。
      }
   }

     void dropResources() {
      
   }

     void dropResources(
   ) {
       // 掉落经验
   }

// 这个方法是在玩家挖掘方块时被调用的
     playerDestroy() {
       //调用了Player的causeFoodExhaustion方法，并将0.005F作为参数传入。这个方法的目的是使玩家因为挖掘方块而感到饥饿，0.005F是饥饿的程度。
      //Forge: Don't drop xp as part of the resources as it is handled by the patches in ServerPlayerGameMode#destroyBlock
       //这个方法的目的是在玩家挖掘方块时掉落资源，false表示不掉落经验值。注释中提到的Forge是一个用于修改和扩展Minecraft的模组，它在ServerPlayerGameMode#destroyBlock方法中处理了经验值的掉落。
   }

//定义了一个公共的、静态的方法getDrops，它接受四个参数：一个BlockState类型的参数，一个ServerLevel类型的参数，一个BlockPos类型的参数，以及一个可以为null的BlockEntity类型的参数。这个方法返回一个ItemStack类型的列表。
// 返回对应的方块的掉落物List
     List<ItemStack> getDrops() {
      //调用了blockstate的getDrops方法，并将lootparams$builder作为参数传入，然后返回这个方法的结果。
   }
//定义了一个公共的、静态的方法getDrops，这个方法返回一个ItemStack类型的列表。
// 返回对应的方块的掉落物List 这几个都是重载方法
     List<ItemStack> getDrops
      BlockState , ServerLevel , BlockPos ,  BlockEntity , Entity , ItemStack
   ) {
       //调用了blockstate的getDrops方法，并将lootparams$builder作为参数传入，然后返回这个方法的结果。
   }


```



# blockstate部分代码

```java
        getDrops() {
          //调用了this.getBlock()的getDrops方法，并将this.asState()和p_287688_作为参数传入，然后返回这个方法的结果。这个方法的目的是获取方块的掉落物
      }

//定义了一个公共的、已被弃用的方法getDrops，它接受两个参数：一个BlockState类型的和一个LootParams.Builder类型的，并返回一个ItemStack类型的列表
     getDrops(  ) {
       //调用了this的getLootTable方法，并将返回的结果赋值给ResourceLocation类型的变量resourcelocation。这个方法的目的是获取方块的掉落物表。
       //检查resourcelocation是否等于BuiltInLootTables.EMPTY。如果等于，则返回一个空的列表
      if () {
      } else {
          //这个方法的目的是创建一个新的LootParams对象，并设置其参数。
          //这个方法的目的是获取LootParams对象的世界。
          //最后将返回的结果赋值给LootTable类型的变量
          //码调用了loottable的getRandomItems方法，并将lootparams作为参数传入，然后返回这个方法的结果。这个方法的目的是获取随机的掉落物。
      }
   }
```



# ServerPlayerGameMode类部分代码

```java
//这个方法是在玩家挖掘方块时被调用的
     destroyBlock(BlockPos) {
       //调用了this.level的getBlockState方法，然后将返回的结果赋值给BlockState类型的变量。这个方法的目的是获取指定位置的方块状态。
       //调用了net.neoforged.neoforge.common.CommonHooks.onBlockBreakEvent方法，并将level，gameModeForPlayer，player作为参数传入，然后将返回的结果赋值给整型变量exp。这个方法的目的是处理玩家挖掘方块时的事件。
       //检查exp是否等于-1。如果等于，则返回false。
      if () {
      } else {
          //获取指定位置的方块实体。
          //获取方块状态对应的方块。
          //检查block是否是GameMasterBlock类的实例，以及this.player是否可以使用GameMasterBlock。如果block是GameMasterBlock类的实例，但this.player不能使用GameMasterBlock，则调用this.level的sendBlockUpdated方法，并返回false。
         if () {
             //查看是否能破坏方块产生掉落，不能情况下直接返回
         } else if () {
            return false;
             
         } else if () {
         } else {
             //查this是否是创造模式。如果是，则调用removeBlock方法并将p_9281_和false作为参数传入，然后返回true。
            if () {
            } else {
                //这个方法的目的是获取玩家主手中的物品。
                //这个方法的目的是检查玩家是否可以采集这个方块。
                //这个方法的目的是在挖掘方块时使用物品。
               if ()
                   //这个方法的目的是在玩家销毁物品时触发事件。
                //这个方法的目的是移除一个方块。

               if ()) {
               }
//检查flag和flag1是否都为true。如果是，则调用block的playerDestroy方法，并将this.level，this.player，p_9281_，blockstate，blockentity，和itemstack1作为参数传入。这个方法的目的是处理玩家挖掘方块时的行为。 
                // true就获得掉落物表并掉落
               if ()

            }
         }
      }
   }
```

