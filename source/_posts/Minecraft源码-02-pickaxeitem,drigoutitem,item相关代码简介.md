---
title: pickaxe代码简介
date: 2023-11-12 14:18:01
tags:
- 我的世界
- 源码
cover: https://w.wallhaven.cc/full/yx/wallhaven-yxk6k7.jpg
---



# PickaxeItem类

```java
//定义了一个名为PickaxeItem的类，这个类继承自DiggerItem。
public class PickaxeItem extends DiggerItem {
    //它接受四个参数：一个Tier对象，一个整数，一个浮点数和一个Item.Properties对象。这个构造函数用于初始化PickaxeItem对象。
   public PickaxeItem(Tier p_42961_, int p_42962_, float p_42963_, Item.Properties p_42964_) {
       //码调用了父类DiggerItem的构造函数，传入了整数的浮点数形式，浮点数，Tier对象，BlockTags.MINEABLE_WITH_PICKAXE和Item.Properties对象。
      super((float)p_42962_, p_42963_, p_42961_, BlockTags.MINEABLE_WITH_PICKAXE, p_42964_);
   }
//义了一个名为canPerformAction的公共方法，它接受两个参数：一个ItemStack对象和一个net.neoforged.neoforge.common.ToolAction对象。这个方法用于检查这个工具是否可以执行给定的动作。
   @Override
   public boolean canPerformAction(ItemStack stack, net.neoforged.neoforge.common.ToolAction toolAction) {
       //行代码检查ToolActions.DEFAULT_PICKAXE_ACTIONS是否包含toolAction。如果包含，则返回true；否则，返回false。这表示如果toolAction是钻石工具的默认动作之一，那么这个工具就可以执行这个动作。
      return net.neoforged.neoforge.common.ToolActions.DEFAULT_PICKAXE_ACTIONS.contains(toolAction);
   }
}

```





# DiggerItem类

```java
package net.minecraft.world.item;

import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.ImmutableMultimap.Builder;
import net.minecraft.core.BlockPos;
import net.minecraft.tags.BlockTags;
import net.minecraft.tags.TagKey;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.ai.attributes.Attribute;
import net.minecraft.world.entity.ai.attributes.AttributeModifier;
import net.minecraft.world.entity.ai.attributes.Attributes;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;

// 类继承自TieredItem
// 实现了Vanishable接口
public class DiggerItem extends TieredItem implements Vanishable {
   // 存储与此工具相关的方块类型
   private final TagKey<Block> blocks;
    // 存储此工具的速度
   protected final float speed;
    //存储此工具的基础攻击伤害
   private final float attackDamageBaseline;
    //存储此工具的默认属性修改器
   private final Multimap<Attribute, AttributeModifier> defaultModifiers;
// 两个浮点数，一个Tier对象，一个TagKey<Block>对象和一个Item.Properties对象。这个构造函数用于初始化DiggerItem对象。
   public DiggerItem(float p_204108_, float p_204109_, Tier p_204110_, TagKey<Block> p_204111_, Item.Properties p_204112_) {
       // 用了父类TieredItem的构造函数，传入了Tier对象和Item.Properties对象。
      super(p_204110_, p_204112_);
       // 将传入的TagKey<Block>对象赋值给blocks字段。
      this.blocks = p_204111_;
       // 行代码调用Tier对象的getSpeed方法，并将返回的结果赋值给speed字段。
      this.speed = p_204110_.getSpeed();
       // 传入的浮点数与Tier对象的getAttackDamageBonus方法返回的结果相加，并将结果赋值给attackDamageBaseline字段。
      this.attackDamageBaseline = p_204108_ + p_204110_.getAttackDamageBonus();
       // Multimap.Builder对象，这是一个用于构建Multimap的建造者对象。Multimap`是一种特殊的映射，它允许每个键映射到多个值
       //它被用来存储属性修改器。
      Builder<Attribute, AttributeModifier> builder = ImmutableMultimap.builder();
      builder.put(
          //行代码向Multimap中添加了一个新的属性修改器。这个属性修改器修改的是攻击伤害属性，它的值是attackDamageBaseline字段的值，操作类型是加法。
         Attributes.ATTACK_DAMAGE,
         new AttributeModifier(BASE_ATTACK_DAMAGE_UUID, "Tool modifier", (double)this.attackDamageBaseline, AttributeModifier.Operation.ADDITION)
      );
       //行代码向Multimap中添加了一个新的属性修改器。这个属性修改器修改的是攻击速度属性，它的值是传入的浮点数，操作类型是加法。
      builder.put(
         Attributes.ATTACK_SPEED, new AttributeModifier(BASE_ATTACK_SPEED_UUID, "Tool modifier", (double)p_204109_, AttributeModifier.Operation.ADDITION)
      );
       //码调用builder.build()方法来构建Multimap，并将结果赋值给defaultModifiers字段。
      this.defaultModifiers = builder.build();
   }
//它接受两个参数：一个ItemStack对象和一个BlockState对象。这个方法用于获取破坏速度。
   @Override
   public float getDestroySpeed(ItemStack p_41004_, BlockState p_41005_) {
       //检查BlockState对象是否与blocks字段相匹配。如果匹配，则返回speed字段的值；否则，返回1.0F。
      return p_41005_.is(this.blocks) ? this.speed : 1.0F;
   }
//它接受三个参数：一个ItemStack对象，两个LivingEntity对象。这个方法用于攻击敌人。
   @Override
   public boolean hurtEnemy(ItemStack p_40994_, LivingEntity p_40995_, LivingEntity p_40996_) {
       //行代码调用ItemStack对象的hurtAndBreak方法来损坏物品。如果物品被损坏，它会广播一个破坏事件。
      p_40994_.hurtAndBreak(2, p_40996_, p_41007_ -> p_41007_.broadcastBreakEvent(EquipmentSlot.MAINHAND));
      return true;
   }
//它接受五个参数：一个ItemStack对象，一个Level对象，一个BlockState对象，一个BlockPos对象和一个LivingEntity对象。这个方法用于破坏方块。
   @Override
   public boolean mineBlock(ItemStack p_40998_, Level p_40999_, BlockState p_41000_, BlockPos p_41001_, LivingEntity p_41002_) {
       //检查Level对象是否为客户端，以及BlockState对象的破坏速度是否不为0.0F。如果这两个条件都满足，则执行下一行代码。
      if (!p_40999_.isClientSide && p_41000_.getDestroySpeed(p_40999_, p_41001_) != 0.0F) {
          //调用ItemStack对象的hurtAndBreak方法来损坏物品。如果物品被损坏，它会广播一个破坏事件。
         p_40998_.hurtAndBreak(1, p_41002_, p_40992_ -> p_40992_.broadcastBreakEvent(EquipmentSlot.MAINHAND));
      }

      return true;
   }
//这行代码定义了一个名为getDefaultAttributeModifiers的公共方法，它接受一个EquipmentSlot对象作为参数。这个方法用于获取默认属性修改器。
   @Override
   public Multimap<Attribute, AttributeModifier> getDefaultAttributeModifiers(EquipmentSlot p_40990_) {
       //这行代码检查EquipmentSlot对象是否为主手槽。如果是，则返回defaultModifiers字段的值；否则，调用父类的getDefaultAttributeModifiers方法。
      return p_40990_ == EquipmentSlot.MAINHAND ? this.defaultModifiers : super.getDefaultAttributeModifiers(p_40990_);
   }
//为getAttackDamage的公共方法，这个方法用于获取攻击伤害
   public float getAttackDamage() {
      return this.attackDamageBaseline;
   }
//代码定义了一个名为isCorrectToolForDrops的公共方法，它接受一个BlockState对象作为参数。这个方法用于检查这个工具是否适合破坏给定的方块。
   @Override
   @Deprecated // FORGE: Use stack sensitive variant below
   public boolean isCorrectToolForDrops(BlockState p_150816_) {
       //行代码检查工具的等级是否已经排序。如果已经排序，则执行下一行代码。
      if (net.neoforged.neoforge.common.TierSortingRegistry.isTierSorted(getTier())) {
          //代码检查工具的等级是否适合破坏给定的方块，以及方块是否与blocks字段相匹配满足，则返回true；否则，返回false。
         return net.neoforged.neoforge.common.TierSortingRegistry.isCorrectTierForDrops(getTier(), p_150816_) && p_150816_.is(this.blocks);
      }
       //这行代码调用getTier方法获取工具的等级，并将结果赋值给i。
      int i = this.getTier().getLevel();
       //这行代码检查i是否小于3，以及BlockState对象是否属于需要钻石工具的方块标签。如果这两个条件都满足，则执行下一行代码。
      if (i < 3 && p_150816_.is(BlockTags.NEEDS_DIAMOND_TOOL)) {
          //这个工具不适合破坏这个方块。
         return false;
          //这行代码检查i是否小于2，以及BlockState对象是否属于需要铁工具的方块标签。如果这两个条件都满足，则执行下一行代码。
      } else if (i < 2 && p_150816_.is(BlockTags.NEEDS_IRON_TOOL)) {
         return false;
      } else {
          //这行代码检查i是否小于1，以及BlockState对象是否属于需要石头工具的方块标签。如果这两个条件都满足，则返回false；否则，检查BlockState对象是否与blocks字段相匹配，并返回结果。
         return i < 1 && p_150816_.is(BlockTags.NEEDS_STONE_TOOL) ? false : p_150816_.is(this.blocks);
      }
   }

   // FORGE START
    //这行代码定义了一个名为isCorrectToolForDrops的公共方法，它接受两个参数：一个ItemStack对象和一个BlockState对象。这个方法用于检查这个工具是否适合破坏给定的方块。
   @Override
   public boolean isCorrectToolForDrops(ItemStack stack, BlockState state) {
       //这行代码检查BlockState对象是否与blocks字段相匹配，以及工具的等级是否适合破坏这个方块。如果这两个条件都满足，则返回true；否则，返回false。
      return state.is(blocks) && net.neoforged.neoforge.common.TierSortingRegistry.isCorrectTierForDrops(getTier(), state);
   }
}

```



# TieredItem类

```java
//代码定义了一个名为TieredItem的公共类，这个类继承自Item类。
public class TieredItem extends Item {
    //行代码定义了一个名为tier的私有常量字段，它的类型是Tier。这个字段可能用于存储此物品的挖掘等级。
   private final Tier tier;
//这行代码定义了一个名为TieredItem的公共构造函数，它接受两个参数：一个Tier对象和一个Item.Properties对象。这个构造函数用于初始化TieredItem对象。
   public TieredItem(Tier p_43308_, Item.Properties p_43309_) {
       //代码调用了父类Item的构造函数，传入了一个新的Item.Properties对象，这个对象的默认耐久度被设置为工具的使用次数。
      super(p_43309_.defaultDurability(p_43308_.getUses()));
       //行代码将传入的Tier对象赋值给tier字段。
      this.tier = p_43308_;
   }
//定义了一个名为getTier的公共方法，这个方法用于获取物品的等级。
   public Tier getTier() {
       //行代码返回tier字段的值，表示物品的等级。
      return this.tier;
   }
//这行代码定义了一个名为getEnchantmentValue的公共方法，这个方法用于获取物品的附魔值。
   @Override
   public int getEnchantmentValue() {
       //这行代码调用Tier对象的getEnchantmentValue方法，并将返回的结果作为附魔值返回。
      return this.tier.getEnchantmentValue();
   }
//定义了一个名为isValidRepairItem的公共方法，它接受两个ItemStack对象作为参数。这个方法用于检查一个物品是否可以修复另一个物品。
   @Override
   public boolean isValidRepairItem(ItemStack p_43311_, ItemStack p_43312_) {
       //这行代码检查Tier对象的修复材料是否与传入的ItemStack对象匹配，如果匹配，则返回true；否则，调用父类的isValidRepairItem方法。如果父类的方法也返回true，那么这个方法就会返回true；否则，返回false。
      return this.tier.getRepairIngredient().test(p_43312_) || super.isValidRepairItem(p_43311_, p_43312_);
   }
}
```



# Item 类

```java
//定义了一个名为Item的类，这个类实现了FeatureElement，ItemLike和IItemExtension接口
public class Item implements FeatureElement, ItemLike, net.neoforged.neoforge.common.extensions.IItemExtension {
    //行代码定义了一个名为LOGGER的私有静态最终字段，它的类型是Logger。这个字段用于记录日志。
   private static final Logger LOGGER = LogUtils.getLogger();
    //代码定义了一个名为BY_BLOCK的公共静态最终字段，它的类型是Map<Block, Item>。这个字段用于将方块映射到对应的物品。
   public static final Map<Block, Item> BY_BLOCK = net.neoforged.neoforge.registries.GameData.getBlockItemMap();
    //定义了一个名为BASE_ATTACK_DAMAGE_UUID的受保护的静态最终字段，它的类型是UUID。这个字段用于存储基础攻击伤害的UUID。
   protected static final UUID BASE_ATTACK_DAMAGE_UUID = UUID.fromString("CB3F55D3-645C-4F38-A497-9C13A33DB5CF");
    //定义了一个名为BASE_ATTACK_SPEED_UUID的受保护的静态最终字段，它的类型是UUID。这个字段用于存储基础攻击速度的UUID。
   protected static final UUID BASE_ATTACK_SPEED_UUID = UUID.fromString("FA233E1C-4180-4865-B01B-BCCE9785ACA3");
    //代码定义了一个名为MAX_STACK_SIZE的公共静态最终字段，它的类型是int。这个字段用于存储物品的最大堆叠大小。
   public static final int MAX_STACK_SIZE = 64;
    //定义了一个名为EAT_DURATION的公共静态最终字段，它的类型是int。这个字段用于存储吃东西的持续时间
   public static final int EAT_DURATION = 32;
    //定义了一个名为MAX_BAR_WIDTH的公共静态最终字段，它的类型是int。这个字段用于存储物品耐久条的最大宽度。
   public static final int MAX_BAR_WIDTH = 13;
    //定义了一个私有的最终的Holder.Reference<Item>对象，它被命名为builtInRegistryHolder。这个对象是通过调用BuiltInRegistries.ITEM.createIntrusiveHolder(this)方法创建的，这个方法返回一个Holder.Reference<Item>对象，这个对象包含了当前的Item对象
   private final Holder.Reference<Item> builtInRegistryHolder = BuiltInRegistries.ITEM.createIntrusiveHolder(this);
    //定义了一个私有的最终的Rarity对象，它被命名为rarity。这个对象代表了物品的稀有度。
   private final Rarity rarity;
    //定义了一个私有的最终的整数，它被命名为maxStackSize。这个整数代表了物品的最大堆叠数量。
   private final int maxStackSize;
    //定义了一个私有的最终的整数，它被命名为maxDamage。这个整数代表了物品的最大耐久度。
   private final int maxDamage;
    //这行代码定义了一个私有的最终的布尔值，它被命名为isFireResistant。这个布尔值表示了物品是否对火有抵抗力。
   private final boolean isFireResistant;
    //定义了一个私有的最终的Item对象，它被命名为craftingRemainingItem。这个对象代表了在制作过程中保留的物品。这个对象可能为null。
   @Nullable
   private final Item craftingRemainingItem;
    // 定义了一个私有的String对象，它被命名为descriptionId。这个字符串代表了物品的描述ID。这个对象可能为null。
   @Nullable
   private String descriptionId;
    // 定义了一个私有的最终的FoodProperties对象，它被命名为foodProperties。这个对象代表了物品的食物属性。这个对象可能为null。
   @Nullable
   private final FoodProperties foodProperties;
    //代码定义了一个私有的最终的FeatureFlagSet对象，它被命名为requiredFeatures。这个对象代表了物品所需的特性标志集。
   private final FeatureFlagSet requiredFeatures;
//它的名字是getId，它接受一个Item对象作为参数，参数的名字是p_41394_。这个方法返回一个整数。：
   public static int getId(Item p_41394_) {
       //它检查p_41394_是否为null。如果p_41394_为null，那么这个表达式就返回0。否则，它调用BuiltInRegistries.ITEM.getId(p_41394_)方法，并返回这个方法的结果。BuiltInRegistries.ITEM.getId(p_41394_)方法返回的是p_41394_在ITEM注册表中的ID。
      return p_41394_ == null ? 0 : BuiltInRegistries.ITEM.getId(p_41394_);
   }
//码定义了一个公共的静态方法，它的名字是byId，它接受一个整数作为参数，参数的名字是p_41446_。这个方法返回一个Item对象。
   public static Item byId(int p_41446_) {
      return BuiltInRegistries.ITEM.byId(p_41446_);
   }
//它的名字是byBlock，它接受一个Block对象作为参数，参数的名字是p_41440_。这个方法返回一个Item对象。这个方法被@Deprecated注解标记，表示这个方法已经被弃用，不建议再使用。
   @Deprecated
   public static Item byBlock(Block p_41440_) {
       //这行代码从BY_BLOCK这个Map中获取键为p_41440_的值，如果这个Map中不存在键为p_41440_的值，那么就返回Items.AIR。
      return BY_BLOCK.getOrDefault(p_41440_, Items.AIR);
   }
    //它接受一个Item.Properties对象作为参数，参数的名字是p_41383_。
   public Item(Item.Properties p_41383_) {
      this.rarity = p_41383_.rarity;
      this.craftingRemainingItem = p_41383_.craftingRemainingItem;
      this.maxDamage = p_41383_.maxDamage;
      this.maxStackSize = p_41383_.maxStackSize;
      this.foodProperties = p_41383_.foodProperties;
      this.isFireResistant = p_41383_.isFireResistant;
      this.requiredFeatures = p_41383_.requiredFeatures;
       //查SharedConstants.IS_RUNNING_IN_IDE是否为true。如果为true，那么它获取this对象的类名，并检查这个类名是否以"Item"结尾。如果不是，那么它记录一个错误。
      if (SharedConstants.IS_RUNNING_IN_IDE) {
         String s = this.getClass().getSimpleName();
         if (!s.endsWith("Item")) {
            LOGGER.error("Item classes should end with Item and {} doesn't.", s);
         }
      }
      this.canRepair = p_41383_.canRepair;
      initClient();
   }
//定义了一个公共的方法，这个方法被@Deprecated注解标记，表示这个方法已经被弃用，不建议再使用。这个方法的名字是builtInRegistryHolder，没有参数，返回一个Holder.Reference<Item>对象。
   @Deprecated
   public Holder.Reference<Item> builtInRegistryHolder() {
      return this.builtInRegistryHolder;
   }
//定义了一个公共的方法，这个方法的名字是onUseTick，接受四个参数：一个Level对象，一个LivingEntity对象，一个ItemStack对象和一个整数。这个方法没有返回值。
   public void onUseTick(Level p_41428_, LivingEntity p_41429_, ItemStack p_41430_, int p_41431_) {
   }
//定义了一个公共的方法，这个方法的名字是onDestroyed，接受一个ItemEntity对象作为参数。这个方法被@Deprecated注解标记，表示这个方法已经被弃用，不建议再使用。这个方法没有返回值。
   /** @deprecated Forge: {@link IItemExtension#onDestroyed(ItemEntity, DamageSource) Use damage source sensitive version} */
   @Deprecated
   public void onDestroyed(ItemEntity p_150887_) {
   }
//定义了一个公共的方法，这个方法的名字是verifyTagAfterLoad，接受一个CompoundTag对象作为参数。这个方法没有返回值。
   public void verifyTagAfterLoad(CompoundTag p_150898_) {
   }
//定义了一个公共的方法，这个方法的名字是canAttackBlock，接受四个参数：一个BlockState对象，一个Level对象，一个BlockPos对象和一个Player对象。这个方法返回一个布尔值。
    //检查一个物品是否可以攻击一个方块。
   public boolean canAttackBlock(BlockState p_41441_, Level p_41442_, BlockPos p_41443_, Player p_41444_) {
      return true;
   }
//这个方法的名字是asItem，没有参数，返回一个Item对象。这个方法被@Override注解标记，表示这个方法覆盖了父类或者接口中的方法。
   @Override
   public Item asItem() {
      return this;
   }
//定义了一个公共的方法，这个方法的名字是useOn，接受一个UseOnContext对象作为参数。这个方法返回一个InteractionResult对象。
   public InteractionResult useOn(UseOnContext p_41427_) {
       //行代码返回InteractionResult.PASS，表示物品使用操作被跳过。
       //一些物品可能会被使用在方块上（例如放置物品），而其他物品可能不会。但在这个例子中，由于方法直接返回了InteractionResult.PASS，所以我们可以说这个物品不能被使用在任何方块上。
      return InteractionResult.PASS;
   }
//定义了一个公共的方法，这个方法的名字是getDestroySpeed，接受两个参数：一个ItemStack对象和一个BlockState对象。这个方法返回一个浮点数。
   public float getDestroySpeed(ItemStack p_41425_, BlockState p_41426_) {
       //行代码返回1.0F，表示物品的销毁速度是1.0。
      return 1.0F;
   }
//定义了一个公共的方法，这个方法的名字是use，接受三个参数：一个Level对象，一个Player对象和一个InteractionHand对象。这个方法返回一个InteractionResultHolder<ItemStack>对象。
   public InteractionResultHolder<ItemStack> use(Level p_41432_, Player p_41433_, InteractionHand p_41434_) {
       //代码获取玩家手中的物品，并将其赋值给itemstack。
      ItemStack itemstack = p_41433_.getItemInHand(p_41434_);
       //代码检查itemstack是否是可食用的。
      if (itemstack.isEdible()) {
          //码检查玩家是否可以吃itemstack。
         if (p_41433_.canEat(itemstack.getFoodProperties(p_41433_).canAlwaysEat())) {
             //两行代码表示玩家开始使用物品，并返回一个表示物品被消耗的InteractionResultHolder对象。
            p_41433_.startUsingItem(p_41434_);
            return InteractionResultHolder.consume(itemstack);
         } else {
             // 返回一个表示操作失败的InteractionResultHolder对象，这个对象包含了itemstack。
            return InteractionResultHolder.fail(itemstack);
         }
      } else {
          // 表示如果itemstack不是可食用的，那么返回一个表示操作被跳过的InteractionResultHolder对象，这个对象包含了玩家手中的物品。
         return InteractionResultHolder.pass(p_41433_.getItemInHand(p_41434_));
      }
   }
//这个方法的名字是finishUsingItem，接受三个参数：一个ItemStack对象，一个Level对象和一个LivingEntity对象。
   public ItemStack finishUsingItem(ItemStack p_41409_, Level p_41410_, LivingEntity p_41411_) {
       //一个条件表达式，它检查当前的物品是否是可食用的。如果是，那么它让p_41411_（一个LivingEntity对象）在p_41410_（一个Level对象）中吃p_41409_（一个ItemStack对象），并返回吃完后的物品。如果不是，那么它直接返回p_41409_。
       //如果物品是可食用的，那么这个方法会让生物吃这个物品；如果物品不是可食用的，那么这个方法会直接返回这个物品。
      return this.isEdible() ? p_41411_.eat(p_41410_, p_41409_) : p_41409_;
   }
//定义了一个公共的最终方法，这个方法的名字是getMaxStackSize，没有参数，返回一个整数。这个方法被@Deprecated注解标记，表示这个方法已经被弃用，不建议再使用。注释也提示应该使用物品堆叠对象敏感的版本。
   @Deprecated // Use ItemStack sensitive version.
   public final int getMaxStackSize() {
       //这行代码返回this.maxStackSize，表示物品的最大堆叠数量。
      return this.maxStackSize;
   }
//这个方法的名字是getMaxDamage，没有参数，返回一个整数。这个方法被@Deprecated注解标记，表示这个方法已经被弃用，不建议再使用。注释也提示应该使用物品堆叠对象敏感的版本。
   @Deprecated // Use ItemStack sensitive version.
   public final int getMaxDamage() {
       //代码返回this.maxDamage，表示物品的最大耐久度。
      return this.maxDamage;
   }
//定义了一个公共的方法，这个方法的名字是canBeDepleted，没有参数，返回一个布尔值。
   public boolean canBeDepleted() {
   //表示物品是否可以耗尽。如果物品的最大耐久度大于0，那么这个物品就可以耗尽；否则，这个物品就不能耗尽
       return this.maxDamage > 0;
   }
//定义了一个公共的方法，这个方法的名字是isBarVisible，接受一个ItemStack对象作为参数。这个方法返回一个布尔值。
   public boolean isBarVisible(ItemStack p_150899_) {
       //代码返回p_150899_.isDamaged()，表示物品的耐久条是否可见。如果物品已经损坏，那么这个物品的耐久条就会显示出来；否则，这个物品的耐久条就不会显示出来。
      return p_150899_.isDamaged();
   }
//这个方法的名字是getBarWidth，接受一个ItemStack对象作为参数。这个方法返回一个整数。
   public int getBarWidth(ItemStack p_150900_) {
       //这行代码计算并返回物品的耐久条的宽度。这个宽度是通过物品的损坏值和最大耐久度计算得出的。
      return Math.round(13.0F - (float)p_150900_.getDamageValue() * 13.0F / (float)this.getMaxDamage(p_150900_));
   }
//定义了一个公共的方法，这个方法的名字是getBarColor，接受一个ItemStack对象作为参数。这个方法返回一个整数。
   public int getBarColor(ItemStack p_150901_) {
       //代码首先获取物品的最大耐久度，然后计算物品的损坏程度，最后将损坏程度转换为RGB颜色。
      float stackMaxDamage = this.getMaxDamage(p_150901_);
       //这个方法的作用是获取物品的耐久条的颜色。例如，一些物品的耐久条颜色会随着物品的损坏程度变化而变化，而其他物品的耐久条颜色可能是固定的。
      float f = Math.max(0.0F, (stackMaxDamage - (float)p_150901_.getDamageValue()) / stackMaxDamage);
      return Mth.hsvToRgb(f / 3.0F, 1.0F, 1.0F);
   }
//这个方法的名字是overrideStackedOnOther，接受四个参数：一个ItemStack对象，一个Slot对象，一个ClickAction对象和一个Player对象。这个方法返回一个布尔值
    //方法的作用是覆盖物品在其他物品上的堆叠行为
   public boolean overrideStackedOnOther(ItemStack p_150888_, Slot p_150889_, ClickAction p_150890_, Player p_150891_) {
      return false;
   }
//定义了一个公共的方法，这个方法的名字是overrideOtherStackedOnMe，接受六个参数：两个ItemStack对象，一个Slot对象，一个ClickAction对象，一个Player对象和一个SlotAccess对象。这个方法返回一个布尔值。
    //这行代码返回false，表示默认情况下，这个物品不会覆盖其他物品堆叠在自己上的行为。
   public boolean overrideOtherStackedOnMe(
      ItemStack p_150892_, ItemStack p_150893_, Slot p_150894_, ClickAction p_150895_, Player p_150896_, SlotAccess p_150897_
   ) {
      return false;
   }
//，这个方法的名字是hurtEnemy，接受三个参数：一个ItemStack对象，一个LivingEntity对象和一个LivingEntity对象。这个方法返回一个布尔值。
    //行代码返回false，表示默认情况下，这个物品不会对敌人造成伤害。
   public boolean hurtEnemy(ItemStack p_41395_, LivingEntity p_41396_, LivingEntity p_41397_) {
      return false;
   }
// 这个方法的名字是mineBlock，接受五个参数：一个ItemStack对象，一个Level对象，一个BlockState对象，一个BlockPos对象和一个LivingEntity对象。这个方法返回一个布尔值。
    //行代码返回false，表示默认情况下，这个物品不能破坏方块。
   public boolean mineBlock(ItemStack p_41416_, Level p_41417_, BlockState p_41418_, BlockPos p_41419_, LivingEntity p_41420_) {
      return false;
   }
//代码定义了一个公共的方法，这个方法的名字是isCorrectToolForDrops，接受一个BlockState对象作为参数。这个方法返回一个布尔值。
   public boolean isCorrectToolForDrops(BlockState p_41450_) {
       //返回false，表示默认情况下，这个物品不是破坏方块后获得掉落物的正确工具。
      return false;
   }
//代码定义了一个公共的方法，这个方法的名字是interactLivingEntity，接受四个参数：一个ItemStack对象，一个Player对象，一个LivingEntity对象和一个InteractionHand对象。这个方法返回一个InteractionResult对象。
   public InteractionResult interactLivingEntity(ItemStack p_41398_, Player p_41399_, LivingEntity p_41400_, InteractionHand p_41401_) {
       //代码返回InteractionResult.PASS，表示物品与生物的交互操作被跳过。
      return InteractionResult.PASS;
   }
//代码定义了一个公共的方法，这个方法的名字是getDescription，没有参数，返回一个Component对象。
   public Component getDescription() {
      return Component.translatable(this.getDescriptionId());
   }
//这段代码定义了一个toString方法，这个方法返回物品的注册表键的字符串表示。
   @Override
   public String toString() {
      return BuiltInRegistries.ITEM.getKey(this).toString();
   }
//定义了一个getOrCreateDescriptionId方法，这个方法首先检查descriptionId是否为null。如果descriptionId为null，那么它就使用Util.makeDescriptionId方法创建一个新的描述ID，并将其赋值给descriptionId。然后，这个方法返回descriptionId。
   protected String getOrCreateDescriptionId() {
      if (this.descriptionId == null) {
         this.descriptionId = Util.makeDescriptionId("item", BuiltInRegistries.ITEM.getKey(this));
      }

      return this.descriptionId;
   }
//定义了一个getDescriptionId方法，这个方法调用getOrCreateDescriptionId方法并返回结果。
   public String getDescriptionId() {
      return this.getOrCreateDescriptionId();
   }
//段代码定义了一个getDescriptionId方法，这个方法接受一个ItemStack对象作为参数，但实际上并没有使用这个参数，而是简单地调用另一个getDescriptionId方法并返回结果。
   public String getDescriptionId(ItemStack p_41455_) {
      return this.getDescriptionId();
   }
    //段代码定义了一个shouldOverrideMultiplayerNbt方法，这个方法返回true，表示在多人游戏中，这个物品应该覆盖NBT（Named Binary Tag）数据。

   public boolean shouldOverrideMultiplayerNbt() {
      return true;
   }
// 定义了一个公共的最终方法，这个方法的名字是getCraftingRemainingItem，没有参数，返回一个Item对象。这个方法被@Deprecated注解标记，表示这个方法已经被弃用，不建议再使用。注释也提示应该使用物品堆叠对象敏感的版本。
   @Nullable
   @Deprecated // Use ItemStack sensitive version.
   public final Item getCraftingRemainingItem() {
       // 获得合成后保留物品
      return this.craftingRemainingItem;
   }
//定义了一个公共的方法，这个方法的名字是hasCraftingRemainingItem，没有参数，返回一个布尔值。这个方法被@Deprecated注解标记，表示这个方法已经被弃用，不建议再使用。注释也提示应该使用物品堆叠对象敏感的版本。
   @Deprecated // Use ItemStack sensitive version.
   public boolean hasCraftingRemainingItem() {
       // 是否具有合成保留物品
      return this.craftingRemainingItem != null;
   }
 // 这个方法的名字是inventoryTick，接受五个参数：一个ItemStack对象，一个Level对象，一个Entity对象，一个整数和一个布尔值。这个方法没有返回值。
   public void inventoryTick(ItemStack p_41404_, Level p_41405_, Entity p_41406_, int p_41407_, boolean p_41408_) {
       // 背包中每tick回调
   }
// 这个方法的名字是onCraftedBy，接受三个参数：一个ItemStack对象，一个Level对象和一个Player对象。这个方法没有返回值
   public void onCraftedBy(ItemStack p_41447_, Level p_41448_, Player p_41449_) {
       //方法的作用是处理物品被玩家制作后的行为
   }
//代码定义了一个公共的方法，这个方法的名字是isComplex，没有参数，返回一个布尔值。
   public boolean isComplex() {
      return false;
   }
//代码定义了一个公共的方法，这个方法的名字是getUseAnimation，接受一个ItemStack对象作为参数，返回一个UseAnim对象。
   public UseAnim getUseAnimation(ItemStack p_41452_) {
       //代码返回p_41452_.getItem().isEdible() ? UseAnim.EAT : UseAnim.NONE，表示物品使用的动画。如果物品是可食用的，那么这个物品的使用动画是UseAnim.EAT；否则，这个物品的使用动画是UseAnim.NONE。
      return p_41452_.getItem().isEdible() ? UseAnim.EAT : UseAnim.NONE;
   }
//定义了一个公共的方法，这个方法的名字是getUseDuration，接受一个ItemStack对象作为参数，返回一个整数。
    //返回使用的持续时间
   public int getUseDuration(ItemStack p_41454_) {
      if (p_41454_.getItem().isEdible()) {
         return p_41454_.getFoodProperties(null).isFastFood() ? 16 : 32;
      } else {
         return 0;
      }
   }
//代码定义了一个公共的方法，这个方法的名字是releaseUsing，接受四个参数：一个ItemStack对象，一个Level对象，一个LivingEntity对象和一个整数。这个方法没有返回值。
    //方法的作用是处理物品停止使用后的行为。
   public void releaseUsing(ItemStack p_41412_, Level p_41413_, LivingEntity p_41414_, int p_41415_) {
   }
//这个方法的名字是appendHoverText，接受四个参数：一个ItemStack对象，一个可能为null的Level对象，一个Component对象的列表和一个TooltipFlag对象。这个方法没有返回值。
    //这个方法的作用是添加物品的悬停文本
   public void appendHoverText(ItemStack p_41421_, @Nullable Level p_41422_, List<Component> p_41423_, TooltipFlag p_41424_) {
   }
//这行代码定义了一个公共的方法，这个方法的名字是getTooltipImage，接受一个ItemStack对象作为参数，返回一个Optional<TooltipComponent>对象。
   public Optional<TooltipComponent> getTooltipImage(ItemStack p_150902_) {
       //代码返回Optional.empty()，表示物品的工具提示图片是空的。
      return Optional.empty();
   }
//这行代码定义了一个公共的方法，这个方法的名字是getName，接受一个ItemStack对象作为参数，返回一个Component对象。
   public Component getName(ItemStack p_41458_) {
      return Component.translatable(this.getDescriptionId(p_41458_));
   }
//定义了一个公共的方法，这个方法的名字是isFoil，接受一个ItemStack对象作为参数，返回一个布尔值。
   public boolean isFoil(ItemStack p_41453_) {
       //表示物品是否有附魔效果。如果物品有附魔效果，那么这个物品就是镀金的；否则，这个物品就不是镀金的
      return p_41453_.isEnchanted();
   }
//定义了一个公共的方法，这个方法的名字是getRarity，接受一个ItemStack对象作为参数，返回一个Rarity对象。
   public Rarity getRarity(ItemStack p_41461_) {
       //首先检查物品是否有附魔效果。如果物品没有附魔效果，那么这个方法返回物品的稀有度；如果物品有附魔效果，那么这个方法根据物品的稀有度返回新的稀有度。
      if (!p_41461_.isEnchanted()) {
         return this.rarity;
      } else {
         switch(this.rarity) {
            case COMMON:
            case UNCOMMON:
               return Rarity.RARE;
            case RARE:
               return Rarity.EPIC;
            case EPIC:
            default:
               return this.rarity;
         }
      }
   }
//定义了一个公共的方法，这个方法的名字是isEnchantable，接受一个ItemStack对象作为参数，返回一个布尔值。
   public boolean isEnchantable(ItemStack p_41456_) {
       //表示物品是否可以附魔。如果物品的最大堆叠数是1且物品是可损坏的，那么这个物品就可以附魔；否则，这个物品就不可以附魔
      return this.getMaxStackSize(p_41456_) == 1 && this.isDamageable(p_41456_);
   }
//这个方法的名字是getPlayerPOVHitResult，接受三个参数：一个Level对象，一个Player对象和一个ClipContext.Fluid对象，返回一个BlockHitResult对象。
    //个方法的作用是获取玩家的视角点击结果。
   protected static BlockHitResult getPlayerPOVHitResult(Level p_41436_, Player p_41437_, ClipContext.Fluid p_41438_) {
       //这个方法获取了玩家的视角旋转角度和位置
      float f = p_41437_.getXRot();
      float f1 = p_41437_.getYRot();
      Vec3 vec3 = p_41437_.getEyePosition();
       //然后计算出了玩家的视角方向向量,这个向量用于确定玩家正在看向哪个方向
      float f2 = Mth.cos(-f1 * (float) (Math.PI / 180.0) - (float) Math.PI);
      float f3 = Mth.sin(-f1 * (float) (Math.PI / 180.0) - (float) Math.PI);
      float f4 = -Mth.cos(-f * (float) (Math.PI / 180.0));
      float f5 = Mth.sin(-f * (float) (Math.PI / 180.0));
      float f6 = f3 * f4;
      float f7 = f2 * f4;
      double d0 = p_41437_.getBlockReach();
      Vec3 vec31 = vec3.add((double)f6 * d0, (double)f5 * d0, (double)f7 * d0);
       //这个方法返回了玩家的视角点击结果。这个结果可以用于判断玩家的视角点击到的方块和位置，以及玩家是否点击到了方块。
       //这个结果是一个BlockHitResult对象，表示玩家的视角点击到的方块和位置。
      return p_41436_.clip(new ClipContext(vec3, vec31, ClipContext.Block.OUTLINE, p_41438_, p_41437_));
   }
//定义了一个公共的方法，这个方法的名字是getEnchantmentValue，没有参数，返回一个整数。这个方法被@Deprecated注解标记，表示这个方法已经被弃用，不建议再使用。
   /** Forge: Use ItemStack sensitive version. */
   @Deprecated
   public int getEnchantmentValue() {
       //这个方法的作用是获取这个物品的附魔值
      return 0;
   }
//定义了一个公共的方法，这个方法的名字是isValidRepairItem，接受两个ItemStack对象作为参数，返回一个布尔值
    //这个物品不是有效的修理物品。
   public boolean isValidRepairItem(ItemStack p_41402_, ItemStack p_41403_) {
      return false;
   }
//这个方法的名字是getDefaultAttributeModifiers，接受一个EquipmentSlot对象作为参数，返回一个Multimap<Attribute, AttributeModifier>对象。这个方法被@Deprecated注解标记，表示这个方法已经被弃用，不建议再使用。
   @Deprecated // Use ItemStack sensitive version.
   public Multimap<Attribute, AttributeModifier> getDefaultAttributeModifiers(EquipmentSlot p_41388_) {
       //代码返回一个不可变的多值映射，这个映射是空的。
      return ImmutableMultimap.of();
   }
//代码定义了一个受保护的最终的布尔变量，这个变量的名字是canRepair。
   protected final boolean canRepair;
//定义了一个公共的方法，这个方法的名字是isRepairable，接受一个ItemStack对象作为参数，返回一个布尔值。
   @Override
   public boolean isRepairable(ItemStack stack) {
       //果这个物品可以修理且是可损坏的，那么这个物品就可以修理；否则，这个物品就不可以修理。
      return canRepair && isDamageable(stack);
   }
//定义了一个公共的方法，这个方法的名字是useOnRelease，接受一个ItemStack对象作为参数，返回一个布尔值。
   public boolean useOnRelease(ItemStack p_41464_) {
       //表示这个物品是否在释放时使用。如果这个物品是一个十字弓，那么这个物品就在释放时使用；否则，这个物品就不在释放时使用。
      return p_41464_.getItem() == Items.CROSSBOW;
   }
//定义了一个公共的方法，这个方法的名字是getDefaultInstance，没有参数，返回一个ItemStack对象。
   public ItemStack getDefaultInstance() {
      return new ItemStack(this);
   }
//这行代码定义了一个公共的方法，这个方法的名字是isEdible，没有参数，返回一个布尔值。
   public boolean isEdible() {
       //表示这个物品是否可以食用。如果这个物品有食物属性，那么这个物品就可以食用；否则，这个物品就不可以食用。
      return this.foodProperties != null;
   }

    //码定义了一个公共的方法，这个方法的名字是getFoodProperties，没有参数，返回一个FoodProperties对象。这个方法被@Deprecated和@Nullable注解标记，表示这个方法已经被弃用，不建议再使用，且可能返回null。
   // Use IForgeItem#getFoodProperties(ItemStack, LivingEntity) in favour of this.
   @Deprecated
   @Nullable
   public FoodProperties getFoodProperties() {
       //行代码返回this.foodProperties，表示这个物品的食物属性。
      return this.foodProperties;
   }
//定义了一个公共的方法，这个方法的名字是getDrinkingSound，没有参数，返回一个SoundEvent对象。
   public SoundEvent getDrinkingSound() {
       //返回SoundEvents.GENERIC_DRINK，表示这个物品的饮用声音。
      return SoundEvents.GENERIC_DRINK;
   }

   public SoundEvent getEatingSound() {
      return SoundEvents.GENERIC_EAT;
   }
//定义了一个公共的方法，这个方法的名字是isFireResistant，没有参数，返回一个布尔值。
   public boolean isFireResistant() {
      return this.isFireResistant;
   }
//这个方法的名字是canBeHurtBy，接受一个DamageSource对象作为参数，返回一个布尔值
   public boolean canBeHurtBy(DamageSource p_41387_) {
       //如果这个物品不对火有抵抗力，或者伤害源不是火伤害，那么这个物品就可以被伤害；否则，这个物品就不可以被伤害
      return !this.isFireResistant || !p_41387_.is(DamageTypeTags.IS_FIRE);
   }
//表示这个物品可以放入容器中。
   public boolean canFitInsideContainerItems() {
      return true;
   }
//这个方法的名字是requiredFeatures，没有参数，返回一个FeatureFlagSet对象。
   @Override
   public FeatureFlagSet requiredFeatures() {
       //示这个物品需要的特性标志集。
      return this.requiredFeatures;
   }

   // FORGE START
    //代码定义了一个私有的对象变量，这个变量的名字是renderProperties。
   private Object renderProperties;

   /*
      DO NOT CALL, IT WILL DISAPPEAR IN THE FUTURE
      Call RenderProperties.get instead
    */
    //这行代码定义了一个公共的方法，这个方法的名字是getRenderPropertiesInternal，没有参数，返回一个Object对象。
   public Object getRenderPropertiesInternal() {
       //代码返回renderProperties，表示这个物品的渲染属性。
      return renderProperties;
   }
//代码定义了一个私有的方法，这个方法的名字是initClient，没有参数。
    //方法的作用是初始化客户端
   private void initClient() {
      // Minecraft instance isn't available in datagen, so don't call initializeClient if in datagen
      if (net.neoforged.fml.loading.FMLEnvironment.dist == net.neoforged.api.distmarker.Dist.CLIENT && !net.neoforged.fml.loading.FMLLoader.getLaunchHandler().isData()) {
         initializeClient(properties -> {
            if (properties == this)
               throw new IllegalStateException("Don't extend IItemRenderProperties in your item, use an anonymous class instead.");
            this.renderProperties = properties;
         });
      }
   }
//这个方法的名字是initializeClient，接受一个java.util.function.Consumer<net.neoforged.neoforge.client.extensions.common.IClientItemExtensions>对象作为参数。
   public void initializeClient(java.util.function.Consumer<net.neoforged.neoforge.client.extensions.common.IClientItemExtensions> consumer) {
   }
   // END FORGE

   public static class Properties {
      int maxStackSize = 64; //物品的最大堆叠数量
      int maxDamage; // 最大耐久度
      @Nullable
      Item craftingRemainingItem; // 制作时是否保留物品
      Rarity rarity = Rarity.COMMON; //物品的稀有度
      @Nullable
      FoodProperties foodProperties; //物品的食物属性
      boolean isFireResistant; //物品是否对火有抵抗力
       //物品需要的特性标志集
      FeatureFlagSet requiredFeatures = FeatureFlags.VANILLA_SET;
      private boolean canRepair = true;//是否可以修理。

      public Item.Properties food(FoodProperties p_41490_) {
          //这个方法的作用是设置物品的食物属性。
         this.foodProperties = p_41490_;
         return this;
      }
//这两行代码检查了物品的最大耐久度是否大于0，如果是，那么抛出一个异常；否则，设置物品的最大堆叠数量，并返回Item.Properties对象。
      public Item.Properties stacksTo(int p_41488_) {
         if (this.maxDamage > 0) {
            throw new RuntimeException("Unable to have damage AND stack.");
         } else {
            this.maxStackSize = p_41488_;
            return this;
         }
      }
//行代码返回this.maxDamage == 0 ? this.durability(p_41500_) : this，表示如果物品的最大耐久度为0，那么设置物品的耐久度，并返回Item.Properties对象；否则，直接返回Item.Properties对象。
      public Item.Properties defaultDurability(int p_41500_) {
         return this.maxDamage == 0 ? this.durability(p_41500_) : this;
      }
//两行代码设置了物品的最大耐久度和最大堆叠数量，并返回了Item.Properties对象。
      public Item.Properties durability(int p_41504_) {
         this.maxDamage = p_41504_;
         this.maxStackSize = 1;
         return this;
      }
//这两行代码设置了物品在制作时剩余的物品，并返回了Item.Properties对象。
      public Item.Properties craftRemainder(Item p_41496_) {
         this.craftingRemainingItem = p_41496_;
         return this;
      }
//设置稀有度
      public Item.Properties rarity(Rarity p_41498_) {
         this.rarity = p_41498_;
         return this;
      }
// 设置扛火
      public Item.Properties fireResistant() {
         this.isFireResistant = true;
         return this;
      }
// 设置不可修复
      public Item.Properties setNoRepair() {
         canRepair = false;
         return this;
      }
//这个方法的名字是requiredFeatures，接受一个FeatureFlag数组作为参数，返回Item.Properties对象。
      public Item.Properties requiredFeatures(FeatureFlag... p_250948_) {
          //设置了物品需要的特性标志集，并返回了Item.Properties对象。
         this.requiredFeatures = FeatureFlags.REGISTRY.subset(p_250948_);
         return this;
      }
   }
}

```

