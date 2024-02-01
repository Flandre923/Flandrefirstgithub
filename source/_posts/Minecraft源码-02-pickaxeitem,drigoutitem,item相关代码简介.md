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
 PickaxeItem 继承 DiggerItem 
    //它接受四个参数：一个Tier对象，一个整数，一个浮点数和一个Item.Properties对象。这个构造函数用于初始化PickaxeItem对象。
PickaxeItem()
//义了一个名为canPerformAction的公共方法，它接受两个参数：一个ItemStack对象和一个net.neoforged.neoforge.common.ToolAction对象。这个方法用于检查这个工具是否可以执行给定的动作。
canPerformAction()

```





# DiggerItem类

```java
// 类继承自TieredItem
// 实现了Vanishable接口
 DiggerItem  继承 TieredItem 实现 Vanishable {
// 存储与此工具相关的方块类型
blocks;
// 存储此工具的速度
speed;
//存储此工具的基础攻击伤害
attackDamageBaseline;
//存储此工具的默认属性修改器
defaultModifiers;
     
// 两个浮点数，一个Tier对象，一个TagKey<Block>对象和一个Item.Properties对象。这个构造函数用于初始化DiggerItem对象。
DiggerItem() {
       // 用了父类TieredItem的构造函数，传入了Tier对象和Item.Properties对象。
       // 将传入的TagKey<Block>对象赋值给blocks字段。
       // 行代码调用Tier对象的getSpeed方法，并将返回的结果赋值给speed字段。
       // 传入的浮点数与Tier对象的getAttackDamageBonus方法返回的结果相加，并将结果赋值给attackDamageBaseline字段。
       // Multimap.Builder对象，这是一个用于构建Multimap的建造者对象。Multimap`是一种特殊的映射，它允许每个键映射到多个值
       //它被用来存储属性修改器。
          //行代码向Multimap中添加属性修改器。修改攻击速度和攻击伤害方式是加法
       //码调用builder.build()方法来构建Multimap，并将结果赋值给defaultModifiers字段。
   }
     
//它接受两个参数：一个ItemStack对象和一个BlockState对象。这个方法用于获取破坏速度。
 getDestroySpeed() {
       //检查BlockState对象是否与blocks字段相匹配。如果匹配，则返回speed字段的值；否则，返回1.0F。
   }
     
//它接受三个参数：一个ItemStack对象，两个LivingEntity对象。这个方法用于攻击敌人。
hurtEnemy() {
       //行代码调用ItemStack对象的hurtAndBreak方法来损坏物品。如果物品被损坏，它会广播一个破坏事件。
   }
     
//它接受五个参数：一个ItemStack对象，一个Level对象，一个BlockState对象，一个BlockPos对象和一个LivingEntity对象。这个方法用于破坏方块。
mineBlock() {
       //检查Level对象是否为客户端，以及BlockState对象的破坏速度是否不为0.0F
 //如果物品被损坏，它会广播一个破坏事件。
   }
     
//这行代码定义了一个名为getDefaultAttributeModifiers的公共方法，它接受一个EquipmentSlot对象作为参数。这个方法用于获取默认属性修改器。
getDefaultAttributeModifiers() {
   }
     
//为getAttackDamage的公共方法，这个方法用于获取攻击伤害
     getAttackDamage() 
         
//代码定义了一个名为isCorrectToolForDrops的公共方法，它接受一个BlockState对象作为参数。这个方法用于检查这个工具是否适合破坏给定的方块。
// FORGE: Use stack sensitive variant below
isCorrectToolForDrops() {
       //行代码检查工具的等级是否已经排序。
          //代码检查工具的等级是否适合破坏给定的方块
// 检查工具挖掘等级和方块是否匹配
   }

   // FORGE START
    //这行代码定义了一个名为isCorrectToolForDrops的公共方法，它接受两个参数：一个ItemStack对象和一个BlockState对象。这个方法用于检查这个工具是否适合破坏给定的方块。
isCorrectToolForDrops()
```



# TieredItem类

```java
//代码定义了一个名为TieredItem的公共类，这个类继承自Item类。
 class TieredItem extends Item {
    //行代码定义了一个名为tier的私有常量字段，它的类型是Tier。这个字段可能用于存储此物品的挖掘等级。
  tier;
    
//这行代码定义了一个名为TieredItem的公共构造函数，它接受两个参数：一个Tier对象和一个Item.Properties对象。这个构造函数用于初始化TieredItem对象。
   public TieredItem( , ) {

   }
//定义了一个名为getTier的公共方法，这个方法用于获取物品的等级。
   public  getTier() {
   }
    
    
//这个方法用于获取物品的附魔值。
   
   public  getEnchantmentValue() {
       //并将返回的结果作为附魔值返回。
   }
    
//这个方法用于检查一个物品是否可以修复另一个物品。
   
     isValidRepairItem( ,  ) {
       //这行代码检查Tier对象的修复材料是否与传入的ItemStack对象匹配

   }
}
```



# Item 类

```java
//定义了一个名为Item的类，这个类实现了FeatureElement，ItemLike和IItemExtension接口
 class Item implements FeatureElement, ItemLike, net.neoforged.neoforge.common.extensions.IItemExtension {
    //行代码定义了一个名为LOGGER的私有静态最终字段，它的类型是Logger。这个字段用于记录日志。
   private static final Logger LOGGER = LogUtils.getLogger();
    //这个字段用于将方块映射到对应的物品。
 BY_BLOCK 
    //这个字段用于存储物品的最大堆叠大小。
 MAX_STACK_SIZE
    //这个字段用于存储吃东西的持续时间
EAT_DURATION 
    //这个字段用于存储物品耐久条的最大宽度。
 MAX_BAR_WIDTH
    //这个方法返回一个Holder.Reference<Item>对象，这个对象包含了当前的Item对象
builtInRegistryHolder 
    //这个对象代表了物品的稀有度。
 rarity;
    //这个整数代表了物品的最大堆叠数量。
maxStackSize;
    //这个整数代表了物品的最大耐久度。
 maxDamage;
    //这个布尔值表示了物品是否对火有抵抗力。
 isFireResistant;
    //这个对象代表了在制作过程中保留的物品。这个对象可能为null。
craftingRemainingItem;
    // 这个字符串代表了物品的描述ID。这个对象可能为null。
descriptionId;
    // 这个对象代表了物品的食物属性。这个对象可能为null。
 foodProperties;

    
    
    
      getId( ) {
       //返回在ITEM注册表中的ID。
   }
//它接受一个整数作为参数。这个方法返回一个Item对象。
    byId(int p_41446_) {
   }
    
//它接受一个Block对象作为参数，返回一个Item对象。这个方法被@Deprecated注解标记，表示这个方法已经被弃用，不建议再使用。
   @Deprecated
   public static Item byBlock(Block p_41440_) {
   }
    
    //它接受一个Item.Properties对象作为参数，参数的名字是p_41383_。
   public Item( ) {
   }
    
    //返回对应的注册表。已弃用
   @Deprecated
    builtInRegistryHolder() {
   }
    
    // 物品使用时候tick回调
   public void onUseTick() {
   }

    
    
    //检查一个物品是否可以攻击一个方块。
     canAttackBlock() {
   }
    
    //返回对应的item
   @Override
   public Item asItem() {
   }
    
    
     useOn( ) {
       //行代码返回InteractionResult.PASS，表示物品使用操作被跳过。
       //一些物品可能会被使用在方块上（例如放置物品），而其他物品可能不会。但在这个例子中，由于方法直接返回了InteractionResult.PASS，所以我们可以说这个物品不能被使用在任何方块上。
   }
    
    
   public float getDestroySpeed(ItemStack p_41425_, BlockState p_41426_) {
       //行代码返回1.0F，表示物品的销毁速度是1.0。
   }
    
    
   public InteractionResultHolder<ItemStack> use() {
       //代码检查itemstack是否是可食用的。
          //码检查玩家是否可以吃itemstack。
             //玩家开始使用物品，并返回一个表示物品被消耗的InteractionResultHolder对象。
          // 表示如果itemstack不是可食用的，那么返回一个表示操作被跳过的InteractionResultHolder对象，这个对象包含了玩家手中的物品。
      
       
//这个方法的名字是finishUsingItem，接受三个参数：一个ItemStack对象，一个Level对象和一个LivingEntity对象。
   public ItemStack finishUsingItem() {
       //如果物品是可食用的，那么这个方法会让生物吃这个物品；如果物品不是可食用的，那么这个方法会直接返回这个物品。
   }

    
   @Deprecated // Use ItemStack sensitive version.
      getMaxStackSize() {
       //这行代码返回this.maxStackSize，表示物品的最大堆叠数量。
   }
    
    
// Use ItemStack sensitive version.
   public final int getMaxDamage() {
       //表示物品的最大耐久度。
   }
    
   public boolean canBeDepleted() {
   //表示物品是否可以耗尽。如果物品的最大耐久度大于0，那么这个物品就可以耗尽；否则，这个物品就不能耗尽
   }
    
     isBarVisible(ItemStack ) {
       //表示物品的耐久条是否可见。如果物品已经损坏，那么这个物品的耐久条就会显示出来；否则，这个物品的耐久条就不会显示出来。
   }
    
     getBarWidth(ItemStack ) {
       //这行代码计算并返回物品的耐久条的宽度。这个宽度是通过物品的损坏值和最大耐久度计算得出的。
   }
    
    
     getBarColor( ) {

       //这个方法的作用是获取物品的耐久条的颜色。例如，一些物品的耐久条颜色会随着物品的损坏程度变化而变化，而其他物品的耐久条颜色可能是固定的。

   }
    //方法的作用是覆盖物品在其他物品上的堆叠行为
   public  overrideStackedOnOther() {
   }
    
    
    //这行代码返回false，表示默认情况下，这个物品不会覆盖其他物品堆叠在自己上的行为。
   public  overrideOtherStackedOnMe(
   ) {
   }

    //行代码返回false，表示默认情况下，这个物品不会对敌人造成伤害。
   public  hurtEnemy() {
   }
    
// 这个方法的名字是mineBlock
    //行代码返回false，表示默认情况下，这个物品不能破坏方块。
     mineBlock() {
   }
    
//代码定义了一个公共的方法，这个方法的名字是isCorrectToolForDrops，接受一个BlockState对象作为参数。这个方法返回一个布尔值。
     isCorrectToolForDrops( ) {
       //返回false，表示默认情况下，这个物品不是破坏方块后获得掉落物的正确工具。
   }

    
    interactLivingEntity() {
       //代码返回InteractionResult.PASS，表示物品与生物的交互操作被跳过。
   }
    
//返回一个getDescriptionId的Component对象。
     getDescription() {
   }
    
//这段代码定义了一个toString方法，这个方法返回物品的注册表键的字符串表示。
   @Override
   public String toString() {
   }
    
//定义了一个getOrCreateDescriptionId方法，
   protected String getOrCreateDescriptionId() {
     
   }
    
//定义了一个getDescriptionId方法，
     getDescriptionId() {
   }
    
//段代码定义了一个getDescriptionId方法
     getDescriptionId(ItemStack ) {

   }
    
//表示在多人游戏中，这个物品应该覆盖NBT（Named Binary Tag）数据。
     shouldOverrideMultiplayerNbt() {
       
   }

    
    // Use ItemStack sensitive version.
      getCraftingRemainingItem() {
       // 获得合成后保留物品
   }
    
    
// Use ItemStack sensitive version.
     hasCraftingRemainingItem() {
       // 是否具有合成保留物品
   }
    
    
     inventoryTick {
       // 背包中每tick回调
   }
    
    
     onCraftedBy() {
       //方法的作用是处理物品被玩家制作后的行为
   }
    
//判断物品是否复杂
    isComplex() {
   }

    getUseAnimation(ItemStack ) {
       //表示物品使用的动画。如果物品是可食用的，那么这个物品的使用动画是UseAnim.EAT；否则，这个物品的使用动画是UseAnim.NONE。
   }
    
    
    //返回使用的持续时间
     getUseDuration(ItemStack ) {
   }

    //方法的作用是处理物品停止使用后的行为。
   public void releaseUsing() {
   }
    
    
    //这个方法的作用是添加物品的悬停文本
   public void appendHoverText(ItemStack p_41421_, @Nullable Level p_41422_, List<Component> p_41423_, TooltipFlag p_41424_) {
   }

    
    
//返回该物品的getDescriptionId的Component对象。
     getName(ItemStack ) {
   }
    
     isFoil(ItemStack ) {
       //表示物品是否有附魔效果。如果物品有附魔效果，那么这个物品就是镀金的；否则，这个物品就不是镀金的
   }
    
     getRarity( ) {
       //首先检查物品是否有附魔效果。如果物品没有附魔效果，那么这个方法返回物品的稀有度；如果物品有附魔效果，那么这个方法根据物品的稀有度返回新的稀有度。

     isEnchantable(ItemStack ) {
       //表示物品是否可以附魔。如果物品的最大堆叠数是1且物品是可损坏的，那么这个物品就可以附魔；否则，这个物品就不可以附魔
   }

    
    //个方法的作用是获取玩家的视角点击结果。
      getPlayerPOVHitResult( ) {
       //这个方法返回了玩家的视角点击结果。这个结果可以用于判断玩家的视角点击到的方块和位置，以及玩家是否点击到了方块。
       //这个结果是一个BlockHitResult对象，表示玩家的视角点击到的方块和位置。
   }
    
   /** Forge: Use ItemStack sensitive version. */
   @Deprecated
   public int getEnchantmentValue() {
       //这个方法的作用是获取这个物品的附魔值
       
   }
//定义了一个公共的方法，这个方法的名字是isValidRepairItem，接受两个ItemStack对象作为参数，返回一个布尔值
    //这个物品不是有效的修理物品。
     isValidRepairItem(  ) {
   }

// Use ItemStack sensitive version.使用ItemStack版本
getDefaultAttributeModifiers( ) {
   }
    
    //是否可以修理
canRepair;

isRepairable( ) {
       //果这个物品可以修理且是可损坏的，那么这个物品就可以修理；否则，这个物品就不可以修理。
   }
    
 useOnRelease() {
       //表示这个物品是否在释放时使用。如果这个物品是一个十字弓，那么这个物品就在释放时使用；否则，这个物品就不在释放时使用。
   }
    
    
//定义了一个公共的方法，这个方法的名字是getDefaultInstance，没有参数，返回一个ItemStack对象。
    getDefaultInstance() {
   }

    isEdible() {
       //表示这个物品是否可以食用。如果这个物品有食物属性，那么这个物品就可以食用；否则，这个物品就不可以食用。
   }

   // Use IForgeItem#getFoodProperties(ItemStack, LivingEntity) in favour of this.
getFoodProperties() {
       //行代码返回this.foodProperties，表示这个物品的食物属性。
   }
getDrinkingSound() {
       //返回SoundEvents.GENERIC_DRINK，表示这个物品的饮用声音。
   }

//定义了一个公共的方法，这个方法的名字是isFireResistant，没有参数，返回一个布尔值。
   public boolean isFireResistant() {
   }
    
// 该物品是否可以被对应的伤害原伤害
 canBeHurtBy( ) {
   }
    
//表示这个物品可以放入容器中。
canFitInsideContainerItems()


   // FORGE START
    //代码定义了一个私有的对象变量，这个变量的名字是renderProperties。
     renderProperties;

   /*
      DO NOT CALL, IT WILL DISAPPEAR IN THE FUTURE
      Call RenderProperties.get instead
    */
    //这行代码定义了一个公共的方法，这个方法的名字是getRenderPropertiesInternal，没有参数，返回一个Object对象。
getRenderPropertiesInternal() 


   



    //Item 属性配置类
class Properties {
maxStackSize; //物品的最大堆叠数量
    
maxDamage; // 最大耐久度
    
 craftingRemainingItem; // 制作时是否保留物品
    
rarity ; //物品的稀有度
    
foodProperties; //物品的食物属性
    
 isFireResistant; //物品是否对火有抵抗力

       //物品需要的特性标志集    
requiredFeatures 
    
 canRepair;//是否可以修理。

food(FoodProperties ) {
          //这个方法的作用是设置物品的食物属性。
      }
       
//这两行代码检查了物品的最大耐久度是否大于0，如果是，那么抛出一个异常；否则，设置物品的最大堆叠数量，并返回Item.Properties对象。
stacksTo(int ) 
    
//行代码返回this.maxDamage == 0 ? this.durability(p_41500_) : this，表示如果物品的最大耐久度为0，那么设置物品的耐久度，并返回Item.Properties对象；否则，直接返回Item.Properties对象。
defaultDurability( )
    
//两行代码设置了物品的最大耐久度和最大堆叠数量，并返回了Item.Properties对象。
durability( ) 
    
//这两行代码设置了物品在制作时剩余的物品，并返回了Item.Properties对象。
      craftRemainder( ) 
    
//设置稀有度
 rarity( ) {
    
// 设置扛火
fireResistant() 
    
// 设置不可修复
 setNoRepair() 
    
//这个方法的名字是requiredFeatures，接受一个FeatureFlag数组作为参数，返回Item.Properties对象。
requiredFeatures()

```

