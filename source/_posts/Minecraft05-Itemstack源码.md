---
title: Minecraft05-Itemstack源码
date: 2023-11-19 18:36:27
tags:
- 我的世界
- 源码
- java
cover: https://view.moezx.cc/images/2022/02/24/38586becfbdbfff6e7c637013359ec3e.png
---

# itemstack代码

```java

// 它扩展了一些能力提供者和物品堆叠扩展。
class ItemStack extends  net.neoforged.neoforge.common.capabilities.CapabilityProvider<ItemStack> implements net.neoforged.neoforge.common.extensions.IItemStackExtension {


    //这行代码声明了一个 capNBT，它是一个复合标签对象，用于存储物品的额外信息。
CompoundTag capNBT;
    //这行代码创建了一个静态的 LOGGER，用于记录日志。
 Logger LOGGER
    //这行代码创建了一个静态的空 ItemStack 对象。
ItemStack EMPTY 

    //接下来的几行声明了一些常量，例如用于标记附魔、显示名称、物品描述等的字符串常量。
 TAG_ENCH 
 TAG_DISPLAY 
 TAG_DISPLAY_NAME 
 TAG_LORE  
 TAG_DAMAGE 
 TAG_COLOR 
 TAG_UNBREAKABLE
TAG_REPAIR_COST
 TAG_CAN_DESTROY_BLOCK_LIST
 TAG_CAN_PLACE_ON_BLOCK_LIST 
 TAG_HIDE_FLAGS 
Component DISABLED_ITEM_TOOLTIP 
int DONT_HIDE_TOOLTIP
Style LORE_STYLE 
     
    //这行代码声明了一个私有的整数 count，表示物品的堆叠数量。
    int count;
    //这行代码声明了一个私有的整数 popTime，可能是与弹出（pop）时间相关的值。
    int popTime;
    //这行代码声明了一个已过时的、可空的 item，可能是表示物品。
     Item item;
    //这行代码声明了一个可空的 tag，可能是用于表示物品的标签信息
    CompoundTag tag;
    //这行代码声明了一个可空的 entityRepresentation，可能是表示物品对应的实体。
 Entity entityRepresentation;
    //这行代码声明了一个可空的 adventureBreakCheck，可能是用于冒险模式下的破坏检查
AdventureModeCheck adventureBreakCheck;
    // 这行代码声明了一个可空的 adventurePlaceCheck，可能是用于冒险模式下的放置检查。
AdventureModeCheck adventurePlaceCheck;
//这个方法返回一个可选的 TooltipComponent，可能是获取物品的提示信息图像。
 getTooltipImage() {
   }
//并设置物品堆叠数量为 1。
    ItemStack( ) {

   }
//接受一个 Holder<Item> 参数，并设置物品堆叠数量为 1。
    ItemStack() {

   }
//接受一个 ItemLike 类型的参数、一个整数参数和一个可选的 CompoundTag 参数，并执行一些初始化操作。
    ItemStack(ItemLike , int , Optional<CompoundTag> ) {

   }
//这个构造函数接受一个 Holder<Item> 类型的参数和一个整数参数，并调用另一个构造函数进行初始化。
    ItemStack(Holder<Item> , int ) {

   }
//这个构造函数接受一个 ItemLike 类型的参数和一个整数参数，并调用另一个构造函数进行初始化，并传递了一个空的 CompoundTag。
    ItemStack(ItemLike , int ) { }
    
    //这个构造函数接受一个 ItemLike 类型的参数、一个整数参数和一个可空的 CompoundTag 参数，并执行了一些初始化操作，包括设置物品的标签和数量等。
    ItemStack(ItemLike , int , @Nullable CompoundTag ) {
//此处具体实现自己去看
   }
//使用类类型和布尔值 ( true ) 调用超类构造函数（在本例中可能是 Object 类）。这将使用提供的参数初始化超类。
    ItemStack(@Nullable Void ) {//
   //  EMPTY属性的构造
   }

    ItemStack(CompoundTag ) {
//此处实现
   }
    
//尝试使用提供的 CompoundTag 创建一个新的 ItemStack 对象。
    static ItemStack of(CompoundTag ) {
   //此处实现
       
   }
    
//检itemstack是否为空。如果itemstack为空或计数为 0 或委托项为 Items.AIR ，则返回 true 。
    boolean isEmpty() {
     //判断itemstack
   }
    

//将项目堆栈分成两个物品堆栈。它创建一个具有指定计数的新物品堆栈，并将当前物品堆栈的计数减少该数量。
    ItemStack split(int ) {
shrink()//方法
   }
    
    
// 创建物品堆栈的副本，并将原始物品堆栈的计数设置为 0。如果原始物品堆栈为空，则返回空堆栈。
    ItemStack copyAndClear() {
    
   }
// 返回与堆栈关联的项目。如果堆栈为空，则返回 Items.AIR 。
    Item getItem() {
      ///
   }
    
//获取该item的holder。
    Holder<Item> getItemHolder() {
      
   }
    
//检查item是否属于指定标签。
    boolean is(TagKey<Item> p_204118_) {
      
   }
    
//检查项目是否与指定项目相同。
    boolean is(Item p_150931_) {
     
       
   }
    
//这些方法检查item是否满足特定条件或谓词。
    boolean is(Predicate<Holder<Item>> p_220168_) {
     
   }

    boolean is(Holder<Item> p_220166_) {
      
   }

    boolean is(HolderSet<Item> p_298683_) {
     
   }
    
    
//返回与该item关联的标签stream。
    Stream<TagKey<Item>> getTags() {
      
       
//处理上下文中项目的使用（就像在块上使用项目）。如果自定义函数不在客户端，它会挂钩。
    InteractionResult useOn(UseOnContext ) {
    //这几个方法都是在方块右键调用，调用对应Item的useOn，
        //注入指mixin
   }
       
//与 useOn 类似，但专门用于第一次使用。
    InteractionResult onItemUseFirst(UseOnContext p_41662_) {
     
   }
       
//此方法是一个名为 onItemUse 的私有函数。它需要一个 UseOnContext 对象  和一个将 UseOnContext 映射到 InteractionResult 的回调函数
    InteractionResult onItemUse(UseOnContext , java.util.function.Function<UseOnContext, InteractionResult> ) {
     // 判断一些内容，调用useon方法
        //这几个方法都是在方块右键调用，调用对应Item的useOn，
   }
       
       
//此方法检索特定方块状态下物品的破坏速度。它将计算委托给关联的 Item 对象的 getDestroySpeed 方法。
    float getDestroySpeed(BlockState ) {
    

       
       // 它将操作委托给关联的 Item 对象的 use 方法，该方法与提供的 Level 、 Player 和 InteractionHand 
    InteractionResultHolder<ItemStack> use(Level , Player , InteractionHand ) {
      // 右键使用方法，例如末影珍珠
   }
       
//该方法表示该item已使用完毕。它将此完成操作委托给关联的 Item 对象的 finishUsingItem 方法并返回结果 ItemStack 。
    ItemStack finishUsingItem(Level , LivingEntity ) {
      // 例如弓箭
   }
       
//它将item的 ID、计数、关联标签（如果存在）和附加功能数据保存到提供的 CompoundTag 中。
    CompoundTag save(CompoundTag ) {
     
   }
       
       
//此方法获取此特定 ItemStack 允许的最大堆栈大小。它将调用委托给关联的 Item 对象的 getMaxStackSize 方法。
    int getMaxStackSize() {
     
   }
       
// 物品是否可以继续堆叠
    boolean isStackable() {
     
   }
       
// 检查物品是否可损坏。
    boolean isDamageableItem() {
     //是否有耐久
   }
       
       
//检查物品是否损坏（掉了耐久）
    boolean isDamaged() {
   //是否已经掉了耐久
   }

    int getDamageValue() {
       //检索物品的损伤的耐久,不是最大，就是当前耐久值
    
   }
       
//设物品的耐久
    void setDamageValue(int ) {
   
   }
       
       
//检索物品可以承受的最大耐久
    int getMaxDamage() {
      //这个最大
   }
       
       
//：模拟由于使用或损坏而导致的耐久性损失。
    boolean hurt(int , RandomSource , @Nullable ServerPlayer ) {
    // 对附魔耐久的物品进行处理
       
//存在玩家并且耐久度损失值不为 0，则触发耐久度变化触发器 
        
//计算并设置新的损坏值，然后返回是否物品已经达到或超过最大损坏值，表示物品已经磨损或损坏。
       //此处实现
      }
   }

       // 减少耐久，如果为0就破坏
    <T extends LivingEntity> void hurtAndBreak(int , T , Consumer<T> ) {
      
   }
       
       
// 耐久条是否可见
    boolean isBarVisible() {
     
   }
// 耐久条宽度
    int getBarWidth() {
      
   }
// 耐久条颜色
    int getBarColor() {
     
   }
       
// 和其他的itemstack堆叠时候
    boolean overrideStackedOnOther(Slot , ClickAction , Player ) {
    //此处实现
   }
// 其他的itemstack和自己堆叠时候
    boolean overrideOtherStackedOnMe(ItemStack , Slot , ClickAction , Player , SlotAccess ) {
     
   }
// 对敌人造成伤害，如果攻击成功则玩家增加分数
    void hurtEnemy(LivingEntity , Player ) {
      //此处 
   }
        
// 用该物品挖去方块，如果成功则增加玩家分数
    void mineBlock(Level , BlockState , BlockPos , Player ) {
      //
   }
        
// 该物品是否是正确物品才能掉落
    boolean isCorrectToolForDrops(BlockState ) {

   }
        
// 用该物品和活着的实体交互
    InteractionResult interactLivingEntity(Player , LivingEntity , InteractionHand ) {
   }
        
// 复制一个itemstack
    ItemStack copy() {
     
   }
        
// 复制itemstack并设置不同的数量
   public ItemStack copyWithCount(int ) {
    
   }
        
// 两个itemstack是否相同
   public static boolean matches(ItemStack , ItemStack ) {
      
   }
        
// 是否同样物品
   public static boolean isSameItem(ItemStack , ItemStack ) {
     
   }
        
// 是否相同的tag
   public static boolean isSameItemSameTags(ItemStack , ItemStack ) {
     //
   }
        
        
// 获得物品的getDescriptionId
   public String getDescriptionId() {
   
       //tostirng()

// 背包的tick每tick回调 更新poptime 以及 条用物品的inventoryTick方法
   public void inventoryTick(Level , Entity , int , boolean ) {
       // item.tick()
   }
       
       
   public void onCraftedBy(Level , Player , int ) {
       // 调用物品的 onCraftedBy 方法，处理物品被合成的情况
   }

   public int getUseDuration() {
       //获取物品的使用持续时间
       item.userduration()
   }

   public UseAnim getUseAnimation() {
       // 获取物品的使用动画类型
       //枚举
   }

   public void releaseUsing(Level , LivingEntity p_1676_, int ) {
       //物品使用结束时触发的方法
       //应该是前后关系
   }

   public boolean useOnRelease() {
       //  检查物品是否在释放时使用
   }

   public boolean hasTag() {
       //检查物品是否有标签
   }
       //提示的部分代码我删了，想看去看这类。
       
       
 	// 返回tag
   public CompoundTag getTag() {
   }
       
       
// 返回tag如果没有就创建
   public CompoundTag getOrCreateTag() {

   }
       
       
// 返回对应string的tag，如果不存在则创建返回
   public CompoundTag getOrCreateTagElement(String ) {
     
   }
       
       
// 返回包含对应string的tag，不存在则返回null
   @Nullable
   public CompoundTag getTagElement(String ) {
     
       
   }
       
       
// 移除string的tag
   public void removeTagKey(String ) {
     
   }
       
       
// 返回附魔的tag
   public ListTag getEnchantmentTags() {
   }
       
       
// 设置tag
   public void setTag(@Nullable CompoundTag ) {
    
   }
       
       
// 获得鼠标悬浮时候名字
   public Component getHoverName() {

   }
       
       
// 设置名字
   public ItemStack setHoverName(@Nullable Component p_41715_) {
      
   }
       
       
// 重置名字
   public void resetHoverName() {
     
   }
       
       
// 自定义悬浮名字
   public boolean hasCustomHoverName() {

   }
       
       
// 根据物品的不同内容添加提示信息，例如附魔
   public List<Component> getTooltipLines(@Nullable Player , TooltipFlag ) {
 
   }
       
       
	//表示显示工具提示。
   private static boolean shouldShowInTooltip(int , ItemStack.TooltipPart ) {
 
   }
       
       

       // 物品添加附魔显示的内容
//用于向给定的 List<Component> 中追加附魔的名称。
   public static void appendEnchantmentNames(List<Component> , ListTag ) {
     
   }
       

// 判断是否具有附魔
   public boolean hasFoil() {//当前的itemstack是否具有附魔。
       
   
       
// 获得物品稀有度
   public Rarity getRarity() {
     
   }
// 物品是否可以附魔
   public boolean isEnchantable() {
     
   }
       
       
// 用于给物品附魔。
   public void enchant(Enchantment , int ) {
     
   }
       
       
// 该物品附魔过
   public boolean isEnchanted() {
     
   }
       

       //这里删除了这部分




// 获得基础修复花费经验
   public int getBaseRepairCost() {

   }
       
       
// 设置花费经验 
   public void setRepairCost(int ) {

   }
       
       
//返回item堆栈的显示名称。显示名称是将鼠标悬停在库存中的项目上时显示的名称。
   public Component getDisplayName() {
     
    
   }
//检查itemstack是否具有指定块的冒险模式地点标签。冒险模式放置标签用于确定该物品是否可以放置在冒险模式下的指定方块上。
   public boolean hasAdventureModePlaceTagForBlock(Registry<Block> p_204122_, BlockInWorld p_204123_) {
     
   }
       
//此方法检查itemstack是否具有指定块的冒险模式中断标记。冒险模式破坏标签用于确定该物品是否可以用于破坏冒险模式中的指定方块。
   public boolean hasAdventureModeBreakTagForBlock(Registry<Block> p_204129_, BlockInWorld p_204130_) {
      
   }

   public int getPopTime() {
   
   }

   public void setPopTime(int ) {
      
   }

   public int getCount() {
 
   }

   public void setCount(int ) {
       
    
// 物品count增加指定数目
   public void grow(int ) {
   }
       
// 物品count减少指定数目
   public void shrink(int ) {
   }
       
// 获得物品的使用时间
   public void onUseTick(Level , LivingEntity , int ) {
   }
       
       
// Forge 中已弃用此方法，不应使用。它调用item实体所代表的项目的 onDestroyed() 方法。
   /** @deprecated Forge: Use {@linkplain IItemStackExtension#onDestroyed(ItemEntity, net.minecraft.world.damagesource.DamageSource) damage source sensitive version} */
   @Deprecated
   public void onDestroyed(ItemEntity ) {
   }
       
       
//此方法返回该物品是否可食用。它调用项目实体所代表的项目的 isEdible() 方法。
   public boolean isEdible() {
   }
       
       //以删
       
//此方法返回消耗物品时播放的声音事件。它调用项目实体所代表的项目的 getDrinkingSound() 方法。
   public SoundEvent getDrinkingSound() {
   }
       
       
//此方法返回消耗物品时播放的声音事件。它调用项目实体所代表的项目的 getEatingSound() 方法。
   public SoundEvent getEatingSound() {
   }
       
//这行代码声明了一个名为 TooltipPart 的枚举。枚举是一种特殊的数据类型，可以用来表示一组有限的值。在这种情况下，TooltipPart 枚举用于表示物品提示中可显示的部分。
   public static enum TooltipPart {
       //这行代码定义了 TooltipPart 枚举的所有成员。每个成员都表示一个可显示在物品提示中的部分。
      ENCHANTMENTS, // 附魔
      MODIFIERS,// 属性修改
      UNBREAKABLE,// 耐久
      CAN_DESTROY,// 是否可以破坏
      CAN_PLACE,// 是否可以放置
      ADDITIONAL,// 物品的附加信息
      DYE,// 染色
      UPGRADES;// 升级信息
       
       

}

```

