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
public final class ItemStack extends  net.neoforged.neoforge.common.capabilities.CapabilityProvider<ItemStack> implements net.neoforged.neoforge.common.extensions.IItemStackExtension {
    //这行代码创建了一个静态的 CODEC 实例，用于序列化和反序列化 ItemStack 对象。
   public static final Codec<ItemStack> CODEC = RecordCodecBuilder.create(
      p_258963_ -> p_258963_.group(
               BuiltInRegistries.ITEM.byNameCodec().fieldOf("id").forGetter(ItemStack::getItem),
               Codec.INT.fieldOf("Count").forGetter(ItemStack::getCount),
               CompoundTag.CODEC.optionalFieldOf("tag").forGetter(p_281115_ -> Optional.ofNullable(p_281115_.getTag()))
            )
            .apply(p_258963_, ItemStack::new)
   );
    //这行代码声明了一个私有的、可空的 delegate，它是一个对物品的引用。
   @org.jetbrains.annotations.Nullable
   private final net.minecraft.core.Holder.Reference<Item> delegate;
    //这行代码声明了一个 capNBT，它是一个复合标签对象，用于存储物品的额外信息。
   private CompoundTag capNBT;
    //这行代码创建了一个静态的 LOGGER，用于记录日志。
   private static final Logger LOGGER = LogUtils.getLogger();
    //这行代码创建了一个静态的空 ItemStack 对象。
   public static final ItemStack EMPTY = new ItemStack((Void)null);
    //这行代码创建了一个静态的格式化数字的实例，用于处理属性修改器的格式化输出。
   public static final DecimalFormat ATTRIBUTE_MODIFIER_FORMAT = Util.make(
      new DecimalFormat("#.##"), p_41704_ -> p_41704_.setDecimalFormatSymbols(DecimalFormatSymbols.getInstance(Locale.ROOT))
   );
    //接下来的几行声明了一些常量，例如用于标记附魔、显示名称、物品描述等的字符串常量。
   public static final String TAG_ENCH = "Enchantments";
   public static final String TAG_DISPLAY = "display";
   public static final String TAG_DISPLAY_NAME = "Name";
   public static final String TAG_LORE = "Lore";
   public static final String TAG_DAMAGE = "Damage";
   public static final String TAG_COLOR = "color";
   private static final String TAG_UNBREAKABLE = "Unbreakable";
   private static final String TAG_REPAIR_COST = "RepairCost";
   private static final String TAG_CAN_DESTROY_BLOCK_LIST = "CanDestroy";
   private static final String TAG_CAN_PLACE_ON_BLOCK_LIST = "CanPlaceOn";
   private static final String TAG_HIDE_FLAGS = "HideFlags";
   private static final Component DISABLED_ITEM_TOOLTIP = Component.translatable("item.disabled").withStyle(ChatFormatting.RED);
   private static final int DONT_HIDE_TOOLTIP = 0;
   private static final Style LORE_STYLE = Style.EMPTY.withColor(ChatFormatting.DARK_PURPLE).withItalic(true);
    //这行代码声明了一个私有的整数 count，表示物品的堆叠数量。
   private int count;
    //这行代码声明了一个私有的整数 popTime，可能是与弹出（pop）时间相关的值。
   private int popTime;
    //这行代码声明了一个已过时的、可空的 item，可能是表示物品。
   @Deprecated
   @Nullable
   private final Item item;
    //这行代码声明了一个可空的 tag，可能是用于表示物品的标签信息。
   @Nullable
   private CompoundTag tag;
    //这行代码声明了一个可空的 entityRepresentation，可能是表示物品对应的实体。
   @Nullable
   private Entity entityRepresentation;
    //这行代码声明了一个可空的 adventureBreakCheck，可能是用于冒险模式下的破坏检查
   @Nullable
   private AdventureModeCheck adventureBreakCheck;
    // 这行代码声明了一个可空的 adventurePlaceCheck，可能是用于冒险模式下的放置检查。
   @Nullable
   private AdventureModeCheck adventurePlaceCheck;
//这个方法返回一个可选的 TooltipComponent，可能是获取物品的提示信息图像。
   public Optional<TooltipComponent> getTooltipImage() {
      return this.getItem().getTooltipImage(this);
   }
//这个构造函数接受一个 ItemLike 参数，并设置物品堆叠数量为 1。
   public ItemStack(ItemLike p_41599_) {
      this(p_41599_, 1);
   }
//接受一个 Holder<Item> 参数，并设置物品堆叠数量为 1。
   public ItemStack(Holder<Item> p_204116_) {
      this(p_204116_.value(), 1);
   }
//接受一个 ItemLike 类型的参数、一个整数参数和一个可选的 CompoundTag 参数，并执行一些初始化操作。
   public ItemStack(ItemLike p_41604_, int p_41605_, Optional<CompoundTag> p_41606_) {
      this(p_41604_, p_41605_);
      p_41606_.ifPresent(this::setTag);
   }
//这个构造函数接受一个 Holder<Item> 类型的参数和一个整数参数，并调用另一个构造函数进行初始化。
   public ItemStack(Holder<Item> p_220155_, int p_220156_) {
      this(p_220155_.value(), p_220156_);
   }
//这个构造函数接受一个 ItemLike 类型的参数和一个整数参数，并调用另一个构造函数进行初始化，并传递了一个空的 CompoundTag。
   public ItemStack(ItemLike p_41601_, int p_41602_) { this(p_41601_, p_41602_, (CompoundTag) null); }
    //这个构造函数接受一个 ItemLike 类型的参数、一个整数参数和一个可空的 CompoundTag 参数，并执行了一些初始化操作，包括设置物品的标签和数量等。
   public ItemStack(ItemLike p_41604_, int p_41605_, @Nullable CompoundTag p_41606_) {
      super(ItemStack.class, true);
      this.capNBT = p_41606_;
      this.item = p_41604_.asItem();
      this.delegate = net.neoforged.neoforge.registries.ForgeRegistries.ITEMS.getDelegateOrThrow(p_41604_.asItem());//注册表中检索 Item 的委托（引用或标识符）并将其分配给 delegate 变量。
      this.count = p_41605_;
      this.forgeInit();//调用 forgeInit 方法，该方法可能会初始化一些 Forge 特定的功能或配置。
      if (this.item.isDamageable(this)) { // 检查物品是否可损坏（可能会损坏）。如果为 true，则会将物品堆栈的耐久值设置为其当前耐久值。
         this.setDamageValue(this.getDamageValue());
      }
   }
//使用类类型和布尔值 ( true ) 调用超类构造函数（在本例中可能是 Object 类）。这将使用提供的参数初始化超类。
   private ItemStack(@Nullable Void p_282703_) {//
      super(ItemStack.class, true);
      this.item = null;
      this.delegate = null;
   }

   private ItemStack(CompoundTag p_41608_) {
      super(ItemStack.class, true);//此行调用超类构造函数来初始化类。
      this.capNBT = p_41608_.contains("ForgeCaps") ? p_41608_.getCompound("ForgeCaps") : null;//检查 CompoundTag 是否包含特定密钥（“ForgeCaps”）。如果是，它将相应的复合标记分配给 capNBT ，否则，它将 capNBT 设置为 null 。
      Item rawItem =
      this.item = BuiltInRegistries.ITEM.get(new ResourceLocation(p_41608_.getString("id")));//使用从 CompoundTag 中的“id”键获取的字符串检索 Item 对象。它将检索到的项目分配给 rawItem 和 item 。
      this.delegate = net.neoforged.neoforge.registries.ForgeRegistries.ITEMS.getDelegateOrThrow(rawItem);//从注册表中检索该项目的委托并将其分配给 delegate 变量。
      this.count = p_41608_.getByte("Count");//从 CompoundTag 中检索项目计数并将其分配给 count 变量。
      if (p_41608_.contains("tag", 10)) {//是否包含类型 10 (CompoundTag) 的“标签”：
         this.tag = p_41608_.getCompound("tag");
         this.getItem().verifyTagAfterLoad(this.tag);
      }
      this.forgeInit();

      if (this.getItem().isDamageable(this)) {
         this.setDamageValue(this.getDamageValue());
      }
   }
//尝试使用提供的 CompoundTag 创建一个新的 ItemStack 对象。
   public static ItemStack of(CompoundTag p_41713_) {
      try {
         return new ItemStack(p_41713_);
      } catch (RuntimeException runtimeexception) {
          //如果在创建 ItemStack 期间发生异常（例如无效数据），它会捕获异常，记录一条调试消息，指示尝试加载无效项，并且返回一个 EMPTY 堆栈。
         LOGGER.debug("Tried to load invalid item: {}", p_41713_, runtimeexception);
         return EMPTY;
      }
   }
//检查项目堆栈是否为空。如果堆栈为空或计数为 0 或委托项为 Items.AIR ，则返回 true 。
   public boolean isEmpty() {
      return this == EMPTY || this.count <= 0 || this.delegate.get() == Items.AIR;
   }
//检查是否基于一组功能标志启用该项目。如果该项目为空或者该项目本身根据提供的标志启用，则返回 true
   public boolean isItemEnabled(FeatureFlagSet p_250869_) {
      return this.isEmpty() || this.getItem().isEnabled(p_250869_);
   }
//将项目堆栈分成两个物品堆栈。它创建一个具有指定计数的新物品堆栈，并将当前物品堆栈的计数减少该数量。
   public ItemStack split(int p_41621_) {
      int i = Math.min(p_41621_, this.getCount());
      ItemStack itemstack = this.copyWithCount(i);
      this.shrink(i);
      return itemstack;
   }
// 创建物品堆栈的副本，并将原始物品堆栈的计数设置为 0。如果原始物品堆栈为空，则返回空堆栈。
   public ItemStack copyAndClear() {
      if (this.isEmpty()) {
         return EMPTY;
      } else {
         ItemStack itemstack = this.copy();
         this.setCount(0);
         return itemstack;
      }
   }
// 返回与堆栈关联的项目。如果堆栈为空，则返回 Items.AIR 。
   public Item getItem() {
      return this.isEmpty() ? Items.AIR : this.delegate.get();
   }
//获取该item的holder。
   public Holder<Item> getItemHolder() {
      return this.getItem().builtInRegistryHolder();
   }
//检查item是否属于指定标签。
   public boolean is(TagKey<Item> p_204118_) {
      return this.getItem().builtInRegistryHolder().is(p_204118_);
   }
//检查项目是否与指定项目相同。
   public boolean is(Item p_150931_) {
      return this.getItem() == p_150931_;
   }
//这些方法检查item是否满足特定条件或谓词。
   public boolean is(Predicate<Holder<Item>> p_220168_) {
      return p_220168_.test(this.getItem().builtInRegistryHolder());
   }

   public boolean is(Holder<Item> p_220166_) {
      return this.getItem().builtInRegistryHolder() == p_220166_;
   }

   public boolean is(HolderSet<Item> p_298683_) {
      return p_298683_.contains(this.getItemHolder());
   }
//返回与该项目关联的标签流。
   public Stream<TagKey<Item>> getTags() {
      return this.getItem().builtInRegistryHolder().tags();
   }
//处理上下文中项目的使用（就像在块上使用项目）。如果自定义函数不在客户端，它会挂钩。
   public InteractionResult useOn(UseOnContext p_41662_) {
      if (!p_41662_.getLevel().isClientSide) return net.neoforged.neoforge.common.CommonHooks.onPlaceItemIntoWorld(p_41662_);
      return onItemUse(p_41662_, (c) -> getItem().useOn(p_41662_));
   }
//与 useOn 类似，但专门用于第一次使用。
   public InteractionResult onItemUseFirst(UseOnContext p_41662_) {
      return onItemUse(p_41662_, (c) -> getItem().onItemUseFirst(this, p_41662_));
   }
//此方法是一个名为 onItemUse 的私有函数。它需要一个 UseOnContext 对象 p_41662_ 和一个将 UseOnContext 映射到 InteractionResult 的回调函数
   private InteractionResult onItemUse(UseOnContext p_41662_, java.util.function.Function<UseOnContext, InteractionResult> callback) {
      Player player = p_41662_.getPlayer();//从 UseOnContext 中检索玩家
      BlockPos blockpos = p_41662_.getClickedPos();//检索交互发生的位置。
       //创建一个 BlockInWorld 对象，表示世界中单击位置处的块。
      BlockInWorld blockinworld = new BlockInWorld(p_41662_.getLevel(), blockpos, false);
      if (player != null//确保有玩家参与交互。
         && !player.getAbilities().mayBuild//检查玩家是否有能力建造
         && !this.hasAdventureModePlaceTagForBlock(p_41662_.getLevel().registryAccess().registryOrThrow(Registries.BLOCK), blockinworld)) {//检查该项目是否具有该块的冒险模式地点标签。
         return InteractionResult.PASS;//如果满足所有条件，则返回 InteractionResult.PASS ，表示交互未发生。
      } else {
         Item item = this.getItem();//检索与此 ItemStack 关联的 Item 。
         InteractionResult interactionresult = callback.apply(p_41662_);//提供的回调函数 ( callback.apply(p_41662_) )，该函数执行实际交互并返回 InteractionResult 。
         if (player != null && interactionresult.shouldAwardStats()) {//存在玩家且交互结果应奖励统计数据：
            player.awardStat(Stats.ITEM_USED.get(item));
         }

         return interactionresult;
      }
   }
//此方法检索特定方块状态下物品的破坏速度。它将计算委托给关联的 Item 对象的 getDestroySpeed 方法。
   public float getDestroySpeed(BlockState p_41692_) {
      return this.getItem().getDestroySpeed(this, p_41692_);
   }
// 它将操作委托给关联的 Item 对象的 use 方法，该方法与提供的 Level 、 Player 和 InteractionHand 
   public InteractionResultHolder<ItemStack> use(Level p_41683_, Player p_41684_, InteractionHand p_41685_) {
      return this.getItem().use(p_41683_, p_41684_, p_41685_);
   }
//该方法表示该item已使用完毕。它将此完成操作委托给关联的 Item 对象的 finishUsingItem 方法并返回结果 ItemStack 。
   public ItemStack finishUsingItem(Level p_41672_, LivingEntity p_41673_) {
      return this.getItem().finishUsingItem(this, p_41672_, p_41673_);
   }
//它将item的 ID、计数、关联标签（如果存在）和附加功能数据保存到提供的 CompoundTag 中。
   public CompoundTag save(CompoundTag p_41740_) {
      ResourceLocation resourcelocation = BuiltInRegistries.ITEM.getKey(this.getItem());
      p_41740_.putString("id", resourcelocation == null ? "minecraft:air" : resourcelocation.toString());
      p_41740_.putByte("Count", (byte)this.count);
      if (this.tag != null) {
         p_41740_.put("tag", this.tag.copy());
      }

      CompoundTag cnbt = this.serializeCaps();
      if (cnbt != null && !cnbt.isEmpty()) {
         p_41740_.put("ForgeCaps", cnbt);
      }
      return p_41740_;
   }
//此方法获取此特定 ItemStack 允许的最大堆栈大小。它将调用委托给关联的 Item 对象的 getMaxStackSize 方法。
   public int getMaxStackSize() {
      return this.getItem().getMaxStackSize(this);
   }
// 物品是否可以继续堆叠
   public boolean isStackable() {
      return this.getMaxStackSize() > 1 && (!this.isDamageableItem() || !this.isDamaged());
   }
// 检查物品是否可损坏。
   public boolean isDamageableItem() {
      if (!this.isEmpty() && this.getItem().isDamageable(this)) {
         CompoundTag compoundtag = this.getTag();//检索项目的标签。
          //如果标签为 null 或者“Unbreakable”标签不存在或设置为 false ，则返回 true ，否则返回 false 。
         return compoundtag == null || !compoundtag.getBoolean("Unbreakable");
      } else {
         return false;
      }
   }
//检查物品是否损坏
   public boolean isDamaged() {
       //项目是可损坏的，并且关联项目的 isDamaged 方法返回 true ，
      return this.isDamageableItem() && getItem().isDamaged(this);
   }

   public int getDamageValue() {
       //检索物品的耐久
      return this.getItem().getDamage(this);
   }
//设置物品的耐久
   public void setDamageValue(int p_41722_) {
      this.getItem().setDamage(this, p_41722_);
   }
//检索物品可以承受的最大耐久
   public int getMaxDamage() {
      return this.getItem().getMaxDamage(this);
   }
//：模拟由于使用或损坏而导致的耐久性损失。
   public boolean hurt(int p_220158_, RandomSource p_220159_, @Nullable ServerPlayer p_220160_) {
       //如果该物品不可损坏，则返回 false 。
      if (!this.isDamageableItem()) {
         return false;
      } else {
          //果该物品是可损坏的，则会根据指定的损坏值应用耐久度损失。
         if (p_220158_ > 0) {
             //触发事件并检查与物品耐用性相关的条件。
             // 获得物品附魔了不毁的附魔等级
            int i = EnchantmentHelper.getItemEnchantmentLevel(Enchantments.UNBREAKING, this);
            int j = 0;
// 对附魔“不毁”进行迭代，看是否可以减少损坏值。每次迭代中，如果附魔生效则计数器 j 会增加。
            for(int k = 0; i > 0 && k < p_220158_; ++k) {
               if (DigDurabilityEnchantment.shouldIgnoreDurabilityDrop(this, i, p_220159_)) {
                  ++j;
               }
            }

            p_220158_ -= j;
            if (p_220158_ <= 0) {
               return false;
            }
         }
//存在玩家并且耐久度损失值不为 0，则触发耐久度变化触发器 
         if (p_220160_ != null && p_220158_ != 0) {
            CriteriaTriggers.ITEM_DURABILITY_CHANGED.trigger(p_220160_, this, this.getDamageValue() + p_220158_);
         }
//计算并设置新的损坏值，然后返回是否物品已经达到或超过最大损坏值，表示物品已经磨损或损坏。
         int l = this.getDamageValue() + p_220158_;
         this.setDamageValue(l);
         return l >= this.getMaxDamage();
      }
   }

   public <T extends LivingEntity> void hurtAndBreak(int p_41623_, T p_41624_, Consumer<T> p_41625_) {
       //检查是否在客户端，以及实体是否具有创造模式
      if (!p_41624_.level().isClientSide && (!(p_41624_ instanceof Player) || !((Player)p_41624_).getAbilities().instabuild)) {
          //查物品是否可以损坏
         if (this.isDamageableItem()) {
             //使用物品的 damageItem 方法对物品进行损坏
            p_41623_ = this.getItem().damageItem(this, p_41623_, p_41624_, p_41625_);
             //根据损坏值调用 hurt 方法，模拟物品的耐久度损失
            if (this.hurt(p_41623_, p_41624_.getRandom(), p_41624_ instanceof ServerPlayer ? (ServerPlayer)p_41624_ : null)) {
                //如果物品已经损坏
               p_41625_.accept(p_41624_);//接受一个实体，并对其执行某些操作
               Item item = this.getItem();
               this.shrink(1);// // 减少物品栈中物品的数量
               if (p_41624_ instanceof Player) {
                  ((Player)p_41624_).awardStat(Stats.ITEM_BROKEN.get(item));//玩家增加分数
               }

               this.setDamageValue(0);//// 将物品的损坏值设为 0，表示物品已经损坏
            }
         }
      }
   }
// 耐久条是否可见
   public boolean isBarVisible() {
      return this.getItem().isBarVisible(this);
   }
// 耐久条宽度
   public int getBarWidth() {
      return this.getItem().getBarWidth(this);
   }
// 耐久条颜色
   public int getBarColor() {
      return this.getItem().getBarColor(this);
   }
// 和其他的itemstack堆叠时候
   public boolean overrideStackedOnOther(Slot p_150927_, ClickAction p_150928_, Player p_150929_) {
      return this.getItem().overrideStackedOnOther(this, p_150927_, p_150928_, p_150929_);
   }
// 其他的itemstack和自己堆叠时候
   public boolean overrideOtherStackedOnMe(ItemStack p_150933_, Slot p_150934_, ClickAction p_150935_, Player p_150936_, SlotAccess p_150937_) {
      return this.getItem().overrideOtherStackedOnMe(this, p_150933_, p_150934_, p_150935_, p_150936_, p_150937_);
   }
// 对敌人造成伤害，如果攻击成功则玩家增加分数
   public void hurtEnemy(LivingEntity p_41641_, Player p_41642_) {
      Item item = this.getItem();
      if (item.hurtEnemy(this, p_41641_, p_41642_)) {
         p_41642_.awardStat(Stats.ITEM_USED.get(item));
      }
   }
// 用该物品挖去方块，如果成功则增加玩家分数
   public void mineBlock(Level p_41687_, BlockState p_41688_, BlockPos p_41689_, Player p_41690_) {
      Item item = this.getItem();
      if (item.mineBlock(this, p_41687_, p_41688_, p_41689_, p_41690_)) {
         p_41690_.awardStat(Stats.ITEM_USED.get(item));
      }
   }
// 该物品是否是正确物品才能掉落
   public boolean isCorrectToolForDrops(BlockState p_41736_) {
      return this.getItem().isCorrectToolForDrops(this, p_41736_);
   }
// 用该物品和活着的实体交互
   public InteractionResult interactLivingEntity(Player p_41648_, LivingEntity p_41649_, InteractionHand p_41650_) {
      return this.getItem().interactLivingEntity(this, p_41648_, p_41649_, p_41650_);
   }
// 复制一个itemstack
   public ItemStack copy() {
      if (this.isEmpty()) {
         return EMPTY;
      } else {
         ItemStack itemstack = new ItemStack(this.getItem(), this.count, this.serializeCaps());
         itemstack.setPopTime(this.getPopTime());
         if (this.tag != null) {
            itemstack.tag = this.tag.copy();
         }

         return itemstack;
      }
   }
// 复制itemstack并设置不同的数量
   public ItemStack copyWithCount(int p_256354_) {
      if (this.isEmpty()) {
         return EMPTY;
      } else {
         ItemStack itemstack = this.copy();
         itemstack.setCount(p_256354_);
         return itemstack;
      }
   }
// 两个itemstack是否相同
   public static boolean matches(ItemStack p_41729_, ItemStack p_41730_) {
      if (p_41729_ == p_41730_) {
         return true;
      } else {
         return p_41729_.getCount() != p_41730_.getCount() ? false : isSameItemSameTags(p_41729_, p_41730_);
      }
   }
// 是否同样物品
   public static boolean isSameItem(ItemStack p_287761_, ItemStack p_287676_) {
      return p_287761_.is(p_287676_.getItem());
   }
// 是否相同的tag
   public static boolean isSameItemSameTags(ItemStack p_150943_, ItemStack p_150944_) {
      if (!p_150943_.is(p_150944_.getItem())) {
         return false;
      } else {
         return p_150943_.isEmpty() && p_150944_.isEmpty() ? true : Objects.equals(p_150943_.tag, p_150944_.tag) && p_150943_.areCapsCompatible(p_150944_);
      }
   }
// 获得物品的getDescriptionId
   public String getDescriptionId() {
      return this.getItem().getDescriptionId(this);
   }
// tostring
   @Override
   public String toString() {
      return this.getCount() + " " + this.getItem();
   }
// 背包的tick每tick回调
   public void inventoryTick(Level p_41667_, Entity p_41668_, int p_41669_, boolean p_41670_) {
       // poptime 减一/tick
      if (this.popTime > 0) {
         --this.popTime;
      }
	// 如果不为空则调用物品的tick
      if (this.getItem() != null) {
         this.getItem().inventoryTick(this, p_41667_, p_41668_, p_41669_, p_41670_);
      }
   }
//
   public void onCraftedBy(Level p_41679_, Player p_41680_, int p_41681_) {
       // 授予玩家相关的统计数据，例如物品的合成
      p_41680_.awardStat(Stats.ITEM_CRAFTED.get(this.getItem()), p_41681_);
       // 调用物品的 onCraftedBy 方法，处理物品被合成的情况
      this.getItem().onCraftedBy(this, p_41679_, p_41680_);
   }

   public int getUseDuration() {
       //获取物品的使用持续时间
      return this.getItem().getUseDuration(this);
   }

   public UseAnim getUseAnimation() {
       // 获取物品的使用动画类型
      return this.getItem().getUseAnimation(this);
   }

   public void releaseUsing(Level p_41675_, LivingEntity p_41676_, int p_41677_) {
       //物品使用结束时触发的方法
      this.getItem().releaseUsing(this, p_41675_, p_41676_, p_41677_);
   }

   public boolean useOnRelease() {
       //  检查物品是否在释放时使用
      return this.getItem().useOnRelease(this);
   }

   public boolean hasTag() {
       //检查物品是否有标签
      return !this.isEmpty() && this.tag != null && !this.tag.isEmpty();
   }
 	// 返回tag
   @Nullable
   public CompoundTag getTag() {
      return this.tag;
   }
// 返回tag如果没有就创建
   public CompoundTag getOrCreateTag() {
      if (this.tag == null) {
         this.setTag(new CompoundTag());
      }

      return this.tag;
   }
// 返回对应string的tag，如果不存在则创建返回
   public CompoundTag getOrCreateTagElement(String p_41699_) {
      if (this.tag != null && this.tag.contains(p_41699_, 10)) {
         return this.tag.getCompound(p_41699_);
      } else {
         CompoundTag compoundtag = new CompoundTag();
         this.addTagElement(p_41699_, compoundtag);
         return compoundtag;
      }
   }
// 返回包含对应string的tag，不存在则返回null
   @Nullable
   public CompoundTag getTagElement(String p_41738_) {
      return this.tag != null && this.tag.contains(p_41738_, 10) ? this.tag.getCompound(p_41738_) : null;
   }
// 移除string的tag
   public void removeTagKey(String p_41750_) {
      if (this.tag != null && this.tag.contains(p_41750_)) {
         this.tag.remove(p_41750_);
         if (this.tag.isEmpty()) {
            this.tag = null;
         }
      }
   }
// 返回附魔的tag
   public ListTag getEnchantmentTags() {
      return this.tag != null ? this.tag.getList("Enchantments", 10) : new ListTag();
   }
// 设置tag
   public void setTag(@Nullable CompoundTag p_41752_) {
      this.tag = p_41752_;
      if (this.getItem().isDamageable(this)) {
         this.setDamageValue(this.getDamageValue());
      }

      if (p_41752_ != null) {
         this.getItem().verifyTagAfterLoad(p_41752_);
      }
   }
// 获得鼠标悬浮时候名字
   public Component getHoverName() {
      CompoundTag compoundtag = this.getTagElement("display");
      if (compoundtag != null && compoundtag.contains("Name", 8)) {
         try {
            Component component = Component.Serializer.fromJson(compoundtag.getString("Name"));
            if (component != null) {
               return component;
            }

            compoundtag.remove("Name");
         } catch (Exception exception) {
            compoundtag.remove("Name");
         }
      }

      return this.getItem().getName(this);
   }
// 设置名字
   public ItemStack setHoverName(@Nullable Component p_41715_) {
      CompoundTag compoundtag = this.getOrCreateTagElement("display");
      if (p_41715_ != null) {
         compoundtag.putString("Name", Component.Serializer.toJson(p_41715_));
      } else {
         compoundtag.remove("Name");
      }

      return this;
   }
// 重置名字
   public void resetHoverName() {
      CompoundTag compoundtag = this.getTagElement("display");
      if (compoundtag != null) {
         compoundtag.remove("Name");
         if (compoundtag.isEmpty()) {
            this.removeTagKey("display");
         }
      }

      if (this.tag != null && this.tag.isEmpty()) {
         this.tag = null;
      }
   }
// 自定义悬浮名字
   public boolean hasCustomHoverName() {
      CompoundTag compoundtag = this.getTagElement("display");
      return compoundtag != null && compoundtag.contains("Name", 8);
   }
//
   public List<Component> getTooltipLines(@Nullable Player p_41652_, TooltipFlag p_41653_) {
           // 创建一个空的组件列表
      List<Component> list = Lists.newArrayList();
       // 获取物品名称并添加到提示信息列表中，根据物品的稀有度修改样式
      MutableComponent mutablecomponent = Component.empty().append(this.getHoverName()).withStyle(this.getRarity().getStyleModifier()); // 如果物品有自定义名称，样式设为斜体
      if (this.hasCustomHoverName()) {
         mutablecomponent.withStyle(ChatFormatting.ITALIC);
      }

      list.add(mutablecomponent);// 将物品名称添加到列表中
       
    // 若非高级提示信息并且物品没有自定义名称且是填充地图类型的物品
      if (!p_41653_.isAdvanced() && !this.hasCustomHoverName() && this.is(Items.FILLED_MAP)) {
         Integer integer = MapItem.getMapId(this);
         if (integer != null) {
            list.add(MapItem.getTooltipForId(this));// 添加提示信息
         }
      }

      int j = this.getHideFlags(); // 获取隐藏标志位
       // 判断是否应该在提示信息中显示额外信息，并根据情况调用物品的 appendHoverText 方法
      if (shouldShowInTooltip(j, ItemStack.TooltipPart.ADDITIONAL)) {
         this.getItem().appendHoverText(this, p_41652_ == null ? null : p_41652_.level(), list, p_41653_);
      }
//判断物品是否有标签信息
      if (this.hasTag()) {
          // 若应该在提示信息中显示升级相关内容，并且玩家不为空，则添加装备升级的提示信息
         if (shouldShowInTooltip(j, ItemStack.TooltipPart.UPGRADES) && p_41652_ != null) {
            ArmorTrim.appendUpgradeHoverText(this, p_41652_.level().registryAccess(), list);
         }
//若应该在提示信息中显示附魔，并且物品有附魔标签，则添加附魔名称信息
         if (shouldShowInTooltip(j, ItemStack.TooltipPart.ENCHANTMENTS)) {
            appendEnchantmentNames(list, this.getEnchantmentTags());
         }
//若标签中包含"display"，则处理颜色和描述信息
         if (this.tag.contains("display", 10)) {
            CompoundTag compoundtag = this.tag.getCompound("display");
            if (shouldShowInTooltip(j, ItemStack.TooltipPart.DYE) && compoundtag.contains("color", 99)) {
               if (p_41653_.isAdvanced()) {
                  list.add(
                     Component.translatable("item.color", String.format(Locale.ROOT, "#%06X", compoundtag.getInt("color"))).withStyle(ChatFormatting.GRAY)
                  );
               } else {
                  list.add(Component.translatable("item.dyed").withStyle(ChatFormatting.GRAY, ChatFormatting.ITALIC));
               }
            }
//处理物品的描述信息（Lore）
            if (compoundtag.getTagType("Lore") == 9) {
               ListTag listtag = compoundtag.getList("Lore", 8);

               for(int i = 0; i < listtag.size(); ++i) {
                  String s = listtag.getString(i);

                  try {
                     MutableComponent mutablecomponent1 = Component.Serializer.fromJson(s);
                     if (mutablecomponent1 != null) {
                        list.add(ComponentUtils.mergeStyles(mutablecomponent1, LORE_STYLE));
                     }
                  } catch (Exception exception) {
                     compoundtag.remove("Lore");
                  }
               }
            }
         }
      }
//如果应该在提示信息中显示属性修饰器，依次处理不同装备槽的属性修饰器
      if (shouldShowInTooltip(j, ItemStack.TooltipPart.MODIFIERS)) {
         for(EquipmentSlot equipmentslot : EquipmentSlot.values()) {
            Multimap<Attribute, AttributeModifier> multimap = this.getAttributeModifiers(equipmentslot);
            if (!multimap.isEmpty()) {
               list.add(CommonComponents.EMPTY);
               list.add(Component.translatable("item.modifiers." + equipmentslot.getName()).withStyle(ChatFormatting.GRAY));

               for(Entry<Attribute, AttributeModifier> entry : multimap.entries()) {
                  AttributeModifier attributemodifier = entry.getValue();
                  double d0 = attributemodifier.getAmount();
                  boolean flag = false;
                  if (p_41652_ != null) {
                     if (attributemodifier.getId() == Item.BASE_ATTACK_DAMAGE_UUID) {
                        d0 += p_41652_.getAttributeBaseValue(Attributes.ATTACK_DAMAGE);
                        d0 += (double)EnchantmentHelper.getDamageBonus(this, MobType.UNDEFINED);
                        flag = true;
                     } else if (attributemodifier.getId() == Item.BASE_ATTACK_SPEED_UUID) {
                        d0 += p_41652_.getAttributeBaseValue(Attributes.ATTACK_SPEED);
                        flag = true;
                     }
                  }

                  double d1;
                  if (attributemodifier.getOperation() == AttributeModifier.Operation.MULTIPLY_BASE
                     || attributemodifier.getOperation() == AttributeModifier.Operation.MULTIPLY_TOTAL) {
                     d1 = d0 * 100.0;
                  } else if (entry.getKey().equals(Attributes.KNOCKBACK_RESISTANCE)) {
                     d1 = d0 * 10.0;
                  } else {
                     d1 = d0;
                  }

                  if (flag) {
                     list.add(
                        CommonComponents.space()
                           .append(
                              Component.translatable(
                                 "attribute.modifier.equals." + attributemodifier.getOperation().toValue(),
                                 ATTRIBUTE_MODIFIER_FORMAT.format(d1),
                                 Component.translatable(entry.getKey().getDescriptionId())
                              )
                           )
                           .withStyle(ChatFormatting.DARK_GREEN)
                     );
                  } else if (d0 > 0.0) {
                     list.add(
                        Component.translatable(
                              "attribute.modifier.plus." + attributemodifier.getOperation().toValue(),
                              ATTRIBUTE_MODIFIER_FORMAT.format(d1),
                              Component.translatable(entry.getKey().getDescriptionId())
                           )
                           .withStyle(ChatFormatting.BLUE)
                     );
                  } else if (d0 < 0.0) {
                     d1 *= -1.0;
                     list.add(
                        Component.translatable(
                              "attribute.modifier.take." + attributemodifier.getOperation().toValue(),
                              ATTRIBUTE_MODIFIER_FORMAT.format(d1),
                              Component.translatable(entry.getKey().getDescriptionId())
                           )
                           .withStyle(ChatFormatting.RED)
                     );
                  }
               }
            }
         }
      }

      if (this.hasTag()) {
         if (shouldShowInTooltip(j, ItemStack.TooltipPart.UNBREAKABLE) && this.tag.getBoolean("Unbreakable")) {
            list.add(Component.translatable("item.unbreakable").withStyle(ChatFormatting.BLUE));
         }

         if (shouldShowInTooltip(j, ItemStack.TooltipPart.CAN_DESTROY) && this.tag.contains("CanDestroy", 9)) {
            ListTag listtag1 = this.tag.getList("CanDestroy", 8);
            if (!listtag1.isEmpty()) {
               list.add(CommonComponents.EMPTY);
               list.add(Component.translatable("item.canBreak").withStyle(ChatFormatting.GRAY));

               for(int k = 0; k < listtag1.size(); ++k) {
                  list.addAll(expandBlockState(listtag1.getString(k)));
               }
            }
         }

         if (shouldShowInTooltip(j, ItemStack.TooltipPart.CAN_PLACE) && this.tag.contains("CanPlaceOn", 9)) {
            ListTag listtag2 = this.tag.getList("CanPlaceOn", 8);
            if (!listtag2.isEmpty()) {
               list.add(CommonComponents.EMPTY);
               list.add(Component.translatable("item.canPlace").withStyle(ChatFormatting.GRAY));

               for(int l = 0; l < listtag2.size(); ++l) {
                  list.addAll(expandBlockState(listtag2.getString(l)));
               }
            }
         }
      }

      if (p_41653_.isAdvanced()) {
         if (this.isDamaged()) {
            list.add(Component.translatable("item.durability", this.getMaxDamage() - this.getDamageValue(), this.getMaxDamage()));
         }

         list.add(Component.literal(BuiltInRegistries.ITEM.getKey(this.getItem()).toString()).withStyle(ChatFormatting.DARK_GRAY));
         if (this.hasTag()) {
            list.add(Component.translatable("item.nbt_tags", this.tag.getAllKeys().size()).withStyle(ChatFormatting.DARK_GRAY));
         }
      }

      if (p_41652_ != null && !this.getItem().isEnabled(p_41652_.level().enabledFeatures())) {
         list.add(DISABLED_ITEM_TOOLTIP);
      }

      net.neoforged.neoforge.event.EventHooks.onItemTooltip(this, p_41652_, list, p_41653_);
      return list;/// 返回生成的物品提示信息列表
   }
	//表示显示工具提示。
   private static boolean shouldShowInTooltip(int p_41627_, ItemStack.TooltipPart p_41628_) {
      return (p_41627_ & p_41628_.getMask()) == 0;
   }
// 检查当前物品是否有标签（tag），并且该标签包含名为 "HideFlags" 的键，类型为 99（表示标签类型为 CompoundTag）。
   private int getHideFlags() {
      return this.hasTag() && this.tag.contains("HideFlags", 99) ? this.tag.getInt("HideFlags") : this.getItem().getDefaultTooltipHideFlags(this);
   }
// 用于隐藏提示部分
   public void hideTooltipPart(ItemStack.TooltipPart p_41655_) {
      CompoundTag compoundtag = this.getOrCreateTag();
       //获取或创建当前物品的标签（tag），然后将 "HideFlags" 键对应的整数值与 p_41655_ 的掩码进行按位或运算，然后将结果设置为新的 "HideFlags" 值。
      compoundtag.putInt("HideFlags", compoundtag.getInt("HideFlags") | p_41655_.getMask());
   }
//用于向给定的 List<Component> 中追加附魔的名称。
   public static void appendEnchantmentNames(List<Component> p_41710_, ListTag p_41711_) {
      for(int i = 0; i < p_41711_.size(); ++i) {
         CompoundTag compoundtag = p_41711_.getCompound(i);
         BuiltInRegistries.ENCHANTMENT
            .getOptional(EnchantmentHelper.getEnchantmentId(compoundtag))
            .ifPresent(p_41708_ -> p_41710_.add(p_41708_.getFullname(EnchantmentHelper.getEnchantmentLevel(compoundtag))));
      }
   }

   private static Collection<Component> expandBlockState(String p_41762_) {
      try {
          //它尝试使用 BlockStateParser 解析给定的字符串 p_41762_ 以展开方块状态。BlockStateParser 似乎是用于解析方块状态的工具类。它使用 BuiltInRegistries.BLOCK.asLookup() 来获取方块注册表，并将 p_41762_ 解析为方块状态。最后一个参数 true 似乎表示进行测试解析。
         return BlockStateParser.parseForTesting(BuiltInRegistries.BLOCK.asLookup(), p_41762_, true)
             //代码使用了 map 方法。它在解析后的结果上进行映射处理。第一个参数是针对正常解析到的方块状态，将其转换为一个列表，并为方块设置一个暗灰色的样式。第二个参数似乎处理那些在解析过程中出现错误的情况，它使用 tag() 获取标签，并将标签值转换为一个列表，同样使用暗灰色的样式。
            .map(
               p_220162_ -> Lists.newArrayList(p_220162_.blockState().getBlock().getName().withStyle(ChatFormatting.DARK_GRAY)),
               p_220164_ -> p_220164_.tag()
                     .stream()
                     .map(p_220172_ -> p_220172_.value().getName().withStyle(ChatFormatting.DARK_GRAY))
                     .collect(Collectors.toList())
            );
      } catch (CommandSyntaxException commandsyntaxexception) {
          //解析过程中出现 CommandSyntaxException 异常，即解析失败，则会捕获异常并返回一个列表，其中包含一个显示为 "missingno" 的文本组件，并应用暗灰色的样式。
         return Lists.newArrayList(Component.literal("missingno").withStyle(ChatFormatting.DARK_GRAY));
      }
   }
// 判断是否foil
   public boolean hasFoil() {
      return this.getItem().isFoil(this);
   }
// 获得物品稀有度
   public Rarity getRarity() {
      return this.getItem().getRarity(this);
   }
// 物品是否可以附魔
   public boolean isEnchantable() {
      if (!this.getItem().isEnchantable(this)) {
         return false;
      } else {
         return !this.isEnchanted();
      }
   }
// 用于给物品附魔。
   public void enchant(Enchantment p_41664_, int p_41665_) {
       // 取或创建物品标签的方法。
      this.getOrCreateTag();
      if (!this.tag.contains("Enchantments", 9)) {
         this.tag.put("Enchantments", new ListTag());
      }

      ListTag listtag = this.tag.getList("Enchantments", 10);
       //它使用了 EnchantmentHelper.storeEnchantment() 方法，将附魔的ID和等级存储为一个 CompoundTag，并将其添加到 "Enchantments" 列表中。
      listtag.add(EnchantmentHelper.storeEnchantment(EnchantmentHelper.getEnchantmentId(p_41664_), (byte)p_41665_));
   }
// 该物品是否附魔了
   public boolean isEnchanted() {
      if (this.tag != null && this.tag.contains("Enchantments", 9)) {
         return !this.tag.getList("Enchantments", 10).isEmpty();
      } else {
         return false;
      }
   }
// 添加tag
   public void addTagElement(String p_41701_, Tag p_41702_) {
      this.getOrCreateTag().put(p_41701_, p_41702_);
   }
// 是否framed
   public boolean isFramed() {
      return this.entityRepresentation instanceof ItemFrame;
   }
// 设置EntityRepresentation
   public void setEntityRepresentation(@Nullable Entity p_41637_) {
      this.entityRepresentation = p_41637_;
   }
// 获得Frame
   @Nullable
   public ItemFrame getFrame() {
      return this.entityRepresentation instanceof ItemFrame ? (ItemFrame)this.getEntityRepresentation() : null;
   }
// 获得EntityRepresentation
   @Nullable
   public Entity getEntityRepresentation() {
      return !this.isEmpty() ? this.entityRepresentation : null;
   }
// 获得基础修复花费经验
   public int getBaseRepairCost() {
      return this.hasTag() && this.tag.contains("RepairCost", 3) ? this.tag.getInt("RepairCost") : 0;
   }
// 设置花费经验 
   public void setRepairCost(int p_41743_) {
      if (p_41743_ > 0) {
         this.getOrCreateTag().putInt("RepairCost", p_41743_);
      } else {
         this.removeTagKey("RepairCost");
      }
   }
// 名为 getAttributeModifiers 的方法，该方法采用 EquipmentSlot 作为参数，并返回 Attribute 和 AttributeModifier 将包含适用于指定 EquipmentSlot 的所有属性修饰符。
   public Multimap<Attribute, AttributeModifier> getAttributeModifiers(EquipmentSlot p_41639_) {
       //声明一个名为 multimap 的类型为 Multimap 的变量，它是一个可以将多个值映射到单个键的集合。在这种情况下， Multimap 将用于将 Attribute 对象映射到 AttributeModifier 对象。
      Multimap<Attribute, AttributeModifier> multimap;
       //当前item是否具有名为 AttributeModifiers 的 NBT 标记。如果是，代码将继续执行下一个代码块。
      if (this.hasTag() && this.tag.contains("AttributeModifiers", 9)) {
          //新的 HashMultimap 对象并将其分配给 multimap 变量。 HashMultimap 是 Multimap 的一种类型，它使用哈希表来存储其数据。
         multimap = HashMultimap.create();
          //此行从当前项目的 NBT 标签复合中检索名为 AttributeModifiers 的 NBT 标签列表。 NBT标签列表是NBT标签组合的集合，每个组合代表一个属性修饰符。
         ListTag listtag = this.tag.getList("AttributeModifiers", 10);
//启动一个循环，迭代 NBT 标签列表。循环将继续，直到到达列表末尾。
         for(int i = 0; i < listtag.size(); ++i) {
             //检索 NBT 标签列表中当前索引处的 NBT 标签组合。 NBT 标签复合包含单个属性修饰符的所有信息。
            CompoundTag compoundtag = listtag.getCompound(i);
             //检查 NBT 标记复合是否具有名为 Slot 的标记，如果有，则检查 Slot 标记的值是否等于指定的 EquipmentSlot 的名称b2> .如果不满足这两个条件中的任何一个，代码将跳到循环的下一次迭代。
            if (!compoundtag.contains("Slot", 8) || compoundtag.getString("Slot").equals(p_41639_.getName())) {
                //使用 NBT 标记复合中的 AttributeName 标记的值从 BuiltInRegistries 类检索 Attribute 对象。如果找到 Attribute 对象，则将其存储在 Optional 对象中。
               Optional<Attribute> optional = BuiltInRegistries.ATTRIBUTE.getOptional(ResourceLocation.tryParse(compoundtag.getString("AttributeName")));
                //检查 Optional 对象是否为空。如果不为空，则代码将继续执行下一个代码块。
               if (!optional.isEmpty()) {
                   //使用 AttributeModifier.load() 方法从 NBT 标记复合创建一个 AttributeModifier 对象。
                  AttributeModifier attributemodifier = AttributeModifier.load(compoundtag);
                   //检查 attributemodifier 变量是否不为空。如果不为空，则代码将继续执行下一个代码块。检查 attributemodifier 对象 ID 的最低有效位是否不等于 0。检查 attributemodifier 对象 ID 的最高有效位是否不等于 0。
                  if (attributemodifier != null
                     && attributemodifier.getId().getLeastSignificantBits() != 0L
                     && attributemodifier.getId().getMostSignificantBits() != 0L) {
                      // 将 attributemodifier 对象添加到 multimap 
                     multimap.put(optional.get(), attributemodifier);
                  }
               }
            }
         }
      } else {
          // 查如果当前物品没有 AttributeModifiers NBT 标签，则将调用当前物品的 getItem() 方法来获取物品实例。调用物品实例的 getAttributeModifiers() 方法来获取适用于指定 EquipmentSlot 的所有属性修饰器。
         multimap = this.getItem().getAttributeModifiers(p_41639_, this);
      }
//调用 net.neoforged.neoforge.common.CommonHooks 类的 getAttributeModifiers() 方法。该方法可以用于修改或添加属性修饰器。
      multimap = net.neoforged.neoforge.common.CommonHooks.getAttributeModifiers(this, p_41639_, multimap);
      return multimap;
   }
//此方法向item元添加属性修饰符。 p_41644_ 参数是要修改的属性， p_41645_ 参数是属性修饰符， p_41646_ 参数是属性修饰符应应用到的设备槽位。
   public void addAttributeModifier(Attribute p_41644_, AttributeModifier p_41645_, @Nullable EquipmentSlot p_41646_) {
       //取或创建项目元的 NBT 标签。 NBT 标签用于存储有关该物品的附加数据，例如其属性和附魔。
      this.getOrCreateTag();
       //检查 NBT 标记是否包含属性修饰符列表。如果没有，那么它会创建一个新的属性修饰符列表。
      if (!this.tag.contains("AttributeModifiers", 9)) {
         this.tag.put("AttributeModifiers", new ListTag());
      }
//此行从 NBT 标记获取属性修饰符列表。
      ListTag listtag = this.tag.getList("AttributeModifiers", 10);
       //此行将属性修饰符转换为 NBT 复合标记。
      CompoundTag compoundtag = p_41645_.save();
       //此行将属性名称添加到 NBT 复合标记中。
      compoundtag.putString("AttributeName", BuiltInRegistries.ATTRIBUTE.getKey(p_41644_).toString());
       //如果设备槽不为空，则此行将设备槽添加到 NBT 复合标记中。
      if (p_41646_ != null) {
         compoundtag.putString("Slot", p_41646_.getName());
      }
//此行将 NBT 复合标签添加到属性修饰符列表中。
      listtag.add(compoundtag);
   }
//返回item堆栈的显示名称。显示名称是将鼠标悬停在库存中的项目上时显示的名称。
   public Component getDisplayName() {
       //创建一个新的可变组件并将项目的悬停名称附加到它。可变组件用于构建项目显示名称的文本组件树。
      MutableComponent mutablecomponent = Component.empty().append(this.getHoverName());
       //检查该项目是否具有自定义悬停名称。如果是，则它将可变组件的样式设置为斜体.
      if (this.hasCustomHoverName()) {
         mutablecomponent.withStyle(ChatFormatting.ITALIC);
      }
//将可变组件括在方括号中。这是 Minecraft 中项目显示名称的默认格式。
      MutableComponent mutablecomponent1 = ComponentUtils.wrapInSquareBrackets(mutablecomponent);
       //该行检查项目堆栈是否不为空。如果它不为空，则它将可变组件的样式设置为该项目的稀有度。它还向可变组件添加了一个悬停事件，当光标悬停在可变组件上时，该事件会显示项目堆栈。
      if (!this.isEmpty()) {
         mutablecomponent1.withStyle(this.getRarity().getStyleModifier()).withStyle((p_220170_) -> {
            return p_220170_.withHoverEvent(new HoverEvent(HoverEvent.Action.SHOW_ITEM, new HoverEvent.ItemStackInfo(this)));
         });
      }

      return mutablecomponent1;
   }
//检查itemstack是否具有指定块的冒险模式地点标签。冒险模式放置标签用于确定该物品是否可以放置在冒险模式下的指定方块上。
   public boolean hasAdventureModePlaceTagForBlock(Registry<Block> p_204122_, BlockInWorld p_204123_) {
      if (this.adventurePlaceCheck == null) {
         this.adventurePlaceCheck = new AdventureModeCheck("CanPlaceOn");
      }

      return this.adventurePlaceCheck.test(this, p_204122_, p_204123_);
   }
//此方法检查itemstack是否具有指定块的冒险模式中断标记。冒险模式破坏标签用于确定该物品是否可以用于破坏冒险模式中的指定方块。

   public boolean hasAdventureModeBreakTagForBlock(Registry<Block> p_204129_, BlockInWorld p_204130_) {
      if (this.adventureBreakCheck == null) {
         this.adventureBreakCheck = new AdventureModeCheck("CanDestroy");
      }

      return this.adventureBreakCheck.test(this, p_204129_, p_204130_);
   }

   public int getPopTime() {
      return this.popTime;
   }

   public void setPopTime(int p_41755_) {
      this.popTime = p_41755_;
   }

   public int getCount() {
      return this.isEmpty() ? 0 : this.count;
   }

   public void setCount(int p_41765_) {
      this.count = p_41765_;
   }
// 物品count增加指定数目
   public void grow(int p_41770_) {
      this.setCount(this.getCount() + p_41770_);
   }
// 物品count减少指定数目
   public void shrink(int p_41775_) {
      this.grow(-p_41775_);
   }
// 获得物品的使用时间
   public void onUseTick(Level p_41732_, LivingEntity p_41733_, int p_41734_) {
      this.getItem().onUseTick(p_41732_, p_41733_, this, p_41734_);
   }
// Forge 中已弃用此方法，不应使用。它调用item实体所代表的项目的 onDestroyed() 方法。
   /** @deprecated Forge: Use {@linkplain IItemStackExtension#onDestroyed(ItemEntity, net.minecraft.world.damagesource.DamageSource) damage source sensitive version} */
   @Deprecated
   public void onDestroyed(ItemEntity p_150925_) {
      this.getItem().onDestroyed(p_150925_);
   }
//此方法返回该物品是否可食用。它调用项目实体所代表的项目的 isEdible() 方法。
   public boolean isEdible() {
      return this.getItem().isEdible();
   }
//此方法从CompoundTag 反序列化NBT 数据。它从 NBT 数据创建一个新的 ItemStack，然后将标签和功能从 ItemStack 复制到item实体。
   // FORGE START
   public void deserializeNBT(CompoundTag nbt) {
      final ItemStack itemStack = ItemStack.of(nbt);
      this.setTag(itemStack.getTag());
      if (itemStack.capNBT != null) deserializeCaps(itemStack.capNBT);
   }

   /**
    * Set up forge's ItemStack additions.
    */
    //此方法初始化 Forge 的 ItemStack 添加项。它从项目中收集功能并反序列化 NBT 数据中存在的任何功能。
   private void forgeInit() {
      if (this.delegate != null) {
         this.gatherCapabilities(() -> item.initCapabilities(this, this.capNBT));
         if (this.capNBT != null) deserializeCaps(this.capNBT);
      }
   }
//此方法返回消耗物品时播放的声音事件。它调用项目实体所代表的项目的 getDrinkingSound() 方法。
   public SoundEvent getDrinkingSound() {
      return this.getItem().getDrinkingSound();
   }
//此方法返回消耗物品时播放的声音事件。它调用项目实体所代表的项目的 getEatingSound() 方法。
   public SoundEvent getEatingSound() {
      return this.getItem().getEatingSound();
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
//这行代码为每个 TooltipPart 成员分配了一个掩码。掩码用于在物品提示中显示特定部分。
      private final int mask = 1 << this.ordinal();
//这行代码返回 TooltipPart 成员的掩码。
      public int getMask() {
         return this.mask;
      }
       //例如，如果要显示物品的附魔效果和属性修饰器，可以使用以下代码：
//int mask = TooltipPart.ENCHANTMENTS.getMask() | TooltipPart.MODIFIERS.getMask();

   }
}

```

