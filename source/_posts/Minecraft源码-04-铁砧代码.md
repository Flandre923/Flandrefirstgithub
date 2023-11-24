---
title: Minecraft源码-04-铁砧代码
date: 2023-11-15 17:29:29
tags:
- 我的世界
- 源码
- Java
cover: https://view.moezx.cc/images/2019/01/12/72301507_p0.png
---

# AnvilBlock类

```java

public class AnvilBlock extends FallingBlock {
    // 朝向
   public static final DirectionProperty FACING = HorizontalDirectionalBlock.FACING;
    // 碰撞box
   private static final VoxelShape BASE = Block.box(2.0, 0.0, 2.0, 14.0, 4.0, 14.0);
   private static final VoxelShape X_LEG1 = Block.box(3.0, 4.0, 4.0, 13.0, 5.0, 12.0);
   private static final VoxelShape X_LEG2 = Block.box(4.0, 5.0, 6.0, 12.0, 10.0, 10.0);
   private static final VoxelShape X_TOP = Block.box(0.0, 10.0, 3.0, 16.0, 16.0, 13.0);
   private static final VoxelShape Z_LEG1 = Block.box(4.0, 4.0, 3.0, 12.0, 5.0, 13.0);
   private static final VoxelShape Z_LEG2 = Block.box(6.0, 5.0, 4.0, 10.0, 10.0, 12.0);
   private static final VoxelShape Z_TOP = Block.box(3.0, 10.0, 0.0, 13.0, 16.0, 16.0);
    // x轴碰撞box
   private static final VoxelShape X_AXIS_AABB = Shapes.or(BASE, X_LEG1, X_LEG2, X_TOP);
    // z轴碰撞box
   private static final VoxelShape Z_AXIS_AABB = Shapes.or(BASE, Z_LEG1, Z_LEG2, Z_TOP);
    // gui的标题
   private static final Component CONTAINER_TITLE = Component.translatable("container.repair");
    // 掉落每block造成伤害
   private static final float FALL_DAMAGE_PER_DISTANCE = 2.0F;
    // 最大造成伤害 
   private static final int FALL_DAMAGE_MAX = 40;

   public AnvilBlock(BlockBehaviour.Properties p_48777_) {
      super(p_48777_);
       // 设置默认为北方
      this.registerDefaultState(this.stateDefinition.any().setValue(FACING, Direction.NORTH));
   }
//方块被放置时获取方块的状态。它接受一个BlockPlaceContext对象作为参数，并返回一个BlockState对象，该对象表示方块的状态。
   @Override
   public BlockState getStateForPlacement(BlockPlaceContext p_48781_) {
       // 设置朝向为当前玩家朝向的反方向
      return this.defaultBlockState().setValue(FACING, p_48781_.getHorizontalDirection().getClockWise());
   }
// 如果方块在客户端（即玩家的设备）上，它将返回InteractionResult.SUCCESS。否则，它将打开一个新的菜单，并返回InteractionResult.CONSUME。

@Override
   @Override
   public InteractionResult use(BlockState p_48804_, Level p_48805_, BlockPos p_48806_, Player p_48807_, InteractionHand p_48808_, BlockHitResult p_48809_) {
      if (p_48805_.isClientSide) {
         return InteractionResult.SUCCESS;
      } else {
          // 打开一个menu
         p_48807_.openMenu(p_48804_.getMenuProvider(p_48805_, p_48806_));
         p_48807_.awardStat(Stats.INTERACT_WITH_ANVIL);
         return InteractionResult.CONSUME;
      }
   }
 //方法用于返回一个新的菜单提供者，该提供者在玩家与方块交互时打开一个新的菜单。
   @Nullable
   @Override
   public MenuProvider getMenuProvider(BlockState p_48821_, Level p_48822_, BlockPos p_48823_) {
      // 返回一个SimpleMenuProvider
      return new SimpleMenuProvider(
         (p_48785_, p_48786_, p_48787_) -> new AnvilMenu(p_48785_, p_48786_, ContainerLevelAccess.create(p_48822_, p_48823_)), CONTAINER_TITLE
      );
   }
// 获取方块的形状。它接受一个BlockState对象、一个BlockGetter对象和一个BlockPos对象作为参数，并返回一个VoxelShape对象，该对象表示方块的形状。
   @Override
   public VoxelShape getShape(BlockState p_48816_, BlockGetter p_48817_, BlockPos p_48818_, CollisionContext p_48819_) {
       //根据 x z 返回不同的碰撞box
      Direction direction = p_48816_.getValue(FACING);
      return direction.getAxis() == Direction.Axis.X ? X_AXIS_AABB : Z_AXIS_AABB;
   }
//falling方法用于处理方块下落时造成的伤害。它接受一个FallingBlockEntity对象作为参数，并设置该对象的伤害属性。
   @Override
   protected void falling(FallingBlockEntity p_48779_) {
      p_48779_.setHurtsEntities(2.0F, 40);
   }
//onLand和onBrokenAfterFall方法用于处理方块落地和破碎后的事件。它们接受一个Level对象、一个BlockPos对象和一个BlockState对象作为参数，并在方块落地或破碎后触发一个level事件。
   @Override
   public void onLand(Level p_48793_, BlockPos p_48794_, BlockState p_48795_, BlockState p_48796_, FallingBlockEntity p_48797_) {
       // 不是静音
      if (!p_48797_.isSilent()) {
         p_48793_.levelEvent(1031, p_48794_, 0);
      }
   }

   @Override
   public void onBrokenAfterFall(Level p_152053_, BlockPos p_152054_, FallingBlockEntity p_152055_) {
       // 不是静音
      if (!p_152055_.isSilent()) {
         p_152053_.levelEvent(1029, p_152054_, 0);
      }
   }
//获取方块下落时造成的伤害源。这个方法接受一个Entity对象作为参数，并返回一个DamageSource对象，该对象表示方块下落时造成的伤害源。
   @Override
   public DamageSource getFallDamageSource(Entity p_254036_) {
      return p_254036_.damageSources().anvil(p_254036_);
   }
//处理方块被损坏的情况。如果方块是ANVIL，那么它会变成破损的ANVIL。如果方块是破损的ANVIL，那么它会变成损坏的锤子。如果方块既不是ANVIL也不是破损的ANVIL，那么这个方法将返回null。
   @Nullable
   public static BlockState damage(BlockState p_48825_) {
      if (p_48825_.is(Blocks.ANVIL)) {
         return Blocks.CHIPPED_ANVIL.defaultBlockState().setValue(FACING, p_48825_.getValue(FACING));
      } else {
         return p_48825_.is(Blocks.CHIPPED_ANVIL) ? Blocks.DAMAGED_ANVIL.defaultBlockState().setValue(FACING, p_48825_.getValue(FACING)) : null;
      }
   }
//rotate方法用于处理方块旋转的情况。这个方法接受一个BlockState对象和一个Rotation对象作为参数，并返回一个新的BlockState对象，该对象表示方块旋转后的状态。
   @Override
   public BlockState rotate(BlockState p_48811_, Rotation p_48812_) {
      return p_48811_.setValue(FACING, p_48812_.rotate(p_48811_.getValue(FACING)));
   }
//于定义方块的状态。这个方法接受一个StateDefinition.Builder对象作为参数，并添加方块的朝向状态。
   @Override
   protected void createBlockStateDefinition(StateDefinition.Builder<Block, BlockState> p_48814_) {
      p_48814_.add(FACING);
   }
//判断方块是否可以被路径查找。这个方法接受一个BlockState对象、一个BlockGetter对象、一个BlockPos对象和一个PathComputationType对象作为参数，并返回一个布尔值，表示方块是否可以被路径查找。在这个例子中，方块不可以被路径查找。
   @Override
   public boolean isPathfindable(BlockState p_48799_, BlockGetter p_48800_, BlockPos p_48801_, PathComputationType p_48802_) {
      return false;
   }
//获取方块的粉尘颜色。这个方法接受一个BlockState对象、一个BlockGetter对象和一个BlockPos对象作为参数，并返回一个整数，表示方块的粉尘颜色。
   @Override
   public int getDustColor(BlockState p_48827_, BlockGetter p_48828_, BlockPos p_48829_) {
      return p_48827_.getMapColor(p_48828_, p_48829_).col;
   }
}

```

# AnvilMenu 类

```java

public class AnvilMenu extends ItemCombinerMenu {
    // 输入slot 0
   public static final int INPUT_SLOT = 0;
    // 另一个输入slot 1 
   public static final int ADDITIONAL_SLOT = 1;
    // 输出slot 2
   public static final int RESULT_SLOT = 2;
    // 日志
   private static final Logger LOGGER = LogUtils.getLogger();
    // 表示是否开启调试模式。
   private static final boolean DEBUG_COST = false;
    // 用于表示物品名称的最大长度。
   public static final int MAX_NAME_LENGTH = 50;
    // 用于表示修复物品的数量成本。
   public int repairItemCountCost;
    // 用于表示物品的名称。
   @Nullable
   private String itemName;
    // cost是一个DataSlot对象，用于表示修复的成本。
   private final DataSlot cost = DataSlot.standalone();
    // 定义的常量，用于表示修复的不同成本。
   private static final int COST_FAIL = 0;
   private static final int COST_BASE = 1;
   private static final int COST_ADDED_BASE = 1;
   private static final int COST_REPAIR_MATERIAL = 1;
   private static final int COST_REPAIR_SACRIFICE = 2;
   private static final int COST_INCOMPATIBLE_PENALTY = 1;
   private static final int COST_RENAME = 1;
    // 是定义的常量，用于表示输入槽、额外槽和结果槽的X坐标。
   private static final int INPUT_SLOT_X_PLACEMENT = 27;
   private static final int ADDITIONAL_SLOT_X_PLACEMENT = 76;
   private static final int RESULT_SLOT_X_PLACEMENT = 134;
    // Y坐标
   private static final int SLOT_Y_PLACEMENT = 47;
	// 构造函数接受一个整数和一个Inventory对象作为参数，并将它们传递给父类的构造函数。
   public AnvilMenu(int p_39005_, Inventory p_39006_) {
      this(p_39005_, p_39006_, ContainerLevelAccess.NULL);
   }
	//
   public AnvilMenu(int p_39008_, Inventory p_39009_, ContainerLevelAccess p_39010_) {
      super(MenuType.ANVIL, p_39008_, p_39009_, p_39010_);
      this.addDataSlot(this.cost);
   }
 	// 创建slot的定义。
   @Override
   protected ItemCombinerMenuSlotDefinition createInputSlotDefinitions() {
      return ItemCombinerMenuSlotDefinition.create()
         .withSlot(0, 27, 47, p_266635_ -> true)
         .withSlot(1, 76, 47, p_266634_ -> true)
         .withResultSlot(2, 134, 47)
         .build();
   }
//判断方块是否是铁砧。
   @Override
   protected boolean isValidBlock(BlockState p_39019_) {
      return p_39019_.is(BlockTags.ANVIL);
   }
//判断玩家是否可以合成物品。
   @Override
   protected boolean mayPickup(Player p_39023_, boolean p_39024_) {
      return (p_39023_.getAbilities().instabuild || p_39023_.experienceLevel >= this.cost.get()) && this.cost.get() > 0;
   }
//处理玩家拾取物品的事件。
   @Override
   protected void onTake(Player p_150474_, ItemStack p_150475_) {
       // 生存模式
      if (!p_150474_.getAbilities().instabuild) {
          //扣除玩家cost的经验
         p_150474_.giveExperienceLevels(-this.cost.get());
      }
       //定义了一个名为breakChance的浮点数变量，用于表示破损的概率,这个概率是通过调用net.neoforged.neoforge.common.CommonHooks.onAnvilRepair方法计算得出的
      float breakChance = net.neoforged.neoforge.common.CommonHooks.onAnvilRepair(p_150474_, p_150475_, AnvilMenu.this.inputSlots.getItem(0), AnvilMenu.this.inputSlots.getItem(1));
//清空第一个输入槽的物品。
      this.inputSlots.setItem(0, ItemStack.EMPTY);
       //检查repairItemCountCost的值。如果repairItemCountCost大于0，那么获取第二个输入槽的物品，并检查它的数量是否大于repairItemCountCost。如果是，那么将repairItemCountCost从物品的数量中减去，并将减少后的物品重新放入第二个输入槽。否则，清空第二个输入槽的物品。
      if (this.repairItemCountCost > 0) {
         ItemStack itemstack = this.inputSlots.getItem(1);
         if (!itemstack.isEmpty() && itemstack.getCount() > this.repairItemCountCost) {
            itemstack.shrink(this.repairItemCountCost);
            this.inputSlots.setItem(1, itemstack);
         } else {
            this.inputSlots.setItem(1, ItemStack.EMPTY);
         }
      } else {
         this.inputSlots.setItem(1, ItemStack.EMPTY);
      }
//将cost的值设置为0。
      this.cost.set(0);
       //执行一个lambda表达式，该表达式用于处理铁砧的破损。如果玩家生存模式，并且铁砧的状态是铁砧，并且随机数小于breakChance，那么调用AnvilBlock.damage方法破损铁砧，并触发相应的破损事件。否则，触发破损事件。
      this.access.execute((p_150479_, p_150480_) -> {
         BlockState blockstate = p_150479_.getBlockState(p_150480_);
         if (!p_150474_.getAbilities().instabuild && blockstate.is(BlockTags.ANVIL) && p_150474_.getRandom().nextFloat() < breakChance) {
            BlockState blockstate1 = AnvilBlock.damage(blockstate);
            if (blockstate1 == null) {
               p_150479_.removeBlock(p_150480_, false);
               p_150479_.levelEvent(1029, p_150480_, 0);
            } else {
               p_150479_.setBlock(p_150480_, blockstate1, 2);
               p_150479_.levelEvent(1030, p_150480_, 0);
            }
         } else {
            p_150479_.levelEvent(1030, p_150480_, 0);
         }
      });
   }
//用于计算修复物品的结果。
   @Override
   public void createResult() {
       //获取第一个输入槽的物品，并将cost的值设置为1。
      ItemStack itemstack = this.inputSlots.getItem(0);
      this.cost.set(1);
       //初始化三个整数变量i、j和k，用于存储修复的成本。
      int i = 0;
      int j = 0;
      int k = 0;
       //检查第一个输入槽的物品是否为空。如果为空，那么清空结果槽的物品，并将cost的值设置为0。
      if (itemstack.isEmpty()) {
         this.resultSlots.setItem(0, ItemStack.EMPTY);
         this.cost.set(0);
      } else {
          //一个输入槽的物品不为空，那么复制第一个输入槽的物品，并获取第二个输入槽的物品。然后，获取第一个输入槽的物品的附魔列表，并计算修复的成本。
         ItemStack itemstack1 = itemstack.copy();
         ItemStack itemstack2 = this.inputSlots.getItem(1);
         Map<Enchantment, Integer> map = EnchantmentHelper.getEnchantments(itemstack1);
         j += itemstack.getBaseRepairCost() + (itemstack2.isEmpty() ? 0 : itemstack2.getBaseRepairCost());
         this.repairItemCountCost = 0;
         boolean flag = false;
//然后，检查是否需要修复物品。如果不需要，那么返回。
         if (!net.neoforged.neoforge.common.CommonHooks.onAnvilChange(this, itemstack, itemstack2, resultSlots, itemName, j, this.player)) return;
          //果需要修复物品，那么检查第二个输入槽的物品是否为空。如果不为空，那么检查第一个输入槽的物品是否可以被修复，并计算修复的成本。
         if (!itemstack2.isEmpty()) {
            flag = itemstack2.getItem() == Items.ENCHANTED_BOOK && !EnchantedBookItem.getEnchantments(itemstack2).isEmpty();
            if (itemstack1.isDamageableItem() && itemstack1.getItem().isValidRepairItem(itemstack, itemstack2)) {
               int l2 = Math.min(itemstack1.getDamageValue(), itemstack1.getMaxDamage() / 4);
               if (l2 <= 0) {
                  this.resultSlots.setItem(0, ItemStack.EMPTY);
                  this.cost.set(0);
                  return;
               }

               int i3;
               for(i3 = 0; l2 > 0 && i3 < itemstack2.getCount(); ++i3) {
                  int j3 = itemstack1.getDamageValue() - l2;
                  itemstack1.setDamageValue(j3);
                  ++i;
                  l2 = Math.min(itemstack1.getDamageValue(), itemstack1.getMaxDamage() / 4);
               }

               this.repairItemCountCost = i3;
            } else {
                //如果第一个输入槽的物品不可以被修复，那么检查第二个输入槽的物品是否可以用来修复第一个输入槽的物品。如果可以，那么计算修复的成本。
               if (!flag && (!itemstack1.is(itemstack2.getItem()) || !itemstack1.isDamageableItem())) {
                  this.resultSlots.setItem(0, ItemStack.EMPTY);
                  this.cost.set(0);
                  return;
               }

               if (itemstack1.isDamageableItem() && !flag) {
                  int l = itemstack.getMaxDamage() - itemstack.getDamageValue();
                  int i1 = itemstack2.getMaxDamage() - itemstack2.getDamageValue();
                  int j1 = i1 + itemstack1.getMaxDamage() * 12 / 100;
                  int k1 = l + j1;
                  int l1 = itemstack1.getMaxDamage() - k1;
                  if (l1 < 0) {
                     l1 = 0;
                  }

                  if (l1 < itemstack1.getDamageValue()) {
                     itemstack1.setDamageValue(l1);
                     i += 2;
                  }
               }
//如果第二个输入槽的物品不能用来修复第一个输入槽的物品，那么检查第二个输入槽的物品的附魔列表。
//如果第二个输入槽的物品的附魔列表不为空，那么遍历附魔列表，并对每个附魔进行处理。如果附魔不为空，那么获取第一个输入槽的物品的附魔列表中的该附魔的等级，或者如果第一个输入槽的物品的附魔列表中没有该附魔，那么获取第二个输入槽的物品的附魔列表中的该附魔的等级。然后，如果附魔的等级不相等，那么将较大的等级设置为新的等级。如果新的等级超过了附魔的最大等级，那么将新的等级设置为附魔的最大等级。然后，将新的等级添加到第一个输入槽的物品的附魔列表中。
               Map<Enchantment, Integer> map1 = EnchantmentHelper.getEnchantments(itemstack2);
               boolean flag2 = false;
               boolean flag3 = false;

               for(Enchantment enchantment1 : map1.keySet()) {
                  if (enchantment1 != null) {
                     int i2 = map.getOrDefault(enchantment1, 0);
                     int j2 = map1.get(enchantment1);
                     j2 = i2 == j2 ? j2 + 1 : Math.max(j2, i2);
                     boolean flag1 = enchantment1.canEnchant(itemstack);
                     if (this.player.getAbilities().instabuild || itemstack.is(Items.ENCHANTED_BOOK)) {
                        flag1 = true;
                     }

                     for(Enchantment enchantment : map.keySet()) {
                        if (enchantment != enchantment1 && !enchantment1.isCompatibleWith(enchantment)) {
                           flag1 = false;
                           ++i;
                        }
                     }

                     if (!flag1) {
                        flag3 = true;
                     } else {
                        flag2 = true;
                        if (j2 > enchantment1.getMaxLevel()) {
                           j2 = enchantment1.getMaxLevel();
                        }

                        map.put(enchantment1, j2);
                        int k3 = 0;
                        switch(enchantment1.getRarity()) {
                           case COMMON:
                              k3 = 1;
                              break;
                           case UNCOMMON:
                              k3 = 2;
                              break;
                           case RARE:
                              k3 = 4;
                              break;
                           case VERY_RARE:
                              k3 = 8;
                        }

                        if (flag) {
                           k3 = Math.max(1, k3 / 2);
                        }

                        i += k3 * j2;
                        if (itemstack.getCount() > 1) {
                           i = 40;
                        }
                     }
                  }
               }
//如果所有的附魔都不能添加到第一个输入槽的物品的附魔列表中，那么清空结果槽的物品，并将cost的值设置为0。
               if (flag3 && !flag2) {
                  this.resultSlots.setItem(0, ItemStack.EMPTY);
                  this.cost.set(0);
                  return;
               }
            }
         }
//第一个输入槽的物品的名称不为空，并且不是空格，那么检查第一个输入槽的物品的名称是否与第二个输入槽的物品的名称相同。如果不相同，那么将k的值设置为1，并将i的值增加k的值。然后，将第一个输入槽的物品的名称设置为itemName。
         if (this.itemName != null && !Util.isBlank(this.itemName)) {
            if (!this.itemName.equals(itemstack.getHoverName().getString())) {
               k = 1;
               i += k;
               itemstack1.setHoverName(Component.literal(this.itemName));
            }
         } else if (itemstack.hasCustomHoverName()) {
            k = 1;
            i += k;
            itemstack1.resetHoverName();
         }
          //如果flag为真，并且第一个输入槽的物品不可以被附魔，那么将itemstack1的值设置为空。
         if (flag && !itemstack1.isBookEnchantable(itemstack2)) itemstack1 = ItemStack.EMPTY;
//然后，将cost的值设置为j和i的和。
         this.cost.set(j + i);
          //如果i的值小于等于0，那么将itemstack1的值设置为空。
         if (i <= 0) {
            itemstack1 = ItemStack.EMPTY;
         }
//如果k的值等于i的值，并且k的值大于0，并且cost的值大于等于40，那么将cost的值设置为39。
         if (k == i && k > 0 && this.cost.get() >= 40) {
            this.cost.set(39);
         }
//如果cost的值大于等于40，并且玩家没有无敌模式，那么将itemstack1的值设置为空。
         if (this.cost.get() >= 40 && !this.player.getAbilities().instabuild) {
            itemstack1 = ItemStack.EMPTY;
         }
//如果itemstack1的值不为空，那么获取第一个输入槽的物品的基础修复成本，并检查第二个输入槽的物品是否为空。如果不为空，并且第二个输入槽的物品的基础修复成本大于第一个输入槽的物品的基础修复成本，那么将第二个输入槽的物品的基础修复成本设置为新的修复成本。
         if (!itemstack1.isEmpty()) {
            int k2 = itemstack1.getBaseRepairCost();
            if (!itemstack2.isEmpty() && k2 < itemstack2.getBaseRepairCost()) {
               k2 = itemstack2.getBaseRepairCost();
            }
//如果k的值不等于i的值，或者k的值等于0，那么将新的修复成本计算为增加的修复成本。
            if (k != i || k == 0) {
               k2 = calculateIncreasedRepairCost(k2);
            }
//然后，将新的修复成本设置为itemstack1的修复成本，并将itemstack1的附魔列表设置为map。
            itemstack1.setRepairCost(k2);
            EnchantmentHelper.setEnchantments(map, itemstack1);
         }
//最后，将itemstack1的值设置为结果槽的物品，并广播更改。
         this.resultSlots.setItem(0, itemstack1);
         this.broadcastChanges();
      }
   }
//它接受一个整数作为参数 ( p_39026_ )，这可能代表该项目的基本成本。然后，该方法返回基本成本乘以 2，然后再增加 1。这表明基本成本每增加一个单位，维修成本就会增加 100%
   public static int calculateIncreasedRepairCost(int p_39026_) {
      return p_39026_ * 2 + 1;
   }
// 它接受一个字符串作为参数 ( p_288970_ )，这是该物品的新名称。该方法首先使用 validateName 方法验证名称。如果名称有效且与当前项目名称不同，则该方法设置新名称，更新slot 2 中项目的悬停名称（如果存在），然后调用createResult方法。如果名称无效或与当前项目名称相同，则该方法返回 
   public boolean setItemName(String p_288970_) {
      String s = validateName(p_288970_);
      if (s != null && !s.equals(this.itemName)) {
         this.itemName = s;
         if (this.getSlot(2).hasItem()) {
            ItemStack itemstack = this.getSlot(2).getItem();
            if (Util.isBlank(s)) {
               itemstack.resetHoverName();
            } else {
               itemstack.setHoverName(Component.literal(s));
            }
         }

         this.createResult();
         return true;
      } else {
         return false;
      }
   }
//它接受一个字符串作为参数 ( p_288995_ )，这是要验证的名称。该方法使用 SharedConstants.filterText 过滤文本，并检查过滤后文本的长度是否小于或等于50。如果是，则返回过滤后的文本；否则，返回过滤后的文本。否则，返回 null 
   @Nullable
   private static String validateName(String p_288995_) {
      String s = SharedConstants.filterText(p_288995_);
      return s.length() <= 50 ? s : null;
   }
//此方法返回该项目的当前成本。它通过调用 cost 对象上的 get 方法来完成此操作。
   public int getCost() {
      return this.cost.get();
   }
//此方法设置项目的最大成本。它接受一个整数作为参数 ( value )，这是新的最大成本。然后，该方法通过调用 cost 对象 ayokoding.com 上的 set 方法将 cost 对象设置为新的最大成本。
   public void setMaximumCost(int value) {
      this.cost.set(value);
   }
}

```

# AnvilScreen类

```java
//用于为游戏中的铁砧创建图形用户界面 (GUI)。
//类只能在游戏的客户端加载
@OnlyIn(Dist.CLIENT)
//它扩展了 ItemCombinerScreen<AnvilMenu> ，这表明它是一种允许玩家以某种方式组合项目的屏幕。
public class AnvilScreen extends ItemCombinerScreen<AnvilMenu> {
    //用于存储 GUI 中使用的各种精灵的位置。这些精灵可能用于在屏幕 nekoyue.github.io 上绘制 GUI 元素。
   private static final ResourceLocation TEXT_FIELD_SPRITE = new ResourceLocation("container/anvil/text_field");
   private static final ResourceLocation TEXT_FIELD_DISABLED_SPRITE = new ResourceLocation("container/anvil/text_field_disabled");
   private static final ResourceLocation ERROR_SPRITE = new ResourceLocation("container/anvil/error");
   private static final ResourceLocation ANVIL_LOCATION = new ResourceLocation("textures/gui/container/anvil.png");
   private static final Component TOO_EXPENSIVE_TEXT = Component.translatable("container.repair.expensive");
    // name 是一个 EditBox ，玩家可以使用它来输入项目的名称。
   private EditBox name;
    //。 player 是一个 Player 对象，代表正在使用 anvil 的玩家。
   private final Player player;
//它需要三个参数： AnvilMenu 、 Inventory 和 Component 。
    //AnvilMenu 可能是玩家使用铁砧时显示的Menu。 Inventory 可能是玩家的背包， Component 可能是屏幕上显示的标题
   public AnvilScreen(AnvilMenu p_97874_, Inventory p_97875_, Component p_97876_) {
      super(p_97874_, p_97875_, p_97876_, ANVIL_LOCATION);
      this.player = p_97875_.player;
       //title 位置
      this.titleLabelX = 60;
   }
//屏幕初始化时被调用。它设置屏幕的 GUI 元素，包括用于输入项目名称的 EditBox 和初始焦点
   @Override
   protected void subInit() {
      int i = (this.width - this.imageWidth) / 2;
      int j = (this.height - this.imageHeight) / 2;
      this.name = new EditBox(this.font, i + 62, j + 24, 103, 12, Component.translatable("container.repair"));
      this.name.setCanLoseFocus(false);
      this.name.setTextColor(-1);
      this.name.setTextColorUneditable(-1);
      this.name.setBordered(false);
      this.name.setMaxLength(50);
      this.name.setResponder(this::onNameChanged);
      this.name.setValue("");
      this.addWidget(this.name);
      this.setInitialFocus(this.name);
      this.name.setEditable(this.menu.getSlot(0).hasItem());
   }
//调整屏幕大小时会调用此方法。它保存 EditBox 的当前值，重新初始化屏幕，然后恢复 EditBox 的值。
   @Override
   public void resize(Minecraft p_97886_, int p_97887_, int p_97888_) {
      String s = this.name.getValue();
      this.init(p_97886_, p_97887_, p_97888_);
      this.name.setValue(s);
   }
//当按下某个键时会调用此方法。它检查该键是否为转义键，如果是，则关闭容器。否则，它会检查 EditBox 是否可以使用输入，如果不能，则将按键传递给超类
   @Override
   public boolean keyPressed(int p_97878_, int p_97879_, int p_97880_) {
      if (p_97878_ == 256) {
         this.minecraft.player.closeContainer();
      }

      return !this.name.keyPressed(p_97878_, p_97879_, p_97880_) && !this.name.canConsumeInput() ? super.keyPressed(p_97878_, p_97879_, p_97880_) : true;
   }
//当item名称更改时调用此方法。它检查菜单槽 0 中的项目是否有项目，如果有，它会向服务器发送一个数据包以将该项目重命名该item
   private void onNameChanged(String p_97899_) {
      Slot slot = this.menu.getSlot(0);
      if (slot.hasItem()) {
         String s = p_97899_;
         if (!slot.getItem().hasCustomHoverName() && p_97899_.equals(slot.getItem().getHoverName().getString())) {
            s = "";
         }

         if (this.menu.setItemName(s)) {
            this.minecraft.player.connection.send(new ServerboundRenameItemPacket(s));
         }
      }
   }
//调用此方法以在屏幕上呈现标签。它检查物品的成本，如果成本大于0，它会根据成本以及玩家是否负担得起来设置文本和文本本身的颜色。如果成本大于或等于 40 并且玩家不是创造模式，则会将文本设置为“太昂贵”。如果成本小于 40，则会将文本设置为“维修成本：[成本]”。如果result slot没有item，则会将文本设置为空。然后它在屏幕上绘制文本
   @Override
   protected void renderLabels(GuiGraphics p_281442_, int p_282417_, int p_283022_) {
      super.renderLabels(p_281442_, p_282417_, p_283022_);
      int i = this.menu.getCost();
      if (i > 0) {
         int j = 8453920;
         Component component;
         if (i >= 40 && !this.minecraft.player.getAbilities().instabuild) {
            component = TOO_EXPENSIVE_TEXT;
            j = 16736352;
         } else if (!this.menu.getSlot(2).hasItem()) {
            component = null;
         } else {
            component = Component.translatable("container.repair.cost", i);
            if (!this.menu.getSlot(2).mayPickup(this.player)) {
               j = 16736352;
            }
         }

         if (component != null) {
            int k = this.imageWidth - 8 - this.font.width(component) - 2;
            int l = 69;
            p_281442_.fill(k - 2, 67, this.imageWidth - 8, 79, 1325400064);
            p_281442_.drawString(this.font, component, k, 69, j);
         }
      }
   }
//调用该方法来渲染屏幕背景。它根据插槽 0 是否有项目来 blit（绘制）精灵 
   @Override
   protected void renderBg(GuiGraphics p_283345_, float p_283412_, int p_282871_, int p_281306_) {
      super.renderBg(p_283345_, p_283412_, p_282871_, p_281306_);
      p_283345_.blitSprite(this.menu.getSlot(0).hasItem() ? TEXT_FIELD_SPRITE : TEXT_FIELD_DISABLED_SPRITE, this.leftPos + 59, this.topPos + 20, 110, 16);
   }
//调用该方法来渲染屏幕的前景。它呈现用于输入item名称 的 EditBox 。
   @Override
   public void renderFg(GuiGraphics p_283449_, int p_283263_, int p_281526_, float p_282957_) {
      this.name.render(p_283449_, p_283263_, p_281526_, p_282957_);
   }
//调用此方法以在屏幕上呈现错误图标。它检查插槽 0 和 1 是否有item，以及结果插槽是否没有item，如果满足这些条件，它会为错误图标生成一个 sprite。
   @Override
   protected void renderErrorIcon(GuiGraphics p_282905_, int p_283237_, int p_282237_) {
      if ((this.menu.getSlot(0).hasItem() || this.menu.getSlot(1).hasItem()) && !this.menu.getSlot(this.menu.getResultSlot()).hasItem()) {
         p_282905_.blitSprite(ERROR_SPRITE, p_283237_ + 99, p_282237_ + 45, 28, 21);
      }
   }

   @Override
   public void slotChanged(AbstractContainerMenu p_97882_, int p_97883_, ItemStack p_97884_) {
      if (p_97883_ == 0) {
         this.name.setValue(p_97884_.isEmpty() ? "" : p_97884_.getHoverName().getString());
         this.name.setEditable(!p_97884_.isEmpty());
         this.setFocused(this.name);
      }
   }
}
```

# ServerGamePacketListenerImpl 类

```java
   @Override
   public void handleRenameItem(ServerboundRenameItemPacket p_9899_) {
      PacketUtils.ensureRunningOnSameThread(p_9899_, this, this.player.serverLevel());
      AbstractContainerMenu abstractcontainermenu = this.player.containerMenu;
      if (abstractcontainermenu instanceof AnvilMenu anvilmenu) {
         if (!anvilmenu.stillValid(this.player)) {
            LOGGER.debug("Player {} interacted with invalid menu {}", this.player, anvilmenu);
            return;
         }

         anvilmenu.setItemName(p_9899_.getName());
      }
   }

```



# ItemCombinerMenu 类

```java

public abstract class ItemCombinerMenu extends AbstractContainerMenu {
    // 玩家背包行
   private static final int INVENTORY_SLOTS_PER_ROW = 9;
    // 玩家背包列
   private static final int INVENTORY_SLOTS_PER_COLUMN = 3;
    // 
   protected final ContainerLevelAccess access;
   protected final Player player;
    // 输入 slot
   protected final Container inputSlots;
    // 输入 slot index
   private final List<Integer> inputSlotIndexes;
    // result slot
   protected final ResultContainer resultSlots = new ResultContainer();
    // result slot index
   private final int resultSlotIndex;
// 子类实现方法
    // 能否被拿起
   protected abstract boolean mayPickup(Player p_39798_, boolean p_39799_);
// 能否拿出
   protected abstract void onTake(Player p_150601_, ItemStack p_150602_);
// 是否合法方块
   protected abstract boolean isValidBlock(BlockState p_39788_);
// 构造函数
   public ItemCombinerMenu(@Nullable MenuType<?> p_39773_, int p_39774_, Inventory p_39775_, ContainerLevelAccess p_39776_) {
      super(p_39773_, p_39774_);
      this.access = p_39776_;
      this.player = p_39775_.player;
      ItemCombinerMenuSlotDefinition itemcombinermenuslotdefinition = this.createInputSlotDefinitions();
      this.inputSlots = this.createContainer(itemcombinermenuslotdefinition.getNumOfInputSlots());
      this.inputSlotIndexes = itemcombinermenuslotdefinition.getInputSlotIndexes();
      this.resultSlotIndex = itemcombinermenuslotdefinition.getResultSlotIndex();
      this.createInputSlots(itemcombinermenuslotdefinition);
      this.createResultSlot(itemcombinermenuslotdefinition);
      this.createInventorySlots(p_39775_);
   }
// 创建输入slot
   private void createInputSlots(ItemCombinerMenuSlotDefinition p_267172_) {
      for(final ItemCombinerMenuSlotDefinition.SlotDefinition itemcombinermenuslotdefinition$slotdefinition : p_267172_.getSlots()) {
         this.addSlot(
            new Slot(
               this.inputSlots,
               itemcombinermenuslotdefinition$slotdefinition.slotIndex(),
               itemcombinermenuslotdefinition$slotdefinition.x(),
               itemcombinermenuslotdefinition$slotdefinition.y()
            ) {
                // 能否被放入
               @Override
               public boolean mayPlace(ItemStack p_267156_) {
                  return itemcombinermenuslotdefinition$slotdefinition.mayPlace().test(p_267156_);
               }
            }
         );
      }
   }
// 创建输出的slot
   private void createResultSlot(ItemCombinerMenuSlotDefinition p_267000_) {
      this.addSlot(new Slot(this.resultSlots, p_267000_.getResultSlot().slotIndex(), p_267000_.getResultSlot().x(), p_267000_.getResultSlot().y()) {
          // 能否被放置
         @Override
         public boolean mayPlace(ItemStack p_39818_) {
            return false;
         }
// 能否被拿起，调用抽象方法，由子类实现
         @Override
         public boolean mayPickup(Player p_39813_) {
            return ItemCombinerMenu.this.mayPickup(p_39813_, this.hasItem());
         }
// 能否被拿出，调用抽象方法，由子类实现
         @Override
         public void onTake(Player p_150604_, ItemStack p_150605_) {
            ItemCombinerMenu.this.onTake(p_150604_, p_150605_);
         }
      });
   }
// 创建玩家的背包
   private void createInventorySlots(Inventory p_267325_) {
      for(int i = 0; i < 3; ++i) {
         for(int j = 0; j < 9; ++j) {
            this.addSlot(new Slot(p_267325_, j + i * 9 + 9, 8 + j * 18, 84 + i * 18));
         }
      }

      for(int k = 0; k < 9; ++k) {
         this.addSlot(new Slot(p_267325_, k, 8 + k * 18, 142));
      }
   }
// 合成结果，子类实现
   public abstract void createResult();
// 输入的slot创建函数，子类实现
   protected abstract ItemCombinerMenuSlotDefinition createInputSlotDefinitions();
// 创建simplecontainer
   private SimpleContainer createContainer(int p_267204_) {
      return new SimpleContainer(p_267204_) {
          // 重写方法，当内容改变时候，设置赃位
         @Override
         public void setChanged() {
            super.setChanged();
            ItemCombinerMenu.this.slotsChanged(this);
         }
      };
   }
//slots改变时候，如果不是输入slot则调用createresult方法。
   @Override
   public void slotsChanged(Container p_39778_) {
      super.slotsChanged(p_39778_);
      if (p_39778_ == this.inputSlots) {
         this.createResult();
      }
   }
// 当移除当前的menu时候
   @Override
   public void removed(Player p_39790_) {
      super.removed(p_39790_);
      this.access.execute((p_39796_, p_39797_) -> this.clearContainer(p_39790_, this.inputSlots));
   }
// 判断是否合法位置打开menu
   @Override
   public boolean stillValid(Player p_39780_) {
      return this.access
         .evaluate(
            (p_39785_, p_39786_) -> !this.isValidBlock(p_39785_.getBlockState(p_39786_))
                  ? false
                  : p_39780_.distanceToSqr((double)p_39786_.getX() + 0.5, (double)p_39786_.getY() + 0.5, (double)p_39786_.getZ() + 0.5) <= 64.0,
            true
         );
   }
// shift的快速移动
   @Override
   public ItemStack quickMoveStack(Player p_39792_, int p_39793_) {
      ItemStack itemstack = ItemStack.EMPTY;
      Slot slot = this.slots.get(p_39793_);
      if (slot != null && slot.hasItem()) {
         ItemStack itemstack1 = slot.getItem();
         itemstack = itemstack1.copy();
         int i = this.getInventorySlotStart();
         int j = this.getUseRowEnd();
         if (p_39793_ == this.getResultSlot()) {
            if (!this.moveItemStackTo(itemstack1, i, j, true)) {
               return ItemStack.EMPTY;
            }

            slot.onQuickCraft(itemstack1, itemstack);
         } else if (this.inputSlotIndexes.contains(p_39793_)) {
            if (!this.moveItemStackTo(itemstack1, i, j, false)) {
               return ItemStack.EMPTY;
            }
         } else if (this.canMoveIntoInputSlots(itemstack1) && p_39793_ >= this.getInventorySlotStart() && p_39793_ < this.getUseRowEnd()) {
            int k = this.getSlotToQuickMoveTo(itemstack);
            if (!this.moveItemStackTo(itemstack1, k, this.getResultSlot(), false)) {
               return ItemStack.EMPTY;
            }
         } else if (p_39793_ >= this.getInventorySlotStart() && p_39793_ < this.getInventorySlotEnd()) {
            if (!this.moveItemStackTo(itemstack1, this.getUseRowStart(), this.getUseRowEnd(), false)) {
               return ItemStack.EMPTY;
            }
         } else if (p_39793_ >= this.getUseRowStart()
            && p_39793_ < this.getUseRowEnd()
            && !this.moveItemStackTo(itemstack1, this.getInventorySlotStart(), this.getInventorySlotEnd(), false)) {
            return ItemStack.EMPTY;
         }

         if (itemstack1.isEmpty()) {
            slot.setByPlayer(ItemStack.EMPTY);
         } else {
            slot.setChanged();
         }

         if (itemstack1.getCount() == itemstack.getCount()) {
            return ItemStack.EMPTY;
         }

         slot.onTake(p_39792_, itemstack1);
      }

      return itemstack;
   }

   protected boolean canMoveIntoInputSlots(ItemStack p_39787_) {
      return true;
   }

   public int getSlotToQuickMoveTo(ItemStack p_267159_) {
      return this.inputSlots.isEmpty() ? 0 : this.inputSlotIndexes.get(0);
   }

   public int getResultSlot() {
      return this.resultSlotIndex;
   }

   private int getInventorySlotStart() {
      return this.getResultSlot() + 1;
   }

   private int getInventorySlotEnd() {
      return this.getInventorySlotStart() + 27;
   }

   private int getUseRowStart() {
      return this.getInventorySlotEnd();
   }

   private int getUseRowEnd() {
      return this.getUseRowStart() + 9;
   }
}

```



# ItemCombinerScreen类

```java

@OnlyIn(Dist.CLIENT)
public abstract class ItemCombinerScreen<T extends ItemCombinerMenu> extends AbstractContainerScreen<T> implements ContainerListener {
    // menuResource
   private final ResourceLocation menuResource;
// 
   public ItemCombinerScreen(T p_98901_, Inventory p_98902_, Component p_98903_, ResourceLocation p_98904_) {
      super(p_98901_, p_98902_, p_98903_);
      this.menuResource = p_98904_;
   }
// 子类实现
   protected void subInit() {
   }
// init函数
   @Override
   protected void init() {
      super.init();
      this.subInit();
      this.menu.addSlotListener(this);
   }
// 关闭scrren时候移除 SlotListener
   @Override
   public void removed() {
      super.removed();
      this.menu.removeSlotListener(this);
   }
// 渲染
   @Override
   public void render(GuiGraphics p_281810_, int p_283312_, int p_283420_, float p_282956_) {
      super.render(p_281810_, p_283312_, p_283420_, p_282956_);
      this.renderFg(p_281810_, p_283312_, p_283420_, p_282956_);
      this.renderTooltip(p_281810_, p_283312_, p_283420_);
   }
// 渲染文字 子类实现
   protected void renderFg(GuiGraphics p_283399_, int p_98928_, int p_98929_, float p_98930_) {
   }
// 渲染背景图片，以及error图片
   @Override
   protected void renderBg(GuiGraphics p_282749_, float p_283494_, int p_283098_, int p_282054_) {
      p_282749_.blit(this.menuResource, this.leftPos, this.topPos, 0, 0, this.imageWidth, this.imageHeight);
      this.renderErrorIcon(p_282749_, this.leftPos, this.topPos);
   }

   protected abstract void renderErrorIcon(GuiGraphics p_281990_, int p_266822_, int p_267045_);
// 两个接口的方法
   @Override
   public void dataChanged(AbstractContainerMenu p_169759_, int p_169760_, int p_169761_) {
   }

   @Override
   public void slotChanged(AbstractContainerMenu p_98910_, int p_98911_, ItemStack p_98912_) {
   }
}
```

