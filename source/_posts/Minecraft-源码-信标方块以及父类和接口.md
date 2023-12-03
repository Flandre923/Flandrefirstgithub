---
title: Minecraft-源码-信标方块以及父类和接口
date: 2023-12-02 12:41:56
tags:
- 我的世界源码
- Minecraft
- Java
cover: 

---



# EntityBlock 接口

```java
// 定义了一个EntityBlock接口
// 该接口用了创建和管理方块实体
// 方块实体是一种特殊的方块，她可以存储额外的数据，并且在游戏中进行更新
public interface EntityBlock {
    // 方法用于在给定的位置和方块状态下创建一个新的方块实体对象，可以返回null
   @Nullable
   BlockEntity newBlockEntity(BlockPos p_153215_, BlockState p_153216_);
// 获得一个方块实体的更新器，每个tick对方块实体进行更新
   // 默认方法，默认实现是返回null，T表示必须为BlockEntity或其子类。
   @Nullable
   default <T extends BlockEntity> BlockEntityTicker<T> getTicker(Level p_153212_, BlockState p_153213_, BlockEntityType<T> p_153214_) {
      return null;
   }
// 获得用于方块实体的游戏事件监听器，查看该方块实体是否击沉了GameEvenListener。Holder，如果继承了就返回Listener或者返回null
   @Nullable
   default <T extends BlockEntity> GameEventListener getListener(ServerLevel p_221121_, T p_221122_) {
      return p_221122_ instanceof GameEventListener.Holder holder ? holder.getListener() : null;
   }
}
```





# BaseEntityBlock类

```java
public abstract class BaseEntityBlock extends Block implements EntityBlock {
    //构造方法
   protected BaseEntityBlock(BlockBehaviour.Properties p_49224_) {
      super(p_49224_);
   }
// 获得rendershape
   @Override
   public RenderShape getRenderShape(BlockState p_49232_) {
      return RenderShape.INVISIBLE;
   }
//重写了父类的方法，处理方块的事件，事件id和事件参数
    //调用父类的处理事件方法
    //获得对于位置的blockentity
    //如果非空就调用方块实体的处理事件的方法，并将事件id和事件参数传入
   @Override
   public boolean triggerEvent(BlockState p_49226_, Level p_49227_, BlockPos p_49228_, int p_49229_, int p_49230_) {
      super.triggerEvent(p_49226_, p_49227_, p_49228_, p_49229_, p_49230_);
      BlockEntity blockentity = p_49227_.getBlockEntity(p_49228_);
      return blockentity == null ? false : blockentity.triggerEvent(p_49229_, p_49230_);
   }
//获得方块的menuprovider，获得方块实体如果方块实体继承于menprovider那么就将其转化为menuprovider
   @Nullable
   @Override
   public MenuProvider getMenuProvider(BlockState p_49234_, Level p_49235_, BlockPos p_49236_) {
      BlockEntity blockentity = p_49235_.getBlockEntity(p_49236_);
      return blockentity instanceof MenuProvider ? (MenuProvider)blockentity : null;
   }
// 创建方块实体的ticker，检测方块实体的类型是否相同，如果相同就返回A类型，否则的返回null
   @Nullable
   protected static <E extends BlockEntity, A extends BlockEntity> BlockEntityTicker<A> createTickerHelper(
      BlockEntityType<A> p_152133_, BlockEntityType<E> p_152134_, BlockEntityTicker<? super E> p_152135_
   ) {
      return p_152134_ == p_152133_ ? (BlockEntityTicker<A>)p_152135_ : null;
   }
}
```



# BeaconBeamBlock 接口

```java
public interface BeaconBeamBlock { // 用于返回颜色
   DyeColor getColor();
}
```



# BeaconBlock 类

```java
public class BeaconBlock extends BaseEntityBlock implements BeaconBeamBlock {
    //构造方法
   public BeaconBlock(BlockBehaviour.Properties p_49421_) {
      super(p_49421_);
   }
	// 返回颜色
   @Override
   public DyeColor getColor() {
      return DyeColor.WHITE;
   }
	// 在给定位置创建一个新的方块实体
   @Override
   public BlockEntity newBlockEntity(BlockPos p_152164_, BlockState p_152165_) {
      return new BeaconBlockEntity(p_152164_, p_152165_);
   }
	// 获得一个方块实体的ticker，调用了createtickerhelper方法 返回一个ticker，每个tick对方块尸体进行更新，其中的调用方法是方块实体的tick
   @Nullable
   @Override
   public <T extends BlockEntity> BlockEntityTicker<T> getTicker(Level p_152160_, BlockState p_152161_, BlockEntityType<T> p_152162_) {
      return createTickerHelper(p_152162_, BlockEntityType.BEACON, BeaconBlockEntity::tick);
   }
// 方块使用的处理
   @Override
   public InteractionResult use(BlockState p_49432_, Level p_49433_, BlockPos p_49434_, Player p_49435_, InteractionHand p_49436_, BlockHitResult p_49437_) {
      if (p_49433_.isClientSide) {//客户端直接返回成功
         return InteractionResult.SUCCESS;
      } else {//服务器端
         BlockEntity blockentity = p_49433_.getBlockEntity(p_49434_);//获得对应的方块实体
         if (blockentity instanceof BeaconBlockEntity) {
             //如果是对应的方块实体 则打开menu
            p_49435_.openMenu((BeaconBlockEntity)blockentity);
            p_49435_.awardStat(Stats.INTERACT_WITH_BEACON);
         }

         return InteractionResult.CONSUME;
      }
   }
// 方块实体使用模型渲染。
   @Override
   public RenderShape getRenderShape(BlockState p_49439_) {
      return RenderShape.MODEL;
   }
// 处理方块被放置的事件
   @Override
   public void setPlacedBy(Level p_49426_, BlockPos p_49427_, BlockState p_49428_, LivingEntity p_49429_, ItemStack p_49430_) {
       // 物品是否有自定义的名称
      if (p_49430_.hasCustomHoverName()) {
         BlockEntity blockentity = p_49426_.getBlockEntity(p_49427_);
         if (blockentity instanceof BeaconBlockEntity) {
            ((BeaconBlockEntity)blockentity).setCustomName(p_49430_.getHoverName());//设置自定义名称
         }
      }
   }
}
```



# BeaconBlockEntity 类

```java

public class BeaconBlockEntity extends BlockEntity implements MenuProvider, Nameable {
   private static final int MAX_LEVELS = 4;//最大层数
   public static final MobEffect[][] BEACON_EFFECTS = new MobEffect[][]{
      {MobEffects.MOVEMENT_SPEED, MobEffects.DIG_SPEED}, {MobEffects.DAMAGE_RESISTANCE, MobEffects.JUMP}, {MobEffects.DAMAGE_BOOST}, {MobEffects.REGENERATION}
   };//信标效果
   private static final Set<MobEffect> VALID_EFFECTS = Arrays.stream(BEACON_EFFECTS).flatMap(Arrays::stream).collect(Collectors.toSet());//信标效果的set集合
   public static final int DATA_LEVELS = 0;//索引值
   public static final int DATA_PRIMARY = 1;
   public static final int DATA_SECONDARY = 2;
   public static final int NUM_DATA_VALUES = 3;//data长度是3
   private static final int BLOCKS_CHECK_PER_TICK = 10;//每tick检测的方块数量
   private static final Component DEFAULT_NAME = Component.translatable("container.beacon"); // 名称
   private static final String TAG_PRIMARY = "primary_effect";
   private static final String TAG_SECONDARY = "secondary_effect";
   List<BeaconBlockEntity.BeaconBeamSection> beamSections = Lists.newArrayList();//信标光束部分的列表，颜色和高度？
   private List<BeaconBlockEntity.BeaconBeamSection> checkingBeamSections = Lists.newArrayList();//颜色和高度？
   int levels; // 层数
   private int lastCheckY;// 上次检测的y坐标位置
   @Nullable
   MobEffect primaryPower;// 主要效果
   @Nullable
   MobEffect secondaryPower;// 次要效果
   @Nullable
   private Component name;// 名字
   private LockCode lockKey = LockCode.NO_LOCK; // 信标没有锁码
    // 信标数据的访问，该类用于传递方块实体的数据
    // get set getCount方法分别是获得，设置，和获得数据元素个数
   private final ContainerData dataAccess = new ContainerData() {
      @Override
      public int get(int p_58711_) {
         return switch(p_58711_) {
            case 0 -> BeaconBlockEntity.this.levels;
            case 1 -> BeaconMenu.encodeEffect(BeaconBlockEntity.this.primaryPower);
            case 2 -> BeaconMenu.encodeEffect(BeaconBlockEntity.this.secondaryPower);
            default -> 0;
         };
      }

      @Override
      public void set(int p_58713_, int p_58714_) {
         switch(p_58713_) {
            case 0:
               BeaconBlockEntity.this.levels = p_58714_;
               break;
            case 1:
               if (!BeaconBlockEntity.this.level.isClientSide && !BeaconBlockEntity.this.beamSections.isEmpty()) {
                  BeaconBlockEntity.playSound(BeaconBlockEntity.this.level, BeaconBlockEntity.this.worldPosition, SoundEvents.BEACON_POWER_SELECT);
               }

               BeaconBlockEntity.this.primaryPower = BeaconBlockEntity.filterEffect(BeaconMenu.decodeEffect(p_58714_));
               break;
            case 2:
               BeaconBlockEntity.this.secondaryPower = BeaconBlockEntity.filterEffect(BeaconMenu.decodeEffect(p_58714_));
         }
      }

      @Override
      public int getCount() {
         return 3;
      }
   };
//检查modeeffect是否合法
   @Nullable
   static MobEffect filterEffect(@Nullable MobEffect p_298440_) {
      return VALID_EFFECTS.contains(p_298440_) ? p_298440_ : null;
   }
// 构造方法
   public BeaconBlockEntity(BlockPos p_155088_, BlockState p_155089_) {
      super(BlockEntityType.BEACON, p_155088_, p_155089_);
   }

   public static void tick(Level p_155108_, BlockPos p_155109_, BlockState p_155110_, BeaconBlockEntity p_155111_) {
       // 获得方块的xyz坐标
      int i = p_155109_.getX();
      int j = p_155109_.getY();
      int k = p_155109_.getZ();
      BlockPos blockpos;
      if (p_155111_.lastCheckY < j) {// 上次检查y小于j
         blockpos = p_155109_;
         p_155111_.checkingBeamSections = Lists.newArrayList();
         p_155111_.lastCheckY = p_155109_.getY() - 1;//更新上次检查的y
      } else {// 上次检查的y大于j
          //方块位置设置为当前xz上，上次检查y+1的位置
         blockpos = new BlockPos(i, p_155111_.lastCheckY + 1, k);
      }
// 如果信标方块实体的检查光束段列表为空，则赋值为null，否则赋值为列表中的最后一个元素，即上一个光束段对象。
      BeaconBlockEntity.BeaconBeamSection beaconblockentity$beaconbeamsection = p_155111_.checkingBeamSections.isEmpty()
         ? null
         : p_155111_.checkingBeamSections.get(p_155111_.checkingBeamSections.size() - 1);
       //信标方块实体所在的世界的表面高度
       //WORLD_SURFACE表示世界表面的高度
       // x，z坐标
      int l = p_155108_.getHeight(Heightmap.Types.WORLD_SURFACE, i, k);
// 表示每个游戏刻最多检查10个方块,且不超过世界表面的高度
      for(int i1 = 0; i1 < 10 && blockpos.getY() <= l; ++i1) {
         BlockState blockstate = p_155108_.getBlockState(blockpos);
         Block block = blockstate.getBlock();
          //这个方法返回一个长度为3的浮点数数组，表示方块的光束颜色的RGB值的乘数，
         float[] afloat = blockstate.getBeaconColorMultiplier(p_155108_, blockpos, p_155109_);
         if (afloat != null) {
             //它判断信标方块实体的检查光束段列表的大小是否小于等于1，如果是，则表示这是第一个或第二个影响光束颜色的方块
            if (p_155111_.checkingBeamSections.size() <= 1) {
               beaconblockentity$beaconbeamsection = new BeaconBlockEntity.BeaconBeamSection(afloat);
               p_155111_.checkingBeamSections.add(beaconblockentity$beaconbeamsection);
            } else if (beaconblockentity$beaconbeamsection != null) {//表示已经有至少两个光束段
               if (Arrays.equals(afloat, beaconblockentity$beaconbeamsection.color)) {//判断afloat是否与beaconblockentity$beaconbeamsection变量的颜色相同,如果是，则表示方块的颜色与上一个光束段的颜色相同
                  beaconblockentity$beaconbeamsection.increaseHeight();
               } else {//表示方块的颜色与上一个光束段的颜色不同，
                  beaconblockentity$beaconbeamsection = new BeaconBlockEntity.BeaconBeamSection(
                     new float[]{
                        (beaconblockentity$beaconbeamsection.color[0] + afloat[0]) / 2.0F,
                        (beaconblockentity$beaconbeamsection.color[1] + afloat[1]) / 2.0F,
                        (beaconblockentity$beaconbeamsection.color[2] + afloat[2]) / 2.0F
                     }
                  );
                  p_155111_.checkingBeamSections.add(beaconblockentity$beaconbeamsection);
               }
            }
         } else {//表示afloat为null
             //判断beaconblockentity$beaconbeamsection变量是否为null
             //者方块的光照等级是否大于等于15
             //方块是否是基
             //如果是，则表示光束被阻断，执行以下语句：
            if (beaconblockentity$beaconbeamsection == null || blockstate.getLightBlock(p_155108_, blockpos) >= 15 && !blockstate.is(Blocks.BEDROCK)) {
                //清空信标方块实体的检查光束段列表，表示清除光束段信息。
               p_155111_.checkingBeamSections.clear();
               p_155111_.lastCheckY = l;
                //跳出循环，表示结束检查。
               break;
            }
             //否则，表示光束没有被阻断，将其高度加一，表示延长光束段的高度。

            beaconblockentity$beaconbeamsection.increaseHeight();
         }
//将blockpos变量赋值为它的上方的位置，表示向上移动一个方块。
         blockpos = blockpos.above();
          //将信标方块实体的最后检查的y坐标加一，表示更新检查的位置。
         ++p_155111_.lastCheckY;
      }

      int j1 = p_155111_.levels;
      if (p_155108_.getGameTime() % 80L == 0L) {
         if (!p_155111_.beamSections.isEmpty()) {
             //表示信标方块的层数，然后将这个值赋值给信标方块实体的levels变量，表示更新信标方块的层数。
            p_155111_.levels = updateBase(p_155108_, i, j, k);
         }

         if (p_155111_.levels > 0 && !p_155111_.beamSections.isEmpty()) {
             
             //用于给信标方块附近的玩家施加相应的状态效果。
            applyEffects(p_155108_, p_155109_, p_155111_.levels, p_155111_.primaryPower, p_155111_.secondaryPower);
             //它表示信标方块的环境声音。
            playSound(p_155108_, p_155109_, SoundEvents.BEACON_AMBIENT);
         }
      }
//最后检查的y坐标是否大于等于世界表面的高度
       //如果是，则表示光束检查已经结束
      if (p_155111_.lastCheckY >= l) {
          //将信标方块实体的最后检查的y坐标赋值为世界的最低建造高度减一，表示重置检查的位置。
         p_155111_.lastCheckY = p_155108_.getMinBuildHeight() - 1;
          //它表示信标方块实体的层数的上一次的值是否大于0，它的值是通过将j1与0比较得到的。
         boolean flag = j1 > 0;
          //将信标方块实体的光束段列表赋值为信标方块实体的检查光束段列表，表示更新光束段信息。
         p_155111_.beamSections = p_155111_.checkingBeamSections;
          //判断信标方块实体所在的世界是否不是客户端世界，如果是，则表示这是服务器端的逻辑
         if (!p_155108_.isClientSide) {
             //信标方块实体的层数是否大于0
            boolean flag1 = p_155111_.levels > 0;
             //判断flag和flag1是否不同，如果是，则表示信标方块实体的层数发生了变化
            if (!flag && flag1) {
                //用于播放信标方块的激活声音。
               playSound(p_155108_, p_155109_, SoundEvents.BEACON_ACTIVATE);
//对信标方块实体所在的世界中的所有符合条件的玩家进行循环，条件是玩家的类别是ServerPlayer类，即服务器端的玩家，以及玩家的位置在信标方块实体的位置的周围的一个长方体区域内，这个区域的大小是20×10×20，
               for(ServerPlayer serverplayer : p_155108_.getEntitiesOfClass(
                  ServerPlayer.class, new AABB((double)i, (double)j, (double)k, (double)i, (double)(j - 4), (double)k).inflate(10.0, 5.0, 10.0)
               )) {
                   //用于触发玩家的建造信标的进度。
                  CriteriaTriggers.CONSTRUCT_BEACON.trigger(serverplayer, p_155111_.levels);
               }
            } else if (flag && !flag1) {
                //它表示信标方块的停用声音。
               playSound(p_155108_, p_155109_, SoundEvents.BEACON_DEACTIVATE);
            }
         }
      }
   }

   private static int updateBase(Level p_155093_, int p_155094_, int p_155095_, int p_155096_) {
      int i = 0;//它表示信标方块的层数，它的初始值是0，表示没有检查通过的层。
//它定义了一个整数类型的局部变量j，它表示当前检查的层数，它的初始值是1，每次循环后将j的值赋给i，并将j自增1，循环的条件是j小于等于4，表示最多检查四层
      for(int j = 1; j <= 4; i = j++) {
         int k = p_155095_ - j;//它表示当前检查的层的y坐标，它的值是信标方块的y坐标减去当前检查的层数，表示从信标方块的下方开始检查
         if (k < p_155093_.getMinBuildHeight()) {//它判断当前检查的层的y坐标是否小于信标方块所在的世界的最低建造高度
            break;
         }
///它表示当前检查的层是否由信标基座方块组成，它的初始值是true
         boolean flag = true;
//它表示当前检查的方块的x坐标，它的初始值是信标方块的x坐标减去当前检查的层数，每次循环后自增1，循环的条件是l小于等于信标方块的x坐标加上当前检查的层数，并且flag为true，表示在当前检查的层的范围内，并且没有发现非信标基座方块
         for(int l = p_155094_ - j; l <= p_155094_ + j && flag; ++l) {
             //它表示当前检查的方块的z坐标，它的初始值是信标方块的z坐标减去当前检查的层数，每次循环后自增1，循环的条件是i1小于等于信标方块的z坐标加上当前检查的层数，表示在当前检查的层的范围内
            for(int i1 = p_155096_ - j; i1 <= p_155096_ + j; ++i1) {//它判断当前检查的方块的状态是否不是信标基座方块，如果不是
               if (!p_155093_.getBlockState(new BlockPos(l, k, i1)).is(BlockTags.BEACON_BASE_BLOCKS)) {
                  flag = false;
                  break;
               }
            }
         }

         if (!flag) {
            break;
         }
      }

      return i;
   }
//实体移除
   @Override
   public void setRemoved() {
      playSound(this.level, this.worldPosition, SoundEvents.BEACON_DEACTIVATE);
      super.setRemoved();
   }

   private static void applyEffects(Level p_155098_, BlockPos p_155099_, int p_155100_, @Nullable MobEffect p_155101_, @Nullable MobEffect p_155102_) {
       //判断信标方块所在的世界是否不是客户端世界，并且信标方块的主要效果是否不为
      if (!p_155098_.isClientSide && p_155101_ != null) {
          //它表示信标方块的效果范围，它的值是信标方块的层数乘以10再加上10，表示信标方块的效果范围随层数增加而增加。
         double d0 = (double)(p_155100_ * 10 + 10);
         int i = 0;//它表示信标方块的效果等级，它的初始值是0，表示默认为0级。
          //信标方块的层数是否大于等于4，并且信标方块的主要效果和次要效果相同
         if (p_155100_ >= 4 && p_155101_ == p_155102_) {
            i = 1;//将i变量赋值为1，表示信标方块的效果等级为1级，表示效果更强。
         }
//定义一个整数类型的局部变量j，它表示信标方块的效果持续时间，它的值是9加上信标方块的层数乘以2，再乘以20，表示信标方块的效果持续时间随层数增加而增加
         int j = (9 + p_155100_ * 2) * 20;
          //得到一个以信标方块为中心的立方体区域，然后调用这个区域的inflate方法，将d0作为参数传递，得到一个扩大d0的区域，再调用这个区域的expandTowards方法，将0.0，信标方块所在的世界的高度和0.0作为参数传递，得到一个向上和向下扩展的区域，表示信标方块的效果区域。AABB类表示一个轴对齐的边界框，它用于表示一个三维的区域。
         AABB aabb = new AABB(p_155099_).inflate(d0).expandTowards(0.0, (double)p_155098_.getHeight(), 0.0);
          //它表示信标方块的效果区域内的所有玩家，
         List<Player> list = p_155098_.getEntitiesOfClass(Player.class, aabb);

         for(Player player : list) {
            player.addEffect(new MobEffectInstance(p_155101_, j, i, true, true));
         }
//判断信标方块的层数是否大于等于4，并且信标方块的主要效果和次要效果不同，次要效果不为空，
         if (p_155100_ >= 4 && p_155101_ != p_155102_ && p_155102_ != null) {
            for(Player player1 : list) {
               player1.addEffect(new MobEffectInstance(p_155102_, j, 0, true, true));
            }
         }
      }
   }

   public static void playSound(Level p_155104_, BlockPos p_155105_, SoundEvent p_155106_) {
      p_155104_.playSound(null, p_155105_, p_155106_, SoundSource.BLOCKS, 1.0F, 1.0F);
   }
//获取信标方块实体的光束段列表，
   public List<BeaconBlockEntity.BeaconBeamSection> getBeamSections() {
      return (List<BeaconBlockEntity.BeaconBeamSection>)(this.levels == 0 ? ImmutableList.of() : this.beamSections);
   }
///用于获取信标方块实体的更新数据包
    //这个类型表示一个从服务器端发送到客户端的方块实体数据包，用于同步方块实体的数据。
   public ClientboundBlockEntityDataPacket getUpdatePacket() {
      return ClientboundBlockEntityDataPacket.create(this);
   }
//用于在方块实体被放置或移除时，发送方块实体的数据。
   @Override
   public CompoundTag getUpdateTag() {
      return this.saveWithoutMetadata();
   }

   private static void storeEffect(CompoundTag p_298214_, String p_298983_, @Nullable MobEffect p_299071_) {
      if (p_299071_ != null) {
         ResourceLocation resourcelocation = BuiltInRegistries.MOB_EFFECT.getKey(p_299071_);
         if (resourcelocation != null) {
            p_298214_.putString(p_298983_, resourcelocation.toString());
         }
      }
   }

   @Nullable
   private static MobEffect loadEffect(CompoundTag p_298570_, String p_299310_) {
      if (p_298570_.contains(p_299310_, 8)) {
         ResourceLocation resourcelocation = ResourceLocation.tryParse(p_298570_.getString(p_299310_));
         return filterEffect(BuiltInRegistries.MOB_EFFECT.get(resourcelocation));
      } else {
         return null;
      }
   }

   @Override
   public void load(CompoundTag p_155113_) {
      super.load(p_155113_);
      this.primaryPower = loadEffect(p_155113_, "primary_effect");
      this.secondaryPower = loadEffect(p_155113_, "secondary_effect");
      if (p_155113_.contains("CustomName", 8)) {
         this.name = Component.Serializer.fromJson(p_155113_.getString("CustomName"));
      }

      this.lockKey = LockCode.fromTag(p_155113_);
   }

   @Override
   protected void saveAdditional(CompoundTag p_187463_) {
      super.saveAdditional(p_187463_);
      storeEffect(p_187463_, "primary_effect", this.primaryPower);
      storeEffect(p_187463_, "secondary_effect", this.secondaryPower);
      p_187463_.putInt("Levels", this.levels);
      if (this.name != null) {
         p_187463_.putString("CustomName", Component.Serializer.toJson(this.name));
      }

      this.lockKey.addToTag(p_187463_);
   }

   public void setCustomName(@Nullable Component p_58682_) {
      this.name = p_58682_;
   }

   @Nullable
   @Override
   public Component getCustomName() {
      return this.name;
   }

   @Nullable
   @Override
   public AbstractContainerMenu createMenu(int p_58696_, Inventory p_58697_, Player p_58698_) {
       //判断玩家是否能解锁信标方块实体的菜单
       //如果是
       //创建一个新的BeaconMenu对象，它表示一个信标方块实体的菜单
       //否则，返回null，表示创建失败。
      return BaseContainerBlockEntity.canUnlock(p_58698_, this.lockKey, this.getDisplayName())
         ? new BeaconMenu(p_58696_, p_58697_, this.dataAccess, ContainerLevelAccess.create(this.level, this.getBlockPos()))
         : null;
   }

   @Override
   public Component getDisplayName() {
      return this.getName();
   }

   @Override
   public Component getName() {
      return this.name != null ? this.name : DEFAULT_NAME;
   }

   @Override
   public void setLevel(Level p_155091_) {
      super.setLevel(p_155091_);
      this.lastCheckY = p_155091_.getMinBuildHeight() - 1;
   }

   public static class BeaconBeamSection {
      final float[] color;
      private int height;

      public BeaconBeamSection(float[] p_58718_) {
         this.color = p_58718_;
         this.height = 1;
      }

      protected void increaseHeight() {
         ++this.height;
      }

      public float[] getColor() {
         return this.color;
      }

      public int getHeight() {
         return this.height;
      }
   }
}
```



# BeaconRenderer 类

```java
@OnlyIn(Dist.CLIENT)

//类负责在游戏中渲染信标光束。 实现了BlockEntityRenderer接口
public class BeaconRenderer implements BlockEntityRenderer<BeaconBlockEntity> {
   public static final ResourceLocation BEAM_LOCATION = new ResourceLocation("textures/entity/beacon_beam.png");
   public static final int MAX_RENDER_Y = 1024;

   public BeaconRenderer(BlockEntityRendererProvider.Context p_173529_) {
   }
//调用render（）方法来渲染每个刻度的信标
    //它从Beacon BlockEntity获取波束部分，并在其中循环。
   public void render(BeaconBlockEntity p_112140_, float p_112141_, PoseStack p_112142_, MultiBufferSource p_112143_, int p_112144_, int p_112145_) {
      long i = p_112140_.getLevel().getGameTime();
      List<BeaconBlockEntity.BeaconBeamSection> list = p_112140_.getBeamSections();
      int j = 0;

      for(int k = 0; k < list.size(); ++k) {
         BeaconBlockEntity.BeaconBeamSection beaconblockentity$beaconbeamsection = list.get(k);
          //对于每个部分，它都会调用renderBeacon beam（）来渲染BeaconBeam的该部分
          //它传递PoseStack、缓冲区、动画时间/数据和颜色。
         renderBeaconBeam(
            p_112142_,
            p_112143_,
            p_112141_,
            i,
            j,
            k == list.size() - 1 ? 1024 : beaconblockentity$beaconbeamsection.getHeight(),
            beaconblockentity$beaconbeamsection.getColor()
         );
          //高度计数器j按截面的高度递增
         j += beaconblockentity$beaconbeamsection.getHeight();
      }
   }
//renderBeachBeam（）执行BeaconBeam渲染逻辑。
   private static void renderBeaconBeam(
      PoseStack p_112177_, MultiBufferSource p_112178_, float p_112179_, long p_112180_, int p_112181_, int p_112182_, float[] p_112183_
   ) {
      renderBeaconBeam(p_112177_, p_112178_, BEAM_LOCATION, p_112179_, 1.0F, p_112180_, p_112181_, p_112182_, p_112183_, 0.2F, 0.25F);
   }
//PoseStack、缓冲区、动画时间/值、颜色等参数来渲染光束。
   public static void renderBeaconBeam(
      PoseStack p_112185_,
      MultiBufferSource p_112186_,
      ResourceLocation p_112187_,
      float p_112188_,
      float p_112189_,
      long p_112190_,
      int p_112191_,
      int p_112192_,
      float[] p_112193_,
      float p_112194_,
      float p_112195_
   ) {
     //根据起点和终点指数计算梁截面的总高度（i）
      int i = p_112191_ + p_112192_;
       //变换矩阵保存到PoseStack
      p_112185_.pushPose();
       //将原点移动到梁的中心。
      p_112185_.translate(0.5, 0.0, 0.5);
       //基于游戏时间来计算动画值（f）。
      float f = (float)Math.floorMod(p_112190_, 40) + p_112188_;
       //其他动画值（如f1、f2）就是从中导出的。
      float f1 = p_112192_ < 0 ? f : -f;
      float f2 = Mth.frac(f1 * 0.2F - (float)Mth.floor(f1 * 0.1F));
       //颜色阵列中提取颜色分量值f3、f4、f5。
      float f3 = p_112193_[0];
      float f4 = p_112193_[1];
      float f5 = p_112193_[2];
       //pushPose（）在围绕Y轴旋转梁之前再次保存矩阵。
      p_112185_.pushPose();
      p_112185_.mulPose(Axis.YP.rotationDegrees(f * 2.25F - 45.0F));
       //其余的浮动用于设置动画曲线和位置的变量。
      float f6 = 0.0F;
      float f8 = 0.0F;
      float f9 = -p_112194_;
      float f10 = 0.0F;
      float f11 = 0.0F;
      float f12 = -p_112194_;
      float f13 = 0.0F;
      float f14 = 1.0F;
      float f15 = -1.0F + f2;
       //f16计算一个将沿光束长度设置动画的值。
      float f16 = (float)p_112192_ * p_112189_ * (0.5F / p_112194_) + f15;
       //对于梁四边形的每个“边”调用一次。
       //通过添加具有变换位置/颜色的顶点来渲染梁的单个四边形。它使用Matrix和VertexConsumer API绘制到缓冲区中。
      renderPart(
         p_112185_,
         p_112186_.getBuffer(RenderType.beaconBeam(p_112187_, false)),
         f3,
         f4,
         f5,
         1.0F,
         p_112191_,
         i,
         0.0F,
         p_112194_,
         p_112194_,
         0.0F,
         f9,
         0.0F,
         0.0F,
         f12,
         0.0F,
         1.0F,
         f16,
         f15
      );
      p_112185_.popPose();
      f6 = -p_112195_;
      float f7 = -p_112195_;
      f8 = -p_112195_;
      f9 = -p_112195_;
      f13 = 0.0F;
      f14 = 1.0F;
      f15 = -1.0F + f2;
      f16 = (float)p_112192_ * p_112189_ + f15;
       //
      renderPart(
         p_112185_,
         p_112186_.getBuffer(RenderType.beaconBeam(p_112187_, true)),
         f3,
         f4,
         f5,
         0.125F,
         p_112191_,
         i,
         f6,
         f7,
         p_112195_,
         f8,
         f9,
         p_112195_,
         p_112195_,
         p_112195_,
         0.0F,
         1.0F,
         f16,
         f15
      );
      p_112185_.popPose();
   }

   private static void renderPart(
      PoseStack p_112156_,
      VertexConsumer p_112157_,
      float p_112158_,
      float p_112159_,
      float p_112160_,
      float p_112161_,
      int p_112162_,
      int p_112163_,
      float p_112164_,
      float p_112165_,
      float p_112166_,
      float p_112167_,
      float p_112168_,
      float p_112169_,
      float p_112170_,
      float p_112171_,
      float p_112172_,
      float p_112173_,
      float p_112174_,
      float p_112175_
   ) {
      PoseStack.Pose posestack$pose = p_112156_.last();
      Matrix4f matrix4f = posestack$pose.pose();
      Matrix3f matrix3f = posestack$pose.normal();
      renderQuad(
         matrix4f,
         matrix3f,
         p_112157_,
         p_112158_,
         p_112159_,
         p_112160_,
         p_112161_,
         p_112162_,
         p_112163_,
         p_112164_,
         p_112165_,
         p_112166_,
         p_112167_,
         p_112172_,
         p_112173_,
         p_112174_,
         p_112175_
      );
      renderQuad(
         matrix4f,
         matrix3f,
         p_112157_,
         p_112158_,
         p_112159_,
         p_112160_,
         p_112161_,
         p_112162_,
         p_112163_,
         p_112170_,
         p_112171_,
         p_112168_,
         p_112169_,
         p_112172_,
         p_112173_,
         p_112174_,
         p_112175_
      );
      renderQuad(
         matrix4f,
         matrix3f,
         p_112157_,
         p_112158_,
         p_112159_,
         p_112160_,
         p_112161_,
         p_112162_,
         p_112163_,
         p_112166_,
         p_112167_,
         p_112170_,
         p_112171_,
         p_112172_,
         p_112173_,
         p_112174_,
         p_112175_
      );
      renderQuad(
         matrix4f,
         matrix3f,
         p_112157_,
         p_112158_,
         p_112159_,
         p_112160_,
         p_112161_,
         p_112162_,
         p_112163_,
         p_112168_,
         p_112169_,
         p_112164_,
         p_112165_,
         p_112172_,
         p_112173_,
         p_112174_,
         p_112175_
      );
   }

   private static void renderQuad(
      Matrix4f p_253960_,
      Matrix3f p_254005_,
      VertexConsumer p_112122_,
      float p_112123_,
      float p_112124_,
      float p_112125_,
      float p_112126_,
      int p_112127_,
      int p_112128_,
      float p_112129_,
      float p_112130_,
      float p_112131_,
      float p_112132_,
      float p_112133_,
      float p_112134_,
      float p_112135_,
      float p_112136_
   ) {
      addVertex(p_253960_, p_254005_, p_112122_, p_112123_, p_112124_, p_112125_, p_112126_, p_112128_, p_112129_, p_112130_, p_112134_, p_112135_);
      addVertex(p_253960_, p_254005_, p_112122_, p_112123_, p_112124_, p_112125_, p_112126_, p_112127_, p_112129_, p_112130_, p_112134_, p_112136_);
      addVertex(p_253960_, p_254005_, p_112122_, p_112123_, p_112124_, p_112125_, p_112126_, p_112127_, p_112131_, p_112132_, p_112133_, p_112136_);
      addVertex(p_253960_, p_254005_, p_112122_, p_112123_, p_112124_, p_112125_, p_112126_, p_112128_, p_112131_, p_112132_, p_112133_, p_112135_);
   }
//私有实用程序方法实际上处理用所有数据写入顶点的问题。
   private static void addVertex(
      Matrix4f p_253955_,
      Matrix3f p_253713_,
      VertexConsumer p_253894_,
      float p_253871_,
      float p_253841_,
      float p_254568_,
      float p_254361_,
      int p_254357_,
      float p_254451_,
      float p_254240_,
      float p_254117_,
      float p_253698_
   ) {
      p_253894_.vertex(p_253955_, p_254451_, (float)p_254357_, p_254240_)
         .color(p_253871_, p_253841_, p_254568_, p_254361_)
         .uv(p_254117_, p_253698_)
         .overlayCoords(OverlayTexture.NO_OVERLAY)
         .uv2(15728880)
         .normal(p_253713_, 0.0F, 1.0F, 0.0F)
         .endVertex();
   }

   public boolean shouldRenderOffScreen(BeaconBlockEntity p_112138_) {
      return true;
   }

   @Override
   public int getViewDistance() {
      return 256;
   }

   public boolean shouldRender(BeaconBlockEntity p_173531_, Vec3 p_173532_) {
      return Vec3.atCenterOf(p_173531_.getBlockPos()).multiply(1.0, 0.0, 1.0).closerThan(p_173532_.multiply(1.0, 0.0, 1.0), (double)this.getViewDistance());
   }
}
```
