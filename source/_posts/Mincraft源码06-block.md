---
title: Mincraft源码06-block
date: 2023-11-23 21:54:42
tags:
- 我的世界
- 源码
- Java
cover: https://view.moezx.cc/images/2018/01/17/violetevergarden66768635_by_DDD.png
---



# Block

```java

public class Block extends BlockBehaviour implements ItemLike, net.neoforged.neoforge.common.extensions.IBlockExtension {
   private static final Logger LOGGER = LogUtils.getLogger();
   private final Holder.Reference<Block> builtInRegistryHolder = BuiltInRegistries.BLOCK.createIntrusiveHolder(this);
    // 声明一个弃用的 BlockState 映射注册表,注释中说明改用 GameRegistry
   @Deprecated //Forge: Do not use, use GameRegistry
   public static final IdMapper<BlockState> BLOCK_STATE_REGISTRY = net.neoforged.neoforge.registries.GameData.getBlockStateIDMap();
    // 创建一个缓存,通过一个 VoxelShape 快速判断它是否是一个完整的方块
   private static final LoadingCache<VoxelShape, Boolean> SHAPE_FULL_BLOCK_CACHE = CacheBuilder.newBuilder()
      .maximumSize(512L)
      .weakKeys()
      .build(new CacheLoader<VoxelShape, Boolean>() {
         public Boolean load(VoxelShape p_49972_) {
            return !Shapes.joinIsNotEmpty(Shapes.block(), p_49972_, BooleanOp.NOT_SAME);
         }
      });
    // 邻近方块更新标记
   public static final int UPDATE_NEIGHBORS = 1;
    // 通知客户端方块改变标记 
   public static final int UPDATE_CLIENTS = 2;
    //  不可见的方块改变标记
   public static final int UPDATE_INVISIBLE = 4;
    // 立即更新标记
   public static final int UPDATE_IMMEDIATE = 8;
    // 更新时使用已知的方块形状标记
   public static final int UPDATE_KNOWN_SHAPE = 16;
   public static final int UPDATE_SUPPRESS_DROPS = 32;
   public static final int UPDATE_MOVE_BY_PISTON = 64;
   public static final int UPDATE_NONE = 4;
   public static final int UPDATE_ALL = 3;
   public static final int UPDATE_ALL_IMMEDIATE = 11;
   public static final float INDESTRUCTIBLE = -1.0F;
   public static final float INSTANT = 0.0F;
   public static final int UPDATE_LIMIT = 512;
    //// 定义Block状态的状态定义,用于映射Block到其状态
   protected final StateDefinition<Block, BlockState> stateDefinition;
    // Block的默认状态
   private BlockState defaultBlockState;
    // Block的描述ID,可为空
   @Nullable
   private String descriptionId;
    // 与这个Block相关的Item,可为空 
   @Nullable
   private Item item;
    // 用于遮挡(光照)计算的缓存大小
   private static final int CACHE_SIZE = 2048;
    // 线程本地的OpenHashMap作为遮挡缓存,默认返回值127代表不透明
   private static final ThreadLocal<Object2ByteLinkedOpenHashMap<Block.BlockStatePairKey>> OCCLUSION_CACHE = ThreadLocal.withInitial(
      () -> {
         Object2ByteLinkedOpenHashMap<Block.BlockStatePairKey> object2bytelinkedopenhashmap = new Object2ByteLinkedOpenHashMap<Block.BlockStatePairKey>(
            2048, 0.25F
         ) {
            @Override
            protected void rehash(int p_49979_) {
            }
         };
         object2bytelinkedopenhashmap.defaultReturnValue((byte)127);
         return object2bytelinkedopenhashmap;
      }
   );
// 获取BlockState的id,如果为null则返回0  
   public static int getId(@Nullable BlockState p_49957_) {
      if (p_49957_ == null) {
         return 0;
      } else {
         int i = BLOCK_STATE_REGISTRY.getId(p_49957_);
         return i == -1 ? 0 : i;
      }
   }
// 通过id获取BlockState,如果不存在则返回AIR的默认状态
   public static BlockState stateById(int p_49804_) {
      BlockState blockstate = BLOCK_STATE_REGISTRY.byId(p_49804_);
      return blockstate == null ? Blocks.AIR.defaultBlockState() : blockstate;
   }
// 通过Item获取其相关的Block,如果不是BlockItem则返回AIR
   public static Block byItem(@Nullable Item p_49815_) {
      return p_49815_ instanceof BlockItem ? ((BlockItem)p_49815_).getBlock() : Blocks.AIR;
   }
// 当一个BlockState被另一个取代时,移动实体到上方避免卡住
   public static BlockState pushEntitiesUp(BlockState p_49898_, BlockState p_49899_, LevelAccessor p_238252_, BlockPos p_49901_) {
        // 计算两个状态碰撞体的形状
      VoxelShape voxelshape = Shapes.joinUnoptimized(
            p_49898_.getCollisionShape(p_238252_, p_49901_), p_49899_.getCollisionShape(p_238252_, p_49901_), BooleanOp.ONLY_SECOND
         )
         .move((double)p_49901_.getX(), (double)p_49901_.getY(), (double)p_49901_.getZ());
      if (voxelshape.isEmpty()) {
         return p_49899_;
      } else {
               // 将实体向上移动以脱离碰撞体
         for(Entity entity : p_238252_.getEntities(null, voxelshape.bounds())) {
            double d0 = Shapes.collide(Direction.Axis.Y, entity.getBoundingBox().move(0.0, 1.0, 0.0), List.of(voxelshape), -1.0);
            entity.teleportRelative(0.0, 1.0 + d0, 0.0);
         }

         return p_49899_;
      }
   }
//创建一个表示方块碰撞箱的VoxelShape,输入是原始像素大小,会自动缩放到1/16大小
   public static VoxelShape box(double p_49797_, double p_49798_, double p_49799_, double p_49800_, double p_49801_, double p_49802_) {
      return Shapes.box(p_49797_ / 16.0, p_49798_ / 16.0, p_49799_ / 16.0, p_49800_ / 16.0, p_49801_ / 16.0, p_49802_ / 16.0);
   }
//// 根据相邻方块更新方块状态
   public static BlockState updateFromNeighbourShapes(BlockState p_49932_, LevelAccessor p_49933_, BlockPos p_49934_) {
      BlockState blockstate = p_49932_;
      BlockPos.MutableBlockPos blockpos$mutableblockpos = new BlockPos.MutableBlockPos();
 // 按顺序检查相邻6个面 
      for(Direction direction : UPDATE_SHAPE_ORDER) {
         blockpos$mutableblockpos.setWithOffset(p_49934_, direction);
         blockstate = blockstate.updateShape(direction, p_49933_.getBlockState(blockpos$mutableblockpos), p_49933_, p_49934_, blockpos$mutableblockpos);
      }

      return blockstate;
   }
// 更新方块或者销毁
   public static void updateOrDestroy(BlockState p_49903_, BlockState p_49904_, LevelAccessor p_49905_, BlockPos p_49906_, int p_49907_) {
      updateOrDestroy(p_49903_, p_49904_, p_49905_, p_49906_, p_49907_, 512);
   }
// 更新或销毁方块
   public static void updateOrDestroy(BlockState p_49909_, BlockState p_49910_, LevelAccessor p_49911_, BlockPos p_49912_, int p_49913_, int p_49914_) {
      if (p_49910_ != p_49909_) {
         if (p_49910_.isAir()) {
            if (!p_49911_.isClientSide()) {
               p_49911_.destroyBlock(p_49912_, (p_49913_ & 32) == 0, null, p_49914_);
            }
         } else {
            p_49911_.setBlock(p_49912_, p_49910_, p_49913_ & -33, p_49914_);
         }
      }
   }

   public Block(BlockBehaviour.Properties p_49795_) {
    // 调用父类(BlockBehaviour)的构造函数
      super(p_49795_);
       // 创建Block状态定义的构建器
      StateDefinition.Builder<Block, BlockState> builder = new StateDefinition.Builder<>(this);
       // 子类实现,定义此Block的所有状态
      this.createBlockStateDefinition(builder);
       // 使用构建器创建状态定义
      this.stateDefinition = builder.create(Block::defaultBlockState, BlockState::new);
           // 注册默认状态
      this.registerDefaultState(this.stateDefinition.any());
       // 开发环境检查,Block类名应以Block结尾
      if (SharedConstants.IS_RUNNING_IN_IDE) {
         String s = this.getClass().getSimpleName();
         if (!s.endsWith("Block")) {
            LOGGER.error("Block classes should end with Block and {} doesn't.", s);
         }
      }
           // 客户端初始化
      initClient();
   }
// 判断一个BlockState是否是用于连接检查的例外
   public static boolean isExceptionForConnection(BlockState p_152464_) {
           // 如果是树叶Block、障碍物、南瓜、瓜块、潜影盒等返回true

      return p_152464_.getBlock() instanceof LeavesBlock
         || p_152464_.is(Blocks.BARRIER)
         || p_152464_.is(Blocks.CARVED_PUMPKIN)
         || p_152464_.is(Blocks.JACK_O_LANTERN)
         || p_152464_.is(Blocks.MELON)
         || p_152464_.is(Blocks.PUMPKIN)
         || p_152464_.is(BlockTags.SHULKER_BOXES);
   }
//// 直接返回是否随机tick的属性
   public boolean isRandomlyTicking(BlockState p_49921_) {
      return this.isRandomlyTicking;
   }
//该方法用于判断是否应该渲染方块的某个面。
   public static boolean shouldRenderFace(BlockState p_152445_, BlockGetter p_152446_, BlockPos p_152447_, Direction p_152448_, BlockPos p_152449_) {
       // 获取指定位置的方块状态
      BlockState blockstate = p_152446_.getBlockState(p_152449_);
       // 如果方块状态指示不需要渲染该面，则返回false
      if (p_152445_.skipRendering(blockstate, p_152448_)) {
         return false;
      } else if (blockstate.hidesNeighborFace(p_152446_, p_152449_, p_152445_, p_152448_.getOpposite()) && p_152445_.supportsExternalFaceHiding()) {//如果相邻方块的指定面被当前方块状态隐藏，并且当前方块状态支持外部面隐藏，则返回false

         return false;
      } else if (blockstate.canOcclude()) {// 如果相邻方块的指定面可以遮挡，则进行进一步判断
// 创建一个方块状态对的键
         Block.BlockStatePairKey block$blockstatepairkey = new Block.BlockStatePairKey(p_152445_, blockstate, p_152448_);
        // 获取遮挡缓存对象
         Object2ByteLinkedOpenHashMap<Block.BlockStatePairKey> object2bytelinkedopenhashmap = OCCLUSION_CACHE.get();
          // 获取并移动到首位的遮挡值
         byte b0 = object2bytelinkedopenhashmap.getAndMoveToFirst(block$blockstatepairkey);
          // 如果遮挡值不为127，则返回遮挡值是否为0
         if (b0 != 127) {
            return b0 != 0;
         } else {
             // 否则，进行实际的遮挡计算
             // 获取当前方块状态的遮挡形状
            VoxelShape voxelshape = p_152445_.getFaceOcclusionShape(p_152446_, p_152447_, p_152448_);
            if (voxelshape.isEmpty()) {// 如果遮挡形状为空，则返回true
               return true;
            } else {
                // 否则，获取相邻方块的遮挡形状
// 判断两个遮挡形状是否有交集
               VoxelShape voxelshape1 = blockstate.getFaceOcclusionShape(p_152446_, p_152449_, p_152448_.getOpposite());
               boolean flag = Shapes.joinIsNotEmpty(voxelshape, voxelshape1, BooleanOp.ONLY_FIRST);
                // 如果遮挡缓存已满，则移除最后一个遮挡值
               if (object2bytelinkedopenhashmap.size() == 2048) {
                  object2bytelinkedopenhashmap.removeLastByte();
               }

               object2bytelinkedopenhashmap.putAndMoveToFirst(block$blockstatepairkey, (byte)(flag ? 1 : 0));
                // 返回是否有遮挡
               return flag;
            }
         }
      } else {// 如果相邻方块的指定面不可遮挡，则返回true

         return true;
      }
   }
// 该方法用于判断指定位置的方块是否支持刚性方块
   public static boolean canSupportRigidBlock(BlockGetter p_49937_, BlockPos p_49938_) {
       // 获取指定位置的方块状态，并判断指定方向的面是否坚固
      return p_49937_.getBlockState(p_49938_).isFaceSturdy(p_49937_, p_49938_, Direction.UP, SupportType.RIGID);
   }
// 该方法用于判断指定位置的方块是否支持中心方块
   public static boolean canSupportCenter(LevelReader p_49864_, BlockPos p_49865_, Direction p_49866_) {
       // 获取指定位置的方块状态
      BlockState blockstate = p_49864_.getBlockState(p_49865_);
       //如果方向为向下且方块状态属于不稳定底部中心方块标签，则返回false；否则判断指定方向的面是否坚固
      return p_49866_ == Direction.DOWN && blockstate.is(BlockTags.UNSTABLE_BOTTOM_CENTER)
         ? false
         : blockstate.isFaceSturdy(p_49864_, p_49865_, p_49866_, SupportType.CENTER);
   }
//该方法用于判断指定方块形状的指定方向的面是否完整
   public static boolean isFaceFull(VoxelShape p_49919_, Direction p_49920_) {
       ///获取指定方块形状的指定方向的面的形状，并判断该形状是否是完整方块
      VoxelShape voxelshape = p_49919_.getFaceShape(p_49920_);
      return isShapeFullBlock(voxelshape);
   }
// 该方法用于判断指定方块形状是否是完整方块
   public static boolean isShapeFullBlock(VoxelShape p_49917_) {
       //通过缓存对象判断指定方块形状是否是完整方块，并返回结果
      return SHAPE_FULL_BLOCK_CACHE.getUnchecked(p_49917_);
   }
//// 该方法用于判断指定方块状态是否向下传播天空光照
   public boolean propagatesSkylightDown(BlockState p_49928_, BlockGetter p_49929_, BlockPos p_49930_) {
      return !isShapeFullBlock(p_49928_.getShape(p_49929_, p_49930_)) && p_49928_.getFluidState().isEmpty();
   }
//// 该方法用于执行方块的动画更新
   public void animateTick(BlockState p_220827_, Level p_220828_, BlockPos p_220829_, RandomSource p_220830_) {
       // 该方法为空，没有具体的实现
   }
//// 该方法用于销毁方块
   public void destroy(LevelAccessor p_49860_, BlockPos p_49861_, BlockState p_49862_) {
   }
/// 该方法用于获取方块掉落的物品堆列表
   public static List<ItemStack> getDrops(BlockState p_49870_, ServerLevel p_49871_, BlockPos p_49872_, @Nullable BlockEntity p_49873_) {
       //// 创建一个掉落参数构建器，并设置掉落上下文的相关参数
      LootParams.Builder lootparams$builder = new LootParams.Builder(p_49871_)
         .withParameter(LootContextParams.ORIGIN, Vec3.atCenterOf(p_49872_))
         .withParameter(LootContextParams.TOOL, ItemStack.EMPTY)
         .withOptionalParameter(LootContextParams.BLOCK_ENTITY, p_49873_);
       //调用方块状态的getDrops方法获取物品堆列表，并返回结果
      return p_49870_.getDrops(lootparams$builder);
   }
//// 该方法用于获取方块掉落的物品堆列表，包括了更多的参数
   public static List<ItemStack> getDrops(
      BlockState p_49875_, ServerLevel p_49876_, BlockPos p_49877_, @Nullable BlockEntity p_49878_, @Nullable Entity p_49879_, ItemStack p_49880_
   ) {
       // 创建一个掉落参数构建器，并设置掉落上下文的相关参数
      LootParams.Builder lootparams$builder = new LootParams.Builder(p_49876_)
         .withParameter(LootContextParams.ORIGIN, Vec3.atCenterOf(p_49877_))
         .withParameter(LootContextParams.TOOL, p_49880_)
         .withOptionalParameter(LootContextParams.THIS_ENTITY, p_49879_)
         .withOptionalParameter(LootContextParams.BLOCK_ENTITY, p_49878_);
       //// 调用方块状态的getDrops方法获取物品堆列表，并返回结果
      return p_49875_.getDrops(lootparams$builder);
   }
//// 该方法用于让方块掉落资源
   public static void dropResources(BlockState p_49951_, Level p_49952_, BlockPos p_49953_) {// 如果当前的世界是服务器级别的
      if (p_49952_ instanceof ServerLevel) {// 获取方块的物品堆列表，并对每个itemstack进行处理
         getDrops(p_49951_, (ServerLevel)p_49952_, p_49953_, null).forEach(p_152406_ -> popResource(p_49952_, p_49953_, p_152406_));// 在方块破坏后生成额外的内容
         p_49951_.spawnAfterBreak((ServerLevel)p_49952_, p_49953_, ItemStack.EMPTY, true);
      }
   }
//// 该方法用于让方块掉落资源
   public static void dropResources(BlockState p_49893_, LevelAccessor p_49894_, BlockPos p_49895_, @Nullable BlockEntity p_49896_) {
      if (p_49894_ instanceof ServerLevel) {// 如果当前的世界是服务器级别的
          // 获取方块的物品堆列表，并对每个物品堆进行处理
         getDrops(p_49893_, (ServerLevel)p_49894_, p_49895_, p_49896_).forEach(p_49859_ -> popResource((ServerLevel)p_49894_, p_49895_, p_49859_));
          // 在方块破坏后生成额外
         p_49893_.spawnAfterBreak((ServerLevel)p_49894_, p_49895_, ItemStack.EMPTY, true);
      }
   }
//// 重载方法：该方法用于让方块掉落资源，并指定是否掉落经验球
   public static void dropResources(
      BlockState p_49882_, Level p_49883_, BlockPos p_49884_, @Nullable BlockEntity p_49885_, @Nullable Entity p_49886_, ItemStack p_49887_
   ) {
       //// 调用原始的dropResources方法，并设置dropXp参数为true
      dropResources(p_49882_, p_49883_, p_49884_, p_49885_, p_49886_, p_49887_, true);
   }
    ///// 重载方法：该方法用于让方块掉落资源，并指定是否掉落经验球
   public static void dropResources(BlockState p_49882_, Level p_49883_, BlockPos p_49884_, @Nullable BlockEntity p_49885_, @Nullable Entity p_49886_, ItemStack p_49887_, boolean dropXp) {// 如果当前的世界是服务器级别的
      if (p_49883_ instanceof ServerLevel) {
          //// 获取方块的物品堆列表，并对每个物品堆进行处理
         getDrops(p_49882_, (ServerLevel)p_49883_, p_49884_, p_49885_, p_49886_, p_49887_).forEach(p_49944_ -> popResource(p_49883_, p_49884_, p_49944_));
          // 在方块破坏后生成额外的掉落物，并根据dropXp参数决定是否掉落经验球
         p_49882_.spawnAfterBreak((ServerLevel)p_49883_, p_49884_, p_49887_, dropXp);
      }
   }
//// 该方法用于在指定位置生成一个掉落物实体
   public static void popResource(Level p_49841_, BlockPos p_49842_, ItemStack p_49843_) {
       //// 计算掉落物的生成位置
      double d0 = (double)EntityType.ITEM.getHeight() / 2.0;
      double d1 = (double)p_49842_.getX() + 0.5 + Mth.nextDouble(p_49841_.random, -0.25, 0.25);
      double d2 = (double)p_49842_.getY() + 0.5 + Mth.nextDouble(p_49841_.random, -0.25, 0.25) - d0;
      double d3 = (double)p_49842_.getZ() + 0.5 + Mth.nextDouble(p_49841_.random, -0.25, 0.25);
       //// 调用popResource方法生成掉落物实体并添加到世界中
      popResource(p_49841_, () -> new ItemEntity(p_49841_, d1, d2, d3, p_49843_), p_49843_);
   }
//// 该方法用于在指定位置的指定方向生成一个掉落物实体
   public static void popResourceFromFace(Level p_152436_, BlockPos p_152437_, Direction p_152438_, ItemStack p_152439_) {
       /// 根据方向计算掉落物的生成位置
      int i = p_152438_.getStepX();
      int j = p_152438_.getStepY();
      int k = p_152438_.getStepZ();
      double d0 = (double)EntityType.ITEM.getWidth() / 2.0;
      double d1 = (double)EntityType.ITEM.getHeight() / 2.0;
      double d2 = (double)p_152437_.getX() + 0.5 + (i == 0 ? Mth.nextDouble(p_152436_.random, -0.25, 0.25) : (double)i * (0.5 + d0));
      double d3 = (double)p_152437_.getY() + 0.5 + (j == 0 ? Mth.nextDouble(p_152436_.random, -0.25, 0.25) : (double)j * (0.5 + d1)) - d1;
      double d4 = (double)p_152437_.getZ() + 0.5 + (k == 0 ? Mth.nextDouble(p_152436_.random, -0.25, 0.25) : (double)k * (0.5 + d0));
      double d5 = i == 0 ? Mth.nextDouble(p_152436_.random, -0.1, 0.1) : (double)i * 0.1;
      double d6 = j == 0 ? Mth.nextDouble(p_152436_.random, 0.0, 0.1) : (double)j * 0.1 + 0.1;
      double d7 = k == 0 ? Mth.nextDouble(p_152436_.random, -0.1, 0.1) : (double)k * 0.1;
      //// 调用popResource方法生成掉落物实体并添加到世界中
       popResource(p_152436_, () -> new ItemEntity(p_152436_, d2, d3, d4, p_152439_, d5, d6, d7), p_152439_);
   }
//// 该方法用于生成掉落物实体并添加到世界中
   private static void popResource(Level p_152441_, Supplier<ItemEntity> p_152442_, ItemStack p_152443_) {
       //// 判断是否需要生成掉落物实体
      if (!p_152441_.isClientSide && !p_152443_.isEmpty() && p_152441_.getGameRules().getBoolean(GameRules.RULE_DOBLOCKDROPS) && !p_152441_.restoringBlockSnapshots) {
          //// 创建掉落物实体并设置默认的拾取延迟
         ItemEntity itementity = p_152442_.get();
         itementity.setDefaultPickUpDelay();
          //// 将掉落物实体添加到世界中
         p_152441_.addFreshEntity(itementity);
      }
   }
// 该方法用于在指定位置生成经验球实体
   public void popExperience(ServerLevel p_49806_, BlockPos p_49807_, int p_49808_) {
      if (p_49806_.getGameRules().getBoolean(GameRules.RULE_DOBLOCKDROPS) && !p_49806_.restoringBlockSnapshots) {// 判断是否允许方块掉落经验球，并且不处于还原方块快照的状态
          // 调用ExperienceOrb.award方法生成经验球实体并添加到世界中
         ExperienceOrb.award(p_49806_, Vec3.atCenterOf(p_49807_), p_49808_);
      }
   }
// 该方法返回方块的爆炸抗性
   @Deprecated //Forge: Use more sensitive version
   public float getExplosionResistance() {
      return this.explosionResistance;
   }
   // 该方法在方块被爆炸时调用
   public void wasExploded(Level p_49844_, BlockPos p_49845_, Explosion p_49846_) {
   }
// 该方法在实体踩踏方块时调用
   public void stepOn(Level p_152431_, BlockPos p_152432_, BlockState p_152433_, Entity p_152434_) {
   }
// 该方法用于在放置方块时获取方块的状态
   @Nullable
   public BlockState getStateForPlacement(BlockPlaceContext p_49820_) {
      return this.defaultBlockState();
   }
// 该方法在玩家破坏方块时调用
   public void playerDestroy(Level p_49827_, Player p_49828_, BlockPos p_49829_, BlockState p_49830_, @Nullable BlockEntity p_49831_, ItemStack p_49832_) {
       // 给玩家奖励“BLOCK_MINED”成就
      p_49828_.awardStat(Stats.BLOCK_MINED.get(this));
       // 使玩家的饱食度减少0.005
      p_49828_.causeFoodExhaustion(0.005F);
      //Forge: Don't drop xp as part of the resources as it is handled by the patches in ServerPlayerGameMode#destroyBlock
       //// 调用dropResources方法丢弃方块的资源（不包括经验球）
      dropResources(p_49830_, p_49827_, p_49829_, p_49831_, p_49828_, p_49832_, false);
   }
//// 该方法在方块被放置时调用
   public void setPlacedBy(Level p_49847_, BlockPos p_49848_, BlockState p_49849_, @Nullable LivingEntity p_49850_, ItemStack p_49851_) {
   }
/// 该方法判断方块是否可以重生
   public boolean isPossibleToRespawnInThis(BlockState p_279289_) {
       //// 如果方块不是实心且不是液体，则可以重生
      return !p_279289_.isSolid() && !p_279289_.liquid();
   }
//// 该方法返回方块的名称
   public MutableComponent getName() {
      return Component.translatable(this.getDescriptionId());
   }
// 该方法返回方块的描述标识
   public String getDescriptionId() {
      if (this.descriptionId == null) {
         this.descriptionId = Util.makeDescriptionId("block", BuiltInRegistries.BLOCK.getKey(this));
      }

      return this.descriptionId;
   }
// 该方法在实体掉落到方块上时调用
   public void fallOn(Level p_152426_, BlockState p_152427_, BlockPos p_152428_, Entity p_152429_, float p_152430_) {
       // 使实体受到摔落伤害
      p_152429_.causeFallDamage(p_152430_, 1.0F, p_152429_.damageSources().fall());
   }
// 该方法在实体掉落到方块上后更新实体状态
   public void updateEntityAfterFallOn(BlockGetter p_49821_, Entity p_49822_) {
       // 将实体的运动速度在水平方向上乘以(1.0, 0.0, 1.0)，即只保留水平方向上的速度
      p_49822_.setDeltaMovement(p_49822_.getDeltaMovement().multiply(1.0, 0.0, 1.0));
       //// 将实体的运动速度在水平方向上乘以(1.0D, 0.0D, 1.0D)，同样只保留水平方向上的速度（使用双精度）
      p_49822_.setDeltaMovement(p_49822_.getDeltaMovement().multiply(1.0D, 0.0D, 1.0D));
   }
// 该方法返回方块的克隆itemstack
   @Deprecated //Forge: Use more sensitive version
   public ItemStack getCloneItemStack(BlockGetter p_49823_, BlockPos p_49824_, BlockState p_49825_) {
      return new ItemStack(this);
   }
// 该方法返回方块的摩擦系数
   public float getFriction() {
      return this.friction;
   }
// 该方法返回方块的速度因子
   public float getSpeedFactor() {
      return this.speedFactor;
   }
// 该方法返回方块的跳跃因子
   public float getJumpFactor() {
      return this.jumpFactor;
   }
// 该方法在方块被破坏时生成破坏粒子效果
   protected void spawnDestroyParticles(Level p_152422_, Player p_152423_, BlockPos p_152424_, BlockState p_152425_) {
      p_152422_.levelEvent(p_152423_, 2001, p_152424_, getId(p_152425_));
   }
//// 该方法在玩家即将破坏方块时调用
   public void playerWillDestroy(Level p_49852_, BlockPos p_49853_, BlockState p_49854_, Player p_49855_) {
       //// 生成破坏粒子效果
      this.spawnDestroyParticles(p_49852_, p_49855_, p_49853_, p_49854_);
       //// 如果方块属于GUARDED_BY_PIGLINS标签，则激怒周围的猪灵
      if (p_49854_.is(BlockTags.GUARDED_BY_PIGLINS)) {
         PiglinAi.angerNearbyPiglins(p_49855_, false);
      }
// 发送方块销毁的游戏事件
      p_49852_.gameEvent(GameEvent.BLOCK_DESTROY, p_49853_, GameEvent.Context.of(p_49855_, p_49854_));
   }
// 该方法处理方块与降水的交互（例如雨、雪等）
   public void handlePrecipitation(BlockState p_152450_, Level p_152451_, BlockPos p_152452_, Biome.Precipitation p_152453_) {
   }
// 该方法确定方块在爆炸中是否会掉落
   @Deprecated //Forge: Use more sensitive version
   public boolean dropFromExplosion(Explosion p_49826_) {
      return true;
   }
// 该方法用于创建方块的状态定义
   protected void createBlockStateDefinition(StateDefinition.Builder<Block, BlockState> p_49915_) {
   }
// 该方法返回方块的状态定义
   public StateDefinition<Block, BlockState> getStateDefinition() {
      return this.stateDefinition;
   }
// 该方法注册方块的默认状态
   protected final void registerDefaultState(BlockState p_49960_) {
      this.defaultBlockState = p_49960_;
   }
// 该方法返回方块的默认状态
   public final BlockState defaultBlockState() {
      return this.defaultBlockState;
   }
//// 该方法通过另一个方块的状态来设置当前方块的状态
   public final BlockState withPropertiesOf(BlockState p_152466_) {
       //// 获取当前方块的默认状态
      BlockState blockstate = this.defaultBlockState();
//// 遍历另一个方块的所有属性
      for(Property<?> property : p_152466_.getBlock().getStateDefinition().getProperties()) {
          // 如果当前方块的默认状态包含相同的属性
         if (blockstate.hasProperty(property)) {
             // 复制另一个方块的属性值到当前方块的状态中
            blockstate = copyProperty(p_152466_, blockstate, property);
         }
      }

      return blockstate;
   }
// 该方法复制一个方块状态的属性值到另一个方块状态中
   private static <T extends Comparable<T>> BlockState copyProperty(BlockState p_152455_, BlockState p_152456_, Property<T> p_152457_) {
       // 使用另一个方块状态的属性值设置当前方块状态的属性值
      return p_152456_.setValue(p_152457_, p_152455_.getValue(p_152457_));
   }
// 该方法返回方块的声音类型
   @Deprecated //Forge: Use more sensitive version {@link IForgeBlockState#getSoundType(IWorldReader, BlockPos, Entity) }
   public SoundType getSoundType(BlockState p_49963_) {
      return this.soundType;
   }
// 重写的方法，返回方块对应的物品对象

   @Override
   public Item asItem() {
      if (this.item == null) {
         this.item = Item.byBlock(this);
      }
// 返回物品对象
      return net.neoforged.neoforge.registries.ForgeRegistries.ITEMS.getDelegateOrThrow(this.item).get(); // Forge: Vanilla caches the items, update with registry replacements.
   }
// 判断方块是否具有动态形状
   public boolean hasDynamicShape() {
      return this.dynamicShape;
   }
// 重写的方法，返回方块的字符串表示
   @Override
   public String toString() {
      return "Block{" + BuiltInRegistries.BLOCK.getKey(this) + "}";
   }
// 添加悬停文本，用于显示方块的信息
   public void appendHoverText(ItemStack p_49816_, @Nullable BlockGetter p_49817_, List<Component> p_49818_, TooltipFlag p_49819_) {
   }
// 重写的方法，返回当前方块对象
   @Override
   protected Block asBlock() {
      return this;
   }
// 获取每个方块状态对应的体素形状，并以不可变映射的形式返回
   protected ImmutableMap<BlockState, VoxelShape> getShapeForEachState(Function<BlockState, VoxelShape> p_152459_) {
      return this.stateDefinition.getPossibleStates().stream().collect(ImmutableMap.toImmutableMap(Function.identity(), p_152459_));
   }

   /* ======================================== FORGE START =====================================*/
    // 渲染属性的私有字段
   private Object renderProperties;

   /*
      DO NOT CALL, IT WILL DISAPPEAR IN THE FUTURE
      Call RenderProperties.get instead
    */
   public Object getRenderPropertiesInternal() {
      return renderProperties;
   }
// 初始化客户端方法
   private void initClient() {
      // Minecraft instance isn't available in datagen, so don't call initializeClient if in datagen
      if (net.neoforged.fml.loading.FMLEnvironment.dist == net.neoforged.api.distmarker.Dist.CLIENT && !net.neoforged.fml.loading.FMLLoader.getLaunchHandler().isData()) {
         initializeClient(properties -> {
            if (properties == this)
               throw new IllegalStateException("Don't extend IBlockRenderProperties in your block, use an anonymous class instead.");
            this.renderProperties = properties;
         });
      }
   }
// 初始化客户端的方法，接受一个 Consumer 参数
   public void initializeClient(java.util.function.Consumer<net.neoforged.neoforge.client.extensions.common.IClientBlockExtensions> consumer) {
   }
// 重写的方法，判断方块是否能支撑植物
   @Override
   public boolean canSustainPlant(BlockState state, BlockGetter world, BlockPos pos, Direction facing, net.neoforged.neoforge.common.IPlantable plantable) {
       // 获取植物的方块状态和植物类型
      BlockState plant = plantable.getPlant(world, pos.relative(facing));
      net.neoforged.neoforge.common.PlantType type = plantable.getPlantType(world, pos.relative(facing));
// 如果植物是仙人掌
      if (plant.getBlock() == Blocks.CACTUS)
         return state.is(Blocks.CACTUS) || state.is(BlockTags.SAND);
// 如果植物是甘蔗，当前方块也是甘蔗
      if (plant.getBlock() == Blocks.SUGAR_CANE && this == Blocks.SUGAR_CANE)
         return true;
// 如果植物是 BushBlock，并且可以放置在当前方块上
      if (plantable instanceof BushBlock && ((BushBlock)plantable).mayPlaceOn(state, world, pos))
         return true;
// 根据植物类型进行判断
      if (net.neoforged.neoforge.common.PlantType.DESERT.equals(type)) {
         return state.is(BlockTags.SAND) || this == Blocks.TERRACOTTA || this instanceof GlazedTerracottaBlock;
      } else if (net.neoforged.neoforge.common.PlantType.NETHER.equals(type)) {
         return this == Blocks.SOUL_SAND;
      } else if (net.neoforged.neoforge.common.PlantType.CROP.equals(type)) {
         return state.is(Blocks.FARMLAND);
      } else if (net.neoforged.neoforge.common.PlantType.CAVE.equals(type)) {
         return state.isFaceSturdy(world, pos, Direction.UP);
      } else if (net.neoforged.neoforge.common.PlantType.PLAINS.equals(type)) {
         return state.is(BlockTags.DIRT) || this == Blocks.FARMLAND;
      } else if (net.neoforged.neoforge.common.PlantType.WATER.equals(type)) {
         return (state.is(Blocks.WATER) || state.getBlock() instanceof IceBlock) && world.getFluidState(pos.relative(facing)).isEmpty();
      } else if (net.neoforged.neoforge.common.PlantType.BEACH.equals(type)) {
         boolean isBeach = state.is(BlockTags.DIRT) || state.is(BlockTags.SAND);
         boolean hasWater = false;
         for (Direction face : Direction.Plane.HORIZONTAL) {
            BlockState adjacentBlockState = world.getBlockState(pos.relative(face));
            var adjacentFluidState = world.getFluidState(pos.relative(face));
            hasWater = hasWater || adjacentBlockState.is(Blocks.FROSTED_ICE) || adjacentFluidState.is(net.minecraft.tags.FluidTags.WATER);
            if (hasWater)
               break; //No point continuing.
         }
         return isBeach && hasWater;
      }
      return false;
   }

   /* ========================================= FORGE END ======================================*/
// builtInRegistryHolder 方法已过时（Deprecated），返回一个 Holder.Reference<Block> 对象
   /** @deprecated */
   @Deprecated
   public Holder.Reference<Block> builtInRegistryHolder() {
      return this.builtInRegistryHolder;
   }
// 尝试掉落经验的方法，接受服务器级别、方块坐标、物品堆叠和经验提供者作为参数
   protected void tryDropExperience(ServerLevel p_220823_, BlockPos p_220824_, ItemStack p_220825_, IntProvider p_220826_) {
       // 如果物品堆叠上没有附魔属性 SILK_TOUCH
      if (EnchantmentHelper.getItemEnchantmentLevel(Enchantments.SILK_TOUCH, p_220825_) == 0) {
          // 从经验提供者获取一个随机值
         int i = p_220826_.sample(p_220823_.random);
         if (i > 0) {
            this.popExperience(p_220823_, p_220824_, i);
         }
      }
   }
// 第一个方块状态、第二个方块状态和方向的私有字段
   public static final class BlockStatePairKey {
      private final BlockState first;
      private final BlockState second;
      private final Direction direction;
// 构造函数，接受第一个方块状态、第二个方块状态和方向作为参数
      public BlockStatePairKey(BlockState p_49984_, BlockState p_49985_, Direction p_49986_) {
         this.first = p_49984_;
         this.second = p_49985_;
         this.direction = p_49986_;
      }
// 重写的 equals 方法，判断两个 BlockStatePairKey 对象是否相等
      @Override
      public boolean equals(Object p_49988_) {
         if (this == p_49988_) {
            return true;
         } else if (!(p_49988_ instanceof Block.BlockStatePairKey)) {
            return false;
         } else {
            Block.BlockStatePairKey block$blockstatepairkey = (Block.BlockStatePairKey)p_49988_;
            return this.first == block$blockstatepairkey.first
               && this.second == block$blockstatepairkey.second
               && this.direction == block$blockstatepairkey.direction;
         }
      }
// 重写的 hashCode 方法，计算 BlockStatePairKey 对象的哈希值
      @Override
      public int hashCode() {
         int i = this.first.hashCode();
         i = 31 * i + this.second.hashCode();
         return 31 * i + this.direction.hashCode();
      }
   }
}

```

