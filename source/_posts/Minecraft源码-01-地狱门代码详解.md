---
title: 地狱门代码详解
date: 2023-11-07 09:22:33
tags:
- 我的世界
- 源码
cover: https://w.wallhaven.cc/full/85/wallhaven-858w9j.png
---



# 结论

通过火焰方块的放置判断当前的位置是否可以生成地狱门传送方块，其中判断是否满足生成的条件使用的PortalShape类。若可以生成EventHooks的onTrySpawnPortal方法。其中EventHooks的onTrySpawnPortal方法调用了PortalSpawnEvent事件。之后得到的结果为true的情况下。通过PortalShape类的createPortalBlocks方法创建方块。

# 意义

地狱门就是一个多方块结构，通过这个实现我们可以设计其他的多方块结构。

# PortalShape类

用于处理和检测游戏中的传送门

```java
package net.minecraft.world.level.portal;

import java.util.Optional;
import java.util.function.Predicate;
import javax.annotation.Nullable;
import net.minecraft.BlockUtil;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.tags.BlockTags;
import net.minecraft.util.Mth;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityDimensions;
import net.minecraft.world.level.BlockGetter;
import net.minecraft.world.level.LevelAccessor;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.NetherPortalBlock;
import net.minecraft.world.level.block.state.BlockBehaviour;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.BlockStateProperties;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;
import net.minecraft.world.phys.shapes.Shapes;
import net.minecraft.world.phys.shapes.VoxelShape;

public class PortalShape { // 处理游戏中的传送门的形状
   private static final int MIN_WIDTH = 2; // 最小宽度
   public static final int MAX_WIDTH = 21; // 最大宽度
   private static final int MIN_HEIGHT = 3; // 最小高度
   public static final int MAX_HEIGHT = 21; // 最大高度
   private static final BlockBehaviour.StatePredicate FRAME = net.neoforged.neoforge.common.extensions.IBlockStateExtension::isPortalFrame; // 传送门框架判断函数
   private static final float SAFE_TRAVEL_MAX_ENTITY_XY = 4.0F; // 
   private static final double SAFE_TRAVEL_MAX_VERTICAL_DELTA = 1.0;
   private final LevelAccessor level;
   private final Direction.Axis axis;
   private final Direction rightDir;
   private int numPortalBlocks;
   @Nullable
   private BlockPos bottomLeft;
   private int height;
   private final int width;
	// 用于给定位置和方向上寻找符合条件的传送门形状
   public static Optional<PortalShape> findEmptyPortalShape(LevelAccessor p_77709_, BlockPos p_77710_, Direction.Axis p_77711_) {
      return findPortalShape(p_77709_, p_77710_, p_77727_ -> p_77727_.isValid() && p_77727_.numPortalBlocks == 0, p_77711_);
   }
	// 在给定的环境中寻找一个符合条件的传送门的形状
   public static Optional<PortalShape> findPortalShape(LevelAccessor p_77713_, BlockPos p_77714_, Predicate<PortalShape> p_77715_, Direction.Axis p_77716_) {
       // 创建一个传送门形状，调用谓词进行过滤。
      Optional<PortalShape> optional = Optional.of(new PortalShape(p_77713_, p_77714_, p_77716_)).filter(p_77715_);
      if (optional.isPresent()) {
         return optional;// 如果存在就返回
      } else {
         Direction.Axis direction$axis = p_77716_ == Direction.Axis.X ? Direction.Axis.Z : Direction.Axis.X;
          // 否则就创建一个新的optional对象，其轴的方向和原来不同，继续过滤。
         return Optional.of(new PortalShape(p_77713_, p_77714_, direction$axis)).filter(p_77715_);
      }
   }
// 初始化一个新的传送门的形状
   public PortalShape(LevelAccessor p_77695_, BlockPos p_77696_, Direction.Axis p_77697_) {
      this.level = p_77695_;// level
      this.axis = p_77697_; // 方向
      this.rightDir = p_77697_ == Direction.Axis.X ? Direction.WEST : Direction.SOUTH;
      this.bottomLeft = this.calculateBottomLeft(p_77696_);// 调用calculateBottomLeft计算
      if (this.bottomLeft == null) {//如果计算为空
         this.bottomLeft = p_77696_;//赋值为blockpos
         this.width = 1;//宽度1
         this.height = 1;//高度1
      } else {
         this.width = this.calculateWidth();//计算宽度
         if (this.width > 0) {// 如果宽度>0
            this.height = this.calculateHeight();//计算高度
         }
      }
   }
	// 计算传送门底部左边的位置
   @Nullable
   private BlockPos calculateBottomLeft(BlockPos p_77734_) {
      int i = Math.max(this.level.getMinBuildHeight(), p_77734_.getY() - 21);

      while(p_77734_.getY() > i && isEmpty(this.level.getBlockState(p_77734_.below()))) {
         p_77734_ = p_77734_.below();
      }

      Direction direction = this.rightDir.getOpposite();
      int j = this.getDistanceUntilEdgeAboveFrame(p_77734_, direction) - 1;
      return j < 0 ? null : p_77734_.relative(direction, j);
   }
	// 计算宽度
   private int calculateWidth() {
      int i = this.getDistanceUntilEdgeAboveFrame(this.bottomLeft, this.rightDir);
      return i >= 2 && i <= 21 ? i : 0;
   }
	// 计算传送门到边缘的位置
   private int getDistanceUntilEdgeAboveFrame(BlockPos p_77736_, Direction p_77737_) {
      BlockPos.MutableBlockPos blockpos$mutableblockpos = new BlockPos.MutableBlockPos();

      for(int i = 0; i <= 21; ++i) {
         blockpos$mutableblockpos.set(p_77736_).move(p_77737_, i);
         BlockState blockstate = this.level.getBlockState(blockpos$mutableblockpos);
         if (!isEmpty(blockstate)) {
            if (FRAME.test(blockstate, this.level, blockpos$mutableblockpos)) {
               return i;
            }
            break;
         }

         BlockState blockstate1 = this.level.getBlockState(blockpos$mutableblockpos.move(Direction.DOWN));
         if (!FRAME.test(blockstate1, this.level, blockpos$mutableblockpos)) {
            break;
         }
      }

      return 0;
   }
	// 计算高度
   private int calculateHeight() {
      BlockPos.MutableBlockPos blockpos$mutableblockpos = new BlockPos.MutableBlockPos();
      int i = this.getDistanceUntilTop(blockpos$mutableblockpos);
      return i >= 3 && i <= 21 && this.hasTopFrame(blockpos$mutableblockpos, i) ? i : 0;
   }

   private boolean hasTopFrame(BlockPos.MutableBlockPos p_77731_, int p_77732_) {
      for(int i = 0; i < this.width; ++i) {
         BlockPos.MutableBlockPos blockpos$mutableblockpos = p_77731_.set(this.bottomLeft).move(Direction.UP, p_77732_).move(this.rightDir, i);
         if (!FRAME.test(this.level.getBlockState(blockpos$mutableblockpos), this.level, blockpos$mutableblockpos)) {
            return false;
         }
      }

      return true;
   }
// 计算传送门顶部的位置
   private int getDistanceUntilTop(BlockPos.MutableBlockPos p_77729_) {
      for(int i = 0; i < 21; ++i) {
         p_77729_.set(this.bottomLeft).move(Direction.UP, i).move(this.rightDir, -1);
         if (!FRAME.test(this.level.getBlockState(p_77729_), this.level, p_77729_)) {
            return i;
         }

         p_77729_.set(this.bottomLeft).move(Direction.UP, i).move(this.rightDir, this.width);
         if (!FRAME.test(this.level.getBlockState(p_77729_), this.level, p_77729_)) {
            return i;
         }

         for(int j = 0; j < this.width; ++j) {
            p_77729_.set(this.bottomLeft).move(Direction.UP, i).move(this.rightDir, j);
            BlockState blockstate = this.level.getBlockState(p_77729_);
            if (!isEmpty(blockstate)) {
               return i;
            }

            if (blockstate.is(Blocks.NETHER_PORTAL)) {
               ++this.numPortalBlocks;
            }
         }
      }

      return 21;
   }
	// 判断一个位置是否是空的
   private static boolean isEmpty(BlockState p_77718_) {
      return p_77718_.isAir() || p_77718_.is(BlockTags.FIRE) || p_77718_.is(Blocks.NETHER_PORTAL);
   }
	// 判断传送门是否是合法的
   public boolean isValid() {
      return this.bottomLeft != null && this.width >= 2 && this.width <= 21 && this.height >= 3 && this.height <= 21;
   }
// 创建传送门方块（紫色那个）
   public void createPortalBlocks() {
      BlockState blockstate = Blocks.NETHER_PORTAL.defaultBlockState().setValue(NetherPortalBlock.AXIS, this.axis);
      BlockPos.betweenClosed(this.bottomLeft, this.bottomLeft.relative(Direction.UP, this.height - 1).relative(this.rightDir, this.width - 1))
         .forEach(p_77725_ -> this.level.setBlock(p_77725_, blockstate, 18));
   }
// 传送门是否完整
   public boolean isComplete() {
      return this.isValid() && this.numPortalBlocks == this.width * this.height;
   }
// 获得相对位置
   public static Vec3 getRelativePosition(BlockUtil.FoundRectangle p_77739_, Direction.Axis p_77740_, Vec3 p_77741_, EntityDimensions p_77742_) {
      double d0 = (double)p_77739_.axis1Size - (double)p_77742_.width;
      double d1 = (double)p_77739_.axis2Size - (double)p_77742_.height;
      BlockPos blockpos = p_77739_.minCorner;
      double d2;
      if (d0 > 0.0) {
         float f = (float)blockpos.get(p_77740_) + p_77742_.width / 2.0F;
         d2 = Mth.clamp(Mth.inverseLerp(p_77741_.get(p_77740_) - (double)f, 0.0, d0), 0.0, 1.0);
      } else {
         d2 = 0.5;
      }

      double d4;
      if (d1 > 0.0) {
         Direction.Axis direction$axis = Direction.Axis.Y;
         d4 = Mth.clamp(Mth.inverseLerp(p_77741_.get(direction$axis) - (double)blockpos.get(direction$axis), 0.0, d1), 0.0, 1.0);
      } else {
         d4 = 0.0;
      }

      Direction.Axis direction$axis1 = p_77740_ == Direction.Axis.X ? Direction.Axis.Z : Direction.Axis.X;
      double d3 = p_77741_.get(direction$axis1) - ((double)blockpos.get(direction$axis1) + 0.5);
      return new Vec3(d2, d4, d3);
   }
// 创建传送门信息
   public static PortalInfo createPortalInfo(
      ServerLevel p_259301_, // 世界
      BlockUtil.FoundRectangle p_259931_,//传送门的矩形区域
      Direction.Axis p_259901_,//传送门的轴
      Vec3 p_259630_,//传送门的位置
      Entity p_259166_,//实体
      Vec3 p_260043_,//实体位置
      float p_259853_,//旋转角度
      float p_259667_//实体旋转角度
   ) {
      BlockPos blockpos = p_259931_.minCorner;//给定区域获得门户的坐标
      BlockState blockstate = p_259301_.getBlockState(blockpos);//获得方块状态
      Direction.Axis direction$axis = blockstate.getOptionalValue(BlockStateProperties.HORIZONTAL_AXIS).orElse(Direction.Axis.X);//从方块状态中获得传送门的轴
      double d0 = (double)p_259931_.axis1Size;
      double d1 = (double)p_259931_.axis2Size;
      EntityDimensions entitydimensions = p_259166_.getDimensions(p_259166_.getPose());
      int i = p_259901_ == direction$axis ? 0 : 90;
      Vec3 vec3 = p_259901_ == direction$axis ? p_260043_ : new Vec3(p_260043_.z, p_260043_.y, -p_260043_.x);
      double d2 = (double)entitydimensions.width / 2.0 + (d0 - (double)entitydimensions.width) * p_259630_.x();
      double d3 = (d1 - (double)entitydimensions.height) * p_259630_.y();
      double d4 = 0.5 + p_259630_.z();
      boolean flag = direction$axis == Direction.Axis.X;
      Vec3 vec31 = new Vec3((double)blockpos.getX() + (flag ? d2 : d4), (double)blockpos.getY() + d3, (double)blockpos.getZ() + (flag ? d4 : d2));
      Vec3 vec32 = findCollisionFreePosition(vec31, p_259301_, p_259166_, entitydimensions);
      return new PortalInfo(vec32, vec3, p_259853_ + (float)i, p_259667_);
   }
// 用于找到碰撞免疫的位置
   private static Vec3 findCollisionFreePosition(Vec3 p_260315_, ServerLevel p_259704_, Entity p_259626_, EntityDimensions p_259816_) {
      if (!(p_259816_.width > 4.0F) && !(p_259816_.height > 4.0F)) {
         double d0 = (double)p_259816_.height / 2.0;
         Vec3 vec3 = p_260315_.add(0.0, d0, 0.0);
         VoxelShape voxelshape = Shapes.create(
            AABB.ofSize(vec3, (double)p_259816_.width, 0.0, (double)p_259816_.width).expandTowards(0.0, 1.0, 0.0).inflate(1.0E-6)
         );
         Optional<Vec3> optional = p_259704_.findFreePosition(
            p_259626_, voxelshape, vec3, (double)p_259816_.width, (double)p_259816_.height, (double)p_259816_.width
         );
         Optional<Vec3> optional1 = optional.map(p_259019_ -> p_259019_.subtract(0.0, d0, 0.0));
         return optional1.orElse(p_260315_);
      } else {
         return p_260315_;
      }
   }
}

```



# BaseFrieBlock类

// 火焰类型方块的基类

```java
package net.minecraft.world.level.block;

import java.util.Optional;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.particles.ParticleTypes;
import net.minecraft.sounds.SoundEvents;
import net.minecraft.sounds.SoundSource;
import net.minecraft.util.RandomSource;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.context.BlockPlaceContext;
import net.minecraft.world.level.BlockGetter;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.state.BlockBehaviour;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.portal.PortalShape;
import net.minecraft.world.phys.shapes.CollisionContext;
import net.minecraft.world.phys.shapes.VoxelShape;

public abstract class BaseFireBlock extends Block {
   private static final int SECONDS_ON_FIRE = 8;
   private final float fireDamage;
   protected static final float AABB_OFFSET = 1.0F;
   protected static final VoxelShape DOWN_AABB = Block.box(0.0, 0.0, 0.0, 16.0, 1.0, 16.0);

   public BaseFireBlock(BlockBehaviour.Properties p_49241_, float p_49242_) {
      super(p_49241_);
      this.fireDamage = p_49242_; // 火焰伤害
   }

    // 根据放置的位置和level决定状态
   @Override
   public BlockState getStateForPlacement(BlockPlaceContext p_49244_) {
      return getState(p_49244_.getLevel(), p_49244_.getClickedPos());
   }
	// 根据getter和blockpos决定火焰方块的状态
   public static BlockState getState(BlockGetter p_49246_, BlockPos p_49247_) {
      BlockPos blockpos = p_49247_.below();
      BlockState blockstate = p_49246_.getBlockState(blockpos);
      return SoulFireBlock.canSurviveOnBlock(blockstate)
         ? Blocks.SOUL_FIRE.defaultBlockState()
         : ((FireBlock)Blocks.FIRE).getStateForPlacement(p_49246_, p_49247_);
   }
	// 返回火焰方块的形状，用于碰撞检测
   @Override
   public VoxelShape getShape(BlockState p_49274_, BlockGetter p_49275_, BlockPos p_49276_, CollisionContext p_49277_) {
      return DOWN_AABB;
   }
	// 每个tick会调用的方法，处理火焰动画的效果，例如播放火焰的环境音效，添加火焰的粒子效果
   @Override
   public void animateTick(BlockState p_220763_, Level p_220764_, BlockPos p_220765_, RandomSource p_220766_) {
      if (p_220766_.nextInt(24) == 0) {
         p_220764_.playLocalSound(
            (double)p_220765_.getX() + 0.5,
            (double)p_220765_.getY() + 0.5,
            (double)p_220765_.getZ() + 0.5,
            SoundEvents.FIRE_AMBIENT,
            SoundSource.BLOCKS,
            1.0F + p_220766_.nextFloat(),
            p_220766_.nextFloat() * 0.7F + 0.3F,
            false
         );
      }

      BlockPos blockpos = p_220765_.below();
      BlockState blockstate = p_220764_.getBlockState(blockpos);
      if (!this.canBurn(blockstate) && !blockstate.isFaceSturdy(p_220764_, blockpos, Direction.UP)) {
         if (this.canBurn(p_220764_.getBlockState(p_220765_.west()))) {
            for(int j = 0; j < 2; ++j) {
               double d3 = (double)p_220765_.getX() + p_220766_.nextDouble() * 0.1F;
               double d8 = (double)p_220765_.getY() + p_220766_.nextDouble();
               double d13 = (double)p_220765_.getZ() + p_220766_.nextDouble();
               p_220764_.addParticle(ParticleTypes.LARGE_SMOKE, d3, d8, d13, 0.0, 0.0, 0.0);
            }
         }

         if (this.canBurn(p_220764_.getBlockState(p_220765_.east()))) {
            for(int k = 0; k < 2; ++k) {
               double d4 = (double)(p_220765_.getX() + 1) - p_220766_.nextDouble() * 0.1F;
               double d9 = (double)p_220765_.getY() + p_220766_.nextDouble();
               double d14 = (double)p_220765_.getZ() + p_220766_.nextDouble();
               p_220764_.addParticle(ParticleTypes.LARGE_SMOKE, d4, d9, d14, 0.0, 0.0, 0.0);
            }
         }

         if (this.canBurn(p_220764_.getBlockState(p_220765_.north()))) {
            for(int l = 0; l < 2; ++l) {
               double d5 = (double)p_220765_.getX() + p_220766_.nextDouble();
               double d10 = (double)p_220765_.getY() + p_220766_.nextDouble();
               double d15 = (double)p_220765_.getZ() + p_220766_.nextDouble() * 0.1F;
               p_220764_.addParticle(ParticleTypes.LARGE_SMOKE, d5, d10, d15, 0.0, 0.0, 0.0);
            }
         }

         if (this.canBurn(p_220764_.getBlockState(p_220765_.south()))) {
            for(int i1 = 0; i1 < 2; ++i1) {
               double d6 = (double)p_220765_.getX() + p_220766_.nextDouble();
               double d11 = (double)p_220765_.getY() + p_220766_.nextDouble();
               double d16 = (double)(p_220765_.getZ() + 1) - p_220766_.nextDouble() * 0.1F;
               p_220764_.addParticle(ParticleTypes.LARGE_SMOKE, d6, d11, d16, 0.0, 0.0, 0.0);
            }
         }

         if (this.canBurn(p_220764_.getBlockState(p_220765_.above()))) {
            for(int j1 = 0; j1 < 2; ++j1) {
               double d7 = (double)p_220765_.getX() + p_220766_.nextDouble();
               double d12 = (double)(p_220765_.getY() + 1) - p_220766_.nextDouble() * 0.1F;
               double d17 = (double)p_220765_.getZ() + p_220766_.nextDouble();
               p_220764_.addParticle(ParticleTypes.LARGE_SMOKE, d7, d12, d17, 0.0, 0.0, 0.0);
            }
         }
      } else {
         for(int i = 0; i < 3; ++i) {
            double d0 = (double)p_220765_.getX() + p_220766_.nextDouble();
            double d1 = (double)p_220765_.getY() + p_220766_.nextDouble() * 0.5 + 0.5;
            double d2 = (double)p_220765_.getZ() + p_220766_.nextDouble();
            p_220764_.addParticle(ParticleTypes.LARGE_SMOKE, d0, d1, d2, 0.0, 0.0, 0.0);
         }
      }
   }
	// 抽象方法，用于判断方块是否可以燃烧
   protected abstract boolean canBurn(BlockState p_49284_);
	// 处理实体进入火焰的状况，例如添加火焰伤害和燃烧效果
   @Override
   public void entityInside(BlockState p_49260_, Level p_49261_, BlockPos p_49262_, Entity p_49263_) {
      if (!p_49263_.fireImmune()) {
         p_49263_.setRemainingFireTicks(p_49263_.getRemainingFireTicks() + 1);
         if (p_49263_.getRemainingFireTicks() == 0) {
            p_49263_.setSecondsOnFire(8);
         }
      }

      p_49263_.hurt(p_49261_.damageSources().inFire(), this.fireDamage);
      super.entityInside(p_49260_, p_49261_, p_49262_, p_49263_);
   }
	// 处理方块被放置的情况，例如在特定的维度中，如果方块上方有足够的空间，就会尝试创建一个传送门
   @Override
   public void onPlace(BlockState p_49279_, Level p_49280_, BlockPos p_49281_, BlockState p_49282_, boolean p_49283_) {
      if (!p_49282_.is(p_49279_.getBlock())) {
         if (inPortalDimension(p_49280_)) {
            Optional<PortalShape> optional = PortalShape.findEmptyPortalShape(p_49280_, p_49281_, Direction.Axis.X);
            optional = net.neoforged.neoforge.event.EventHooks.onTrySpawnPortal(p_49280_, p_49281_, optional);
            if (optional.isPresent()) {
               optional.get().createPortalBlocks();
               return;
            }
         }

         if (!p_49279_.canSurvive(p_49280_, p_49281_)) {
            p_49280_.removeBlock(p_49281_, false);
         }
      }
   }
	// 判断当前维度世界是否可以创建传送门
   private static boolean inPortalDimension(Level p_49249_) {
      return p_49249_.dimension() == Level.OVERWORLD || p_49249_.dimension() == Level.NETHER;
   }
	// 处理方块被破坏时候的粒子效果
   @Override
   protected void spawnDestroyParticles(Level p_152139_, Player p_152140_, BlockPos p_152141_, BlockState p_152142_) {
   }
	// 处理玩家破坏方块时候的情况
   @Override
   public void playerWillDestroy(Level p_49251_, BlockPos p_49252_, BlockState p_49253_, Player p_49254_) {
      if (!p_49251_.isClientSide()) {
         p_49251_.levelEvent(null, 1009, p_49252_, 0);
      }

      super.playerWillDestroy(p_49251_, p_49252_, p_49253_, p_49254_);
   }
	//它用于判断方块是否可以被放置在给定的位置。
   public static boolean canBePlacedAt(Level p_49256_, BlockPos p_49257_, Direction p_49258_) {
      BlockState blockstate = p_49256_.getBlockState(p_49257_);
      if (!blockstate.isAir()) {
         return false;
      } else {
         return getState(p_49256_, p_49257_).canSurvive(p_49256_, p_49257_) || isPortal(p_49256_, p_49257_, p_49258_);
      }
   }
	// 静态方法，用于判断给定的位置是否可以创建传送门
   private static boolean isPortal(Level p_49270_, BlockPos p_49271_, Direction p_49272_) {
      if (!inPortalDimension(p_49270_)) {
         return false;
      } else {
         BlockPos.MutableBlockPos blockpos$mutableblockpos = p_49271_.mutable();
         boolean flag = false;

         for(Direction direction : Direction.values()) {
            if (p_49270_.getBlockState(blockpos$mutableblockpos.set(p_49271_).move(direction)).isPortalFrame(p_49270_, blockpos$mutableblockpos)) {
               flag = true;
               break;
            }
         }

         if (!flag) {
            return false;
         } else {
            Direction.Axis direction$axis = p_49272_.getAxis().isHorizontal()
               ? p_49272_.getCounterClockWise().getAxis()
               : Direction.Plane.HORIZONTAL.getRandomAxis(p_49270_.random);
            return PortalShape.findEmptyPortalShape(p_49270_, p_49271_, direction$axis).isPresent();
         }
      }
   }
}

```

