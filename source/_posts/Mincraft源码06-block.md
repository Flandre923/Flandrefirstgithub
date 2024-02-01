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


    // 创建一个缓存,通过一个 VoxelShape 快速判断它是否是一个完整的方块
 LoadingCache<VoxelShape, Boolean> SHAPE_FULL_BLOCK_CACHE
    
    
    // 邻近方块更新标记
UPDATE_NEIGHBORS 
    // 通知客户端方块改变标记 
UPDATE_CLIENTS
    //  不可见的方块改变标记
  UPDATE_INVISIBLE
    // 立即更新标记
UPDATE_IMMEDIATE
    // 更新时使用已知的方块形状标记
UPDATE_KNOWN_SHAPE 
UPDATE_SUPPRESS_DROPS 
UPDATE_MOVE_BY_PISTON 
UPDATE_NONE 
UPDATE_ALL 
UPDATE_ALL_IMMEDIATE
INDESTRUCTIBLE
INSTANT 
UPDATE_LIMIT 
    //// 定义Block状态的状态定义,用于映射Block到其状态
StateDefinition<Block, BlockState> stateDefinition;
    // Block的默认状态
 BlockState defaultBlockState;
    // Block的描述ID,可为空
String descriptionId;
    // 与这个Block相关的Item,可为空 
Item item;
    // 用于遮挡(光照)计算的缓存大小
int CACHE_SIZE = 2048;
   
// 获取BlockState的id,如果为null则返回0  
   public static int getId(@Nullable BlockState ) {

   }
    
    
// 通过id获取BlockState,如果不存在则返回AIR的默认状态
     BlockState stateById(int ) {
   
   }
// 通过Item获取其相关的Block,如果不是BlockItem则返回AIR
     Block byItem(@Nullable Item ) {

   }
    
    
// 当一个BlockState被另一个取代时,移动实体到上方避免卡住
     BlockState pushEntitiesUp(_) {
      
   
   }
    
    
//创建一个表示方块碰撞箱的VoxelShape,输入是原始像素大小,会自动缩放到1/16大小
   public static VoxelShape box() {
     
       
       
//// 根据相邻方块更新方块状态
   public static BlockState updateFromNeighbourShapes() {
     
     
   }
       
       
// 更新方块或者销毁
     void updateOrDestroy() {
         //
   }
       
       
// 更新或销毁方块
   public static void updateOrDestroy() {
     
   }
//
   public Block(BlockBehaviour.Properties ) 
       //此处实现
   }
       
       
// 判断一个BlockState是否是用于连接检查的例外
     boolean isExceptionForConnection(BlockState ) {
           // 如果是树叶Block、障碍物、南瓜、瓜块、潜影盒等返回true
   }
//// 直接返回是否随机tick的属性
   public boolean isRandomlyTicking(BlockState ) 
   }


//该方法用于判断是否应该渲染方块的某个面。
   public static boolean shouldRenderFace(BlockState , BlockGetter , BlockPos , Direction , BlockPos ) {
    
   }

//删了 刚体


//该方法用于判断指定方块形状的指定方向的面是否完整
     boolean isFaceFull(VoxelShape , Direction ) {
       ///获取指定方块形状的指定方向的面的形状，并判断该形状是否是完整方块

   }
// 该方法用于判断指定方块形状是否是完整方块
     boolean isShapeFullBlock(VoxelShape ) {
       //通过缓存对象判断指定方块形状是否是完整方块，并返回结果
   }


//// 该方法用于判断指定方块状态是否向下传播天空光照
    boolean propagatesSkylightDown() {
     
        
        
//// 该方法用于执行方块的动画更新
   public void animateTick(BlockState , Level , BlockPos , RandomSource ) {
       // 该方法为空，没有具体的实现
       //低于传送门
   }
        
        
//// 该方法用于销毁方块
    void destroy(LevelAccessor , BlockPos , BlockState ) {
        
   }
        
        
/// 该方法用于获取方块掉落的物品堆列表
     List<ItemStack> getDrops {
       //// 创建一个掉落参数构建器，并设置掉落上下文的相关参数
 
       //调用方块状态的getDrops方法获取物品堆列表，并返回结果
  
   }
        
//// 该方法用于获取方块掉落的物品堆列表，包括了更多的参数
   public static List<ItemStack> getDrops(
      BlockState , ServerLevel , BlockPos ,  BlockEntity ,  Entity , ItemStack 
   ) {
       // 创建一个掉落参数构建器，并设置掉落上下文的相关参数
    
       //// 调用方块状态的getDrops方法获取物品堆列表，并返回结果
    
   }
//// 该方法用于让方块掉落资源
   public static void dropResources(BlockState , Level , BlockPos ) {

   }
        
//// 该方法用于让方块掉落资源
   public static void dropResources(BlockState , LevelAccessor , BlockPos ,  BlockEntity ) {
  
   }
        
        
//// 重载方法：该方法用于让方块掉落资源，并指定是否掉落经验球
   public static void dropResources(
      BlockState , Level , BlockPos , @Nullable BlockEntity , @Nullable Entity , ItemStack 
   ) {
       //// 调用原始的dropResources方法，并设置dropXp参数为true
   }
        
        
    ///// 重载方法：该方法用于让方块掉落资源，并指定是否掉落经验球
     void dropResources(BlockState , Level , BlockPos , @Nullable BlockEntity , @Nullable Entity  ItemStack , boolean dropXp) {
   }
        
        
//// 该方法用于在指定位置生成一个掉落物实体
    void popResource(Level , BlockPos , ItemStack ) {
       //// 计算掉落物的生成位置
    
       //// 调用popResource方法生成掉落物实体并添加到世界中
      
   }
        //这里说的乱，等我之后有时间重新配音，先用旧视频的配音。
        //掉落延迟是玩家拾取的间隔
        // 这里的就是在此方块生成掉落物
        
//// 该方法用于在指定位置的指定方向生成一个掉落物实体
   public static void popResourceFromFace(Level , BlockPos , Direction , ItemStack ) {
       /// 根据方向计算掉落物的生成位置
     
      //// 调用popResource方法生成掉落物实体并添加到世界中
       
   }
        
        
// 该方法用于生成掉落物实体并添加到世界中
   private static void popResource(Level , Supplier<ItemEntity> , ItemStack ) 
   }
        
        
// 该方法用于在指定位置生成经验球实体
   public void popExperience(ServerLevel , BlockPos , int ) {

   }


// 该方法返回方块的爆炸抗性
   @Deprecated //Forge: Use more sensitive version
     getExplosionResistance() {
     
   }
   // 该方法在方块被爆炸时调用
   public void wasExploded {
   }


// 该方法在实体踩踏方块时调用
   public void stepOn(Level , BlockPos , BlockState , Entity ) {
   }


// 该方法用于在放置方块时获取方块的状态
   @Nullable
   public BlockState getStateForPlacement(BlockPlaceContext ) {
   }


// 该方法在玩家破坏方块时调用
   public void playerDestroy(Level , Player , BlockPos , BlockState , @Nullable BlockEntity , ItemStack ) {

       //// 调用dropResources方法丢弃方块的资源（不包括经验球）
   }


//// 该方法在方块被放置时调用
    void setPlacedBy(Level , BlockPos , BlockState , @Nullable LivingEntity , ItemStack ) {
   }


/// 该方法判断方块是否可以重生
    boolean isPossibleToRespawnInThis(BlockState ) {
       //// 如果方块不是实心且不是液体，则可以重生
   
   }


//// 该方法返回方块的名称
   public MutableComponent getName() {
      
   }


// 该方法返回方块的描述标识
   public String getDescriptionId() {
    
   }


// 该方法在实体掉落到方块上时调用
   public void fallOn(Level , BlockState , BlockPos , Entity , float ) {
       // 使实体受到摔落伤害
   
   }


// 该方法在实体掉落到方块上后更新实体状态
   public void updateEntityAfterFallOn(BlockGetter , Entity ) {
      
   }


// 该方法返回方块的克隆itemstack
   @Deprecated //Forge: Use more sensitive version
    ItemStack getCloneItemStack(BlockGetter , BlockPos , BlockState ) {
   }

// 该方法返回方块的摩擦系数
    float getFriction() {
      
   }

// 该方法返回方块的速度因子
   public float getSpeedFactor() {
 
   }

// 该方法返回方块的跳跃因子
   public float getJumpFactor() {
     
   }

// 该方法在方块被破坏时生成破坏粒子效果
   protected void spawnDestroyParticles(Level , Player , BlockPos , BlockState ) {
    
   }


//// 该方法在玩家即将破坏方块时调用
   public void playerWillDestroy(Level, BlockPos , BlockState , Player ) {
       //// 生成破坏粒子效果
       //// 如果方块属于GUARDED_BY_PIGLINS标签，则激怒周围的猪灵
// 发送方块销毁的游戏事件
      
   }


// 该方法处理方块与降水的交互（例如雨、雪等
   public void handlePrecipitation(BlockState , Level , BlockPos , Biome.Precipitation ) {
   }


// 该方法确定方块在爆炸中是否会掉落
   @Deprecated //Forge: Use more sensitive version
    boolean dropFromExplosion(Explosion ) {
   
       
// 该方法用于创建方块的状态定义
    void createBlockStateDefinition(StateDefinition.Builder<Block, BlockState> ) {
   }
       
       
// 该方法返回方块的状态定义
    StateDefinition<Block, BlockState> getStateDefinition() {
   }
       
       
// 该方法注册方块的默认状态
   protected final void registerDefaultState(BlockState p_49960_) {
   }
        
        
// 该方法返回方块的默认状态
   public final BlockState defaultBlockState() {
   }
        
        
//// 该方法通过另一个方块的状态来设置当前方块的状态
   public final BlockState withPropertiesOf(BlockState ) {
     
    
   }
        
        
// 该方法复制一个方块状态的属性值到另一个方块状态中
   private static <T extends Comparable<T>> BlockState copyProperty(BlockState p_152455_, BlockState , Property<T> ) {
       // 使用另一个方块状态的属性值设置当前方块状态的属性值
   }
        
        
// 该方法返回方块的声音类型
   @Deprecated //Forge: Use more sensitive version {@link IForgeBlockState#getSoundType(IWorldReader, BlockPos, Entity) }
   public SoundType getSoundType(BlockState ) {

   }
        
// 重写的方法，返回方块对应的物品对象
   @Override
   public Item asItem() {

       
// 返回物品对象forge部分删除了
 // Forge: Vanilla caches the items, update with registry replacements.
   }
        
        
// 判断方块是否具有动态形状
   public boolean hasDynamicShape() {
   }
        
        

        
// 添加悬停文本，用于显示方块的信息
   public void appendHoverText(ItemStack p_49816_, @Nullable BlockGetter p_49817_, List<Component> p_49818_, TooltipFlag p_49819_) {
   }
        
        
// 重写的方法，返回当前方块对象
   @Override
   protected Block asBlock() {
      return this;
   }
        
// 重写的方法，判断方块是否能支撑植物
   @Override
   public boolean canSustainPlant(BlockState state, BlockGetter world, BlockPos pos, Direction facing, net.neoforged.neoforge.common.IPlantable plantable) {
       // 获取植物的方块状态和植物类型
     
// 如果植物是仙人掌
  
// 如果植物是甘蔗，当前方块也是甘蔗
     
// 如果植物是 BushBlock，并且可以放置在当前方块上
      
// 根据植物类型进行判断
   
   }
        

  

// 尝试掉落经验的方法，接受服务器级别、方块坐标、物品堆叠和经验提供者作为参数
    void tryDropExperience(ServerLevel , BlockPos , ItemStack , IntProvider ) 
     
  
   }
// 第一个方块状态、第二个方块状态和方向的私有字段
// 比较的方块，可以自己去看相关内容
   public static final class BlockStatePairKey {

// 重写的 equals 方法，判断两个 BlockStatePairKey 对象是否相等
      @Override
      public boolean equals(Object ) {
 
}

```

