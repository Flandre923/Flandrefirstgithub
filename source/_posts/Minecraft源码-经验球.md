---
title: Minecraft源码-经验球
date: 2023-11-30 14:55:12
tags:
- 我的世界源码
- 经验球
- Java
cover: https://view.moezx.cc/images/2018/01/15/PID66745952byBison.jpg
---



# ExperienceOrb 经验球

```java

public class ExperienceOrb extends Entity {
    int LIFETIME//经验求生存时间 tick  300S 5min
    int ENTITY_SCAN_PERIOD// 经验求每间隔多少tick扫描一次周围的玩家
    int MAX_FOLLOW_DIST;// 经验球最大的追随玩家的距离
    int ORB_GROUPS_PER_AREA // 每个区域的最多经验球
    double ORB_MERGE_DISTANCE// 每个经验求合并的最小距离
    int age // 经验球的年龄用于消失
    int health// 经验球的生命
   int value // 经验球的数值
    int count// 经验球的数值，当合并时候会增加，影响声音
    Player followingPlayer; // 追随玩家
//创建一个经验求实体
   public ExperienceOrb(Level , double , double , double , int ) {
    //设置了经验球的属性
   }
    
// 从存档中加载实体
   public ExperienceOrb(EntityType<? extends ExperienceOrb> , Level ) {
   }
    
// 重写父类的方法，根据运动发出声音和粒子效果，返回值是一个枚举。
   @Override
   protected Entity.MovementEmission getMovementEmission() {
   }
    

// 每个tick回调方法，
   @Override
   public void tick() {
       //如果在水中或者岩浆中模拟水中和岩浆中的浮力效果
       
// 检测经验求是否和其他方块实体发生碰撞，如果发生了就移动到最近的空间中
      
// 如果当前的tick计时为20的倍数
      
          // 扫描周围的实体
       
// 如果追随玩家存在 玩家死亡 玩家处于观察者模式
    
// 玩家不为空则添加一个想玩家移动的速度

     //模拟不同方块上的速度
       
       // 超出最大存活时间则消失
   }
    
    
// 重写父类的方法，用于获得实体下面的方块
   @Override
   protected BlockPos getBlockPosBelowThatAffectsMyMovement() {
   }
    
    
// 获得扫描附近实体，是否有实体和可以合并的经验求
   private void scanForEntities() {
   }
    
// 公共静态方法，用于在给定世界的位置生成一定数量的经验球
   public static void award(ServerLevel p_147083_, Vec3 p_147084_, int p_147085_) {
      
   }
    
    
// 私有静态的方法 尝试将经验添加到已存在的经验求中，
   private static boolean tryMergeToExisting(ServerLevel p_147097_, Vec3 p_147098_, int p_147099_) {
     
   }
    
    
// 判断是否可以和另一个经验求合并。
   private boolean canMerge(ExperienceOrb p_147087_) {
   }
    
    
// 判断是否可以和另一个经验求合并，id%40为0.并且value数值相等
   private static boolean canMerge(ExperienceOrb p_147089_, int p_147090_, int p_147091_) {
   }
    
    
//  合并
   private void merge(ExperienceOrb p_147101_) {
   }
    
    
// 设置水下移动速度
   private void setUnderwaterMovement() {
   }
    
    
//  溅起水花？
   @Override
   protected void doWaterSplashEffect() {
   }
    
    
// 受到伤害，伤害的来源和伤害的大小
   @Override
   public boolean hurt(DamageSource , float ) {
       //如果是客户端，经验求已经移除，那么就返回false，这是forge的一个修复，处理客户端和服务器不同步的问题。
       // 是否对伤害源免疫
    	// 不免疫则减少hp。为0消失
   }
    
    
// 将经验求的数据存储到一个nbt中
   @Override
   public void addAdditionalSaveData(CompoundTag ) {
   }
// 从nbt中读取数据
   @Override
   public void readAdditionalSaveData(CompoundTag ) {
   }
    
    
// 重写的父类的方法。用于处理经验求和玩家接触
   @Override
   public void playerTouch(Player p_20792_) {
      
   }
    
    
// 修复玩家身上的物品，玩家和经验球的数值
   private int repairPlayerItems(Player p_147093_, int p_147094_) {
       // 附魔助手的方法，随机获得一个玩家身上有的经验修补的物品，参数分别是附魔对象，玩家实体，一个是否损坏的判断器
   }
    
    
// 损伤值转为经验值
   private int durabilityToXp(int ) {
   }
// 经验值转为损伤值
   private int xpToDurability(int ) {
   }
    
    
// 获得数值
   public int getValue() {
   }
    
    
// 根据value返回不同的icon图表
   public int getIcon() {
   }
    
    
//获得合适经验值的方法
   public static int getExperienceValue() {
   }
    
    
	// 经验球是否被攻击
   @Override
   public boolean isAttackable() {
   }

   @Override
   public SoundSource getSoundSource() {
       // 经验球的声音源是环境
   }
}

```

