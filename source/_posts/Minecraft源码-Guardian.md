---
title: Minecraft源码-Guardian
date: 2023-11-27 20:35:51
tags:
- 我的世界
- 源码
- Java
cover: https://view.moezx.cc/images/2017/11/24/InnocentmurdererPID53630844by212a.jpg
---

# 



# Guardian 类

```java

public class Guardian extends Monster {
    //这是一个保护的静态常量，表示守卫者攻击的持续时间，单位是刻（1刻=0.05秒），所以80刻就是4秒
    int ATTACK_TIME

    //这是一个私有的浮点数变量，表示守卫者尾巴的动画状态，它的值会随着时间变化而变化，用于客户端渲染
    clientSideTailAnimation;
    //这是一个私有的浮点数变量，表示守卫者尾巴的上一个动画状态，它的值会随着时间变化而变化，用于客户端渲染
    clientSideTailAnimationO;
    //这是一个私有的浮点数变量，表示守卫者尾巴的动画速度，它的值会随着时间变化而变化，用于客户端渲染
    clientSideTailAnimationSpeed;
    //这是一个私有的浮点数变量，表示守卫者身上的尖刺的动画状态，它的值会随着时间变化而变化，用于客户端渲染
    clientSideSpikesAnimation;
    //这是一个私有的浮点数变量，表示守卫者身上的尖刺的上一个动画状态，它的值会随着时间变化而变化，用于客户端渲染
 clientSideSpikesAnimationO;
    //这是一个私有的可空的生物实体变量，表示守卫者的攻击目标，它的值会根据守卫者的行为而变化，用于客户端渲染

clientSideCachedAttackTarget;
    //这是一个私有的整数变量，表示守卫者的攻击时间，它的值会根据守卫者的行为而变化，用于客户端渲染
clientSideAttackTime;
    //这是一个私有的布尔值变量，表示守卫者是否触碰到了地面，它的值会根据守卫者的位置而变化，用于客户端渲染
 clientSideTouchedGround;
    //这是一个受保护的可空的随机漫步目标变量，表示守卫者的移动目标，它的值会根据守卫者的行为而变化，用于服务端控制
randomStrollGoal;
    

    ////这是一个公共的构造方法，它接受两个参数，一个是守卫者的实体类型，一个是守卫者所在的世界，它们的类型分别是EntityType<? extends Guardian>和Level
    //调用父类的构造方法，初始化守卫者的基本属性，如生命值、攻击力等。
//设置守卫者的经验值奖励，为10。
//设置守卫者的寻路惩罚，在水中移动没有任何障碍。
//创建守卫者的移动控制器，用于控制守卫者的移动行为。
//初始化守卫者尾巴的动画状态，为一个随机的浮点数。
//初始化守卫者尾巴的上一个动画状态，与当前动画状态相同。
   public Guardian(EntityType<? extends Guardian> , Level ) {
     
   }
   
   
//这是一个覆盖父类的方法，它的作用是注册守卫者的目标，它没有参数，也没有返回值
   // 在限制区域内随机漫步
//朝向限制区域移动
//看向附近的玩家和其他守卫者
//随机看向周围
//攻击最近的可攻击的生物实体
   @Override
   protected void registerGoals() {
       
   }
// 设置生物的血量等属性
   public static AttributeSupplier.Builder createAttributes() {
      
   }
    
//这是一个覆盖父类的方法，它的作用是创建守卫者的导航，它接受一个参数，就是守卫者所在的世界，它的类型是Level，它返回一个导航的对象，它的类型是PathNavigation
   @Override
   protected PathNavigation createNavigation(Level ) {
       ////这是一个返回一个水中绑定的导航的语句，守卫者的导航是一个水中绑定的导航，用于控制守卫者在水中的移动，它的构造方法接受两个参数，一个是守卫者实体，一个是守卫者所在的世界，分别是this和p_32846_

   }
    
    
//这是一个覆盖父类的方法，它的作用是定义守卫者的同步数据，守卫者的同步数据是一些用于在客户端和服务端之间同步守卫者的状态的数据，它没有参数，也没有返回值
   @Override
   protected void defineSynchedData() {
   }
    
    
//这是一个覆盖父类的方法，它的作用是判断守卫者是否可以在水下呼吸，它没有参数，它返回一个布尔值，表示守卫者是否可以在水下呼吸
   @Override
   public boolean canBreatheUnderwater() {
   }
    
// 获取实体的类型为水生
   @Override
   public MobType getMobType() {
   }
    
// 检查实体是否在移动
   public boolean isMoving() {

   }
    
// 设置实体的移动状态
   void setMoving(boolean ) {

       
   }
// 获取攻击持续时间
   public int getAttackDuration() {

   }
   
    
    
// 设置实体的主动攻击目标
   void setActiveAttackTarget(int ) {
    
   }
    
    
// 检查是否存在活跃的攻击目标
    boolean hasActiveAttackTarget() {
   }
    
    
//  获取活跃的攻击目标，可能为 null
   @Nullable
    LivingEntity getActiveAttackTarget() {
     
   }
    
    

    
    
// 获取环境声音的播放间隔时间
  
    int getAmbientSoundInterval() {
   }
    
    
// 获取环境声音
   @Override
    SoundEvent getAmbientSound() {
   }
    
// 获取受伤声音
   @Override
   
    SoundEvent getHurtSound(DamageSource p_32852_) {
   }
    
// 获取死亡声音
   @Override
    SoundEvent getDeathSound() {
   }
    
// 获取移动时事件
   @Override
    Entity.MovementEmission getMovementEmission() {
   }
    
// 获取站立时的眼睛高度
   @Override
   protected float getStandingEyeHeight(Pose , EntityDimensions ) {
   }
    
// 获取走到对应方块的代价方法
   @Override
   public float getWalkTargetValue(BlockPos , LevelReader ) {
      
   }
    
//守卫者实体在 AI 步骤中会根据其状态进行不同的动作，例如移动、攻击、生成粒子效果等。
    //如果在水中或气泡中，则设置氧气值为 300。
    //如果在地面上，则在当前位置上随机移动，并随机旋转。
    //如果有攻击目标，则设置旋转方向为攻击目标的方向。
    //如果正在攻击，则生成从自己到攻击目标的粒子效果。
   @Override
   public void aiStep() {
     
   }

   protected SoundEvent getFlopSound() {// 返回跳跃声音
   }

   public float getTailAnimation(float ) {// 返回尾巴动画相关的数值

   public float getSpikesAnimation(float ) {　//尖刺
   }

   public float getAttackAnimationScale(float ) {// 获得攻击动画
   }

   public float getClientSideAttackTime() {
   }

   @Override
   public boolean checkSpawnObstruction(LevelReader p_32829_) {//检测出生点
   }
       
       
// 检查守卫的生成规则
   public static boolean checkGuardianSpawnRules(
      
   }
       
       
// 处理守卫受到伤害的情况
       
       
   @Override
   public boolean hurt(DamageSource p_32820_, float p_32821_) {
      
   }
       
       
// 获得实体头部旋转的最大角度
   @Override
   public int getMaxHeadXRot() {
   }
       
       
// 处理实体移动
   @Override
   public void travel(Vec3 p_32858_) {
// 是本地控制 在水中
      
   
   }
       
  
// 定义守护者的攻击目标
     GuardianAttackGoal extends Goal {
 Guardian guardian;// 守护者
int attackTime;// 攻击时间
 boolean elder;//是否elderGuadian

      public GuardianAttackGoal(Guardian ) {
         

      @Override
      public boolean canUse() {// 此goal是否可用
        //存在敌人则可用
      }
	
      @Override
      public boolean canContinueToUse() {//是否继续调用 是否年长，并且目标不为空，和敌人距离大于9
        
      }

      @Override
      public void start() {//开始执行该goal
       
      }

      @Override
      public void stop() {//停止只从该goal
     
      }

      @Override
      public boolean requiresUpdateEveryTick() {
          // 是否需要每tick更新
      }

      @Override
      public void tick() {//每tick行为
//停止导航，转向目标。
//如果没有视野，则设置目标为空。
//如果攻击时间为 0，则设置活跃攻击目标为目标的 ID，并广播事件。
//如果攻击时间大于持续时间，则向目标造成伤害，并设置目标为空。
      }
   }
         
// 攻击选择器
    class GuardianAttackSelector implements Predicate<LivingEntity> {
\Guardian guardian;

      public GuardianAttackSelector(Guardian \) {
\      }

      public boolean test(@Nullable LivingEntity p_32881_) {// 是否执行test
       
      }
   }
         
// 移动控制器
    class GuardianMoveControl extends MoveControl {
Guardian guardian;

      public GuardianMoveControl(Guardian ) {
      }

      @Override
      public void tick() { 
        //计算目标方向向量，并根据向量计算偏航角。
//设置转向和身体转向。
//计算守卫者的速度，并根据速度线性插值。
//计算移动方向，并设置移动方向。
//计算视线位置，并设置视线位置。
   }
}

```

