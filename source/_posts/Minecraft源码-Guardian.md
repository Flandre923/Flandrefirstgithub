---
title: Minecraft源码-Guardian
date: 2023-11-27 20:35:51
tags:
- 我的世界
- 源码
- Jａｖａ
cover: https://view.moezx.cc/images/2017/11/24/InnocentmurdererPID53630844by212a.jpg
---

# 



# Guardian 类

```java

public class Guardian extends Monster {
    //这是一个保护的静态常量，表示守卫者攻击的持续时间，单位是刻（1刻=0.05秒），所以80刻就是4秒
   protected static final int ATTACK_TIME = 80;
    //这是一个私有的静态常量，表示守卫者是否在移动的数据标识符，用于同步实体数据，它的类型是布尔值，它的值由SynchedEntityData类的defineId方法根据守卫者类和布尔值序列化器生成
   private static final EntityDataAccessor<Boolean> DATA_ID_MOVING = SynchedEntityData.defineId(Guardian.class, EntityDataSerializers.BOOLEAN);
    //这是一个私有的静态常量，表示守卫者攻击目标的数据标识符，用于同步实体数据，它的类型是整数，它的值由SynchedEntityData类的defineId方法根据守卫者类和整数序列化器生成
   private static final EntityDataAccessor<Integer> DATA_ID_ATTACK_TARGET = SynchedEntityData.defineId(Guardian.class, EntityDataSerializers.INT);
    //这是一个私有的浮点数变量，表示守卫者尾巴的动画状态，它的值会随着时间变化而变化，用于客户端渲染
   private float clientSideTailAnimation;
    //这是一个私有的浮点数变量，表示守卫者尾巴的上一个动画状态，它的值会随着时间变化而变化，用于客户端渲染
   private float clientSideTailAnimationO;
    //这是一个私有的浮点数变量，表示守卫者尾巴的动画速度，它的值会随着时间变化而变化，用于客户端渲染
   private float clientSideTailAnimationSpeed;
    //这是一个私有的浮点数变量，表示守卫者身上的尖刺的动画状态，它的值会随着时间变化而变化，用于客户端渲染
   private float clientSideSpikesAnimation;
    //这是一个私有的浮点数变量，表示守卫者身上的尖刺的上一个动画状态，它的值会随着时间变化而变化，用于客户端渲染
   private float clientSideSpikesAnimationO;
    //这是一个私有的可空的生物实体变量，表示守卫者的攻击目标，它的值会根据守卫者的行为而变化，用于客户端渲染
   @Nullable
   private LivingEntity clientSideCachedAttackTarget;
    //这是一个私有的整数变量，表示守卫者的攻击时间，它的值会根据守卫者的行为而变化，用于客户端渲染
   private int clientSideAttackTime;
    //这是一个私有的布尔值变量，表示守卫者是否触碰到了地面，它的值会根据守卫者的位置而变化，用于客户端渲染
   private boolean clientSideTouchedGround;
    //这是一个受保护的可空的随机漫步目标变量，表示守卫者的移动目标，它的值会根据守卫者的行为而变化，用于服务端控制
   @Nullable
   protected RandomStrollGoal randomStrollGoal;

    ////这是一个公共的构造方法，它接受两个参数，一个是守卫者的实体类型，一个是守卫者所在的世界，它们的类型分别是EntityType<? extends Guardian>和Level
   public Guardian(EntityType<? extends Guardian> p_32810_, Level p_32811_) {
       //这是一个调用父类的构造方法，也就是水生生物类的构造方法，它把两个参数传递给父类，用于初始化守卫者的基本属性
      super(p_32810_, p_32811_);
       //这是一个设置守卫者的经验值奖励的语句，守卫者的经验值奖励是10，这意味着当玩家杀死守卫者时，可以获得10点经验值
      this.xpReward = 10;
       //这是一个设置守卫者的寻路惩罚的语句，守卫者的寻路惩罚是一个浮点数数组，用于表示守卫者在不同类型的方块上移动的难易程度，这里把水方块的寻路惩罚设置为0.0F，表示守卫者在水中移动没有任何障碍
      this.setPathfindingMalus(BlockPathTypes.WATER, 0.0F);
       ////这是一个创建守卫者的移动控制器的语句，守卫者的移动控制器是一个内部类，用于控制守卫者的移动行为，它的构造方法接受一个守卫者实体作为参数
      this.moveControl = new Guardian.GuardianMoveControl(this);
       //这是一个初始化守卫者尾巴的动画状态的语句，守卫者尾巴的动画状态是一个随机的浮点数，用于客户端渲染
      this.clientSideTailAnimation = this.random.nextFloat();
       //这是一个初始化守卫者尾巴的上一个动画状态的语句，守卫者尾巴的上一个动画状态是和当前动画状态相同的浮点数，用于客户端渲染
      this.clientSideTailAnimationO = this.clientSideTailAnimation;
   }
//这是一个覆盖父类的方法，它的作用是注册守卫者的目标，它没有参数，也没有返回值
   @Override
   protected void registerGoals() {
       //这是一个创建一个朝向限制区域移动的目标的语句，守卫者的限制区域是一个水下的方块区域，守卫者会在这个区域内活动，这个目标的参数是守卫者实体和移动速度，分别是this和1.0
      MoveTowardsRestrictionGoal movetowardsrestrictiongoal = new MoveTowardsRestrictionGoal(this, 1.0);
       //这是一个创建一个随机漫步的目标的语句，守卫者会在限制区域内随机移动，这个目标的参数是守卫者实体，移动速度和执行间隔，分别是this，1.0和80
      this.randomStrollGoal = new RandomStrollGoal(this, 1.0, 80);
       //这是一个把守卫者的攻击目标添加到目标选择器的语句，守卫者的攻击目标是一个内部类，用于控制守卫者的攻击行为，它的构造方法接受一个守卫者实体作为参数，这个目标的优先级是4，表示它比其他目标更重要
      this.goalSelector.addGoal(4, new Guardian.GuardianAttackGoal(this));
       //这是一个把朝向限制区域移动的目标添加到目标选择器的语句，这个目标的优先级是5，表示它比随机漫步的目标更重要，但比攻击目标更次要
      this.goalSelector.addGoal(5, movetowardsrestrictiongoal);
       //这是一个把随机漫步的目标添加到目标选择器的语句，这个目标的优先级是7，表示它比其他目标更次要
      this.goalSelector.addGoal(7, this.randomStrollGoal);
       //这是一个把看向玩家的目标添加到目标选择器的语句，守卫者会看向附近的玩家实体，这个目标的参数是守卫者实体，玩家类，和最大距离，分别是this，Player.class和8.0F，这个目标的优先级是8，表示它和看向其他守卫者的目标一样重要
      this.goalSelector.addGoal(8, new LookAtPlayerGoal(this, Player.class, 8.0F));
       //这是一个把看向其他守卫者的目标添加到目标选择器的语句，守卫者会看向附近的其他守卫者实体，这个目标的参数是守卫者实体，守卫者类，最大距离，和概率，分别是this，Guardian.class，12.0F和0.01F，这个目标的优先级是8，表示它和看向玩家的目标一样重要
      this.goalSelector.addGoal(8, new LookAtPlayerGoal(this, Guardian.class, 12.0F, 0.01F));
       //这是一个把随机看向周围的目标添加到目标选择器的语句，守卫者会随机改变自己的视角，这个目标的参数是守卫者实体，也就是this，这个目标的优先级是9，表示它比其他目标更次要
      this.goalSelector.addGoal(9, new RandomLookAroundGoal(this));
       //这是一个设置随机漫步的目标的标志的语句，守卫者的目标的标志是一个枚举集合，用于表示守卫者在执行目标时的状态，这里把随机漫步的目标的标志设置为移动和看向，表示守卫者在随机漫步时会移动自己的位置和视角
      this.randomStrollGoal.setFlags(EnumSet.of(Goal.Flag.MOVE, Goal.Flag.LOOK));
       //这是一个设置朝向限制区域移动的目标的标志的语句，这里把朝向限制区域移动的目标的标志也设置为移动和看向，表示守卫者在朝向限制区域移动时也会移动自己的位置和视角
      movetowardsrestrictiongoal.setFlags(EnumSet.of(Goal.Flag.MOVE, Goal.Flag.LOOK));
       //这是一个把最近的可攻击目标添加到目标选择器的语句，守卫者会寻找附近的可攻击的生物实体，这个目标的参数是守卫者实体，生物类，最大距离，是否需要视线，是否需要忘记，和攻击选择器，分别是this，LivingEntity.class，10，true，false，和new Guardian.GuardianAttackSelector(this)，攻击选择器是一个内部类，用于判断守卫者是否可以攻击某个实体，这个目标的优先级是1，表示它比其他目标更重要
      this.targetSelector.addGoal(1, new NearestAttackableTargetGoal<>(this, LivingEntity.class, 10, true, false, new Guardian.GuardianAttackSelector(this)));
   }

   public static AttributeSupplier.Builder createAttributes() {
      return Monster.createMonsterAttributes()
         .add(Attributes.ATTACK_DAMAGE, 6.0)
         .add(Attributes.MOVEMENT_SPEED, 0.5)
         .add(Attributes.FOLLOW_RANGE, 16.0)
         .add(Attributes.MAX_HEALTH, 30.0);
   }
//这是一个覆盖父类的方法，它的作用是创建守卫者的导航，它接受一个参数，就是守卫者所在的世界，它的类型是Level，它返回一个导航的对象，它的类型是PathNavigation
   @Override
   protected PathNavigation createNavigation(Level p_32846_) {
       ////这是一个返回一个水中绑定的导航的语句，守卫者的导航是一个水中绑定的导航，用于控制守卫者在水中的移动，它的构造方法接受两个参数，一个是守卫者实体，一个是守卫者所在的世界，分别是this和p_32846_
      return new WaterBoundPathNavigation(this, p_32846_);
   }
//这是一个覆盖父类的方法，它的作用是定义守卫者的同步数据，守卫者的同步数据是一些用于在客户端和服务端之间同步守卫者的状态的数据，它没有参数，也没有返回值
   @Override
   protected void defineSynchedData() {
       //这是一个调用父类的方法，它的作用是定义守卫者的父类的同步数据，也就是水生生物类的同步数据，它没有参数，也没有返回值
      super.defineSynchedData();
       //这是一个定义守卫者是否在移动的同步数据的语句，守卫者是否在移动的同步数据是一个布尔值，它的初始值是false，表示守卫者默认不在移动，它的数据标识符是DATA_ID_MOVING，它是一个私有的静态常量
      this.entityData.define(DATA_ID_MOVING, false);
       //这是一个定义守卫者攻击目标的同步数据的语句，守卫者攻击目标的同步数据是一个整数，它的初始值是0，表示守卫者默认没有攻击目标，它的数据标识符是DATA_ID_ATTACK_TARGET，它也是一个私有的静态常量
      this.entityData.define(DATA_ID_ATTACK_TARGET, 0);
   }
//这是一个覆盖父类的方法，它的作用是判断守卫者是否可以在水下呼吸，它没有参数，它返回一个布尔值，表示守卫者是否可以在水下呼吸
   @Override
   public boolean canBreatheUnderwater() {
      return true;
   }
// 获取实体的类型为水生
   @Override
   public MobType getMobType() {
      return MobType.WATER;
   }
// 检查实体是否在移动
   public boolean isMoving() {
      return this.entityData.get(DATA_ID_MOVING);
   }
// 设置实体的移动状态
   void setMoving(boolean p_32862_) {
      this.entityData.set(DATA_ID_MOVING, p_32862_);
   }
// 获取攻击持续时间
   public int getAttackDuration() {
      return 80;
   }
// 设置实体的主动攻击目标
   void setActiveAttackTarget(int p_32818_) {
      this.entityData.set(DATA_ID_ATTACK_TARGET, p_32818_);
   }
// 检查是否存在活跃的攻击目标
   public boolean hasActiveAttackTarget() {
      return this.entityData.get(DATA_ID_ATTACK_TARGET) != 0;
   }
//  获取活跃的攻击目标，可能为 null
   @Nullable
   public LivingEntity getActiveAttackTarget() {
      if (!this.hasActiveAttackTarget()) { // 没有攻击目标
         return null;
      } else if (this.level().isClientSide) {// 有攻击目标。是客户端
         if (this.clientSideCachedAttackTarget != null) {// 客户端攻击目标不为空
            return this.clientSideCachedAttackTarget;//返回客户端攻击目标
         } else {//客户端攻击目标为空
            Entity entity = this.level().getEntity(this.entityData.get(DATA_ID_ATTACK_TARGET));//获得攻击目标
            if (entity instanceof LivingEntity) {//如果攻击目标是实体
               this.clientSideCachedAttackTarget = (LivingEntity)entity;//赋值
               return this.clientSideCachedAttackTarget;// 返回
            } else {
               return null;
            }
         }
      } else {//又目标切
         return this.getTarget();
      }
   }
// 当同步数据更新时触发的方法
   @Override
   public void onSyncedDataUpdated(EntityDataAccessor<?> p_32834_) {
      super.onSyncedDataUpdated(p_32834_);// 
      if (DATA_ID_ATTACK_TARGET.equals(p_32834_)) {// 当目标不一致时候更新
         this.clientSideAttackTime = 0;// 客户端攻击事件设置为0
         this.clientSideCachedAttackTarget = null;// 攻击目标设置为空
      }
   }
// 获取环境声音的播放间隔时间
   @Override
   public int getAmbientSoundInterval() {
      return 160;
   }
// 获取环境声音
   @Override
   protected SoundEvent getAmbientSound() {
      return this.isInWaterOrBubble() ? SoundEvents.GUARDIAN_AMBIENT : SoundEvents.GUARDIAN_AMBIENT_LAND;
   }
// 获取受伤声音
   @Override
   protected SoundEvent getHurtSound(DamageSource p_32852_) {
      return this.isInWaterOrBubble() ? SoundEvents.GUARDIAN_HURT : SoundEvents.GUARDIAN_HURT_LAND;
   }
// 获取死亡声音
   @Override
   protected SoundEvent getDeathSound() {
      return this.isInWaterOrBubble() ? SoundEvents.GUARDIAN_DEATH : SoundEvents.GUARDIAN_DEATH_LAND;
   }
// 获取移动时事件
   @Override
   protected Entity.MovementEmission getMovementEmission() {
      return Entity.MovementEmission.EVENTS;
   }
// 获取站立时的眼睛高度
   @Override
   protected float getStandingEyeHeight(Pose p_32843_, EntityDimensions p_32844_) {
      return p_32844_.height * 0.5F;
   }
// 获取走到对应方块的代价方法
   @Override
   public float getWalkTargetValue(BlockPos p_32831_, LevelReader p_32832_) {
      return p_32832_.getFluidState(p_32831_).is(FluidTags.WATER)// 是否是水方块
         ? 10.0F + p_32832_.getPathfindingCostFromLightLevels(p_32831_)//是水方块返回代价
         : super.getWalkTargetValue(p_32831_, p_32832_);// 不是水方块
   }

   @Override
   public void aiStep() {
      if (this.isAlive()) { // 本生物存活 
         if (this.level().isClientSide) {// 是客户端
            this.clientSideTailAnimationO = this.clientSideTailAnimation; // 当前尾巴动画存储起来
            if (!this.isInWater()) {  // 不在水中
               this.clientSideTailAnimationSpeed = 2.0F; // 尾巴播放速度设置
               Vec3 vec3 = this.getDeltaMovement();// 获得移动状态
               if (vec3.y > 0.0 && this.clientSideTouchedGround && !this.isSilent()) {// 在上升 并且 没有在地面上， 并且 不是静音的
                  this.level().playLocalSound(this.getX(), this.getY(), this.getZ(), this.getFlopSound(), this.getSoundSource(), 1.0F, 1.0F, false);//播放声音
               }

               this.clientSideTouchedGround = vec3.y < 0.0 && this.level().loadedAndEntityCanStandOn(this.blockPosition().below(), this);//在下降 并且 下一个方块是可以站立的。
            } else if (this.isMoving()) {// 在水中 移动
               if (this.clientSideTailAnimationSpeed < 0.5F) {// 尾巴动画播放速度<0.5f
                  this.clientSideTailAnimationSpeed = 4.0F;// 设置动画播放速度
               } else {// 尾巴播放速度 > 0.5f
                  this.clientSideTailAnimationSpeed += (0.5F - this.clientSideTailAnimationSpeed) * 0.1F;//尾巴播放速度设置为
               }
            } else {//即不再水中，也不再移动
               this.clientSideTailAnimationSpeed += (0.125F - this.clientSideTailAnimationSpeed) * 0.2F;//设置尾巴播放速度
            }

            this.clientSideTailAnimation += this.clientSideTailAnimationSpeed;
            this.clientSideSpikesAnimationO = this.clientSideSpikesAnimation;//保存实体齿状动画
            if (!this.isInWaterOrBubble()) {// 不再水中和气泡中
               this.clientSideSpikesAnimation = this.random.nextFloat();//刺动画设置一个随机时间
            } else if (this.isMoving()) {// 在移动 ，调整动画时间
               this.clientSideSpikesAnimation += (0.0F - this.clientSideSpikesAnimation) * 0.25F;
            } else {// 
               this.clientSideSpikesAnimation += (1.0F - this.clientSideSpikesAnimation) * 0.06F;
            }

            if (this.isMoving() && this.isInWater()) { // 在移动并且在水中
               Vec3 vec31 = this.getViewVector(0.0F);// 获得实体视线方向

               for(int i = 0; i < 2; ++i) {//添加气泡的粒子效果
                  this.level()
                     .addParticle(
                        ParticleTypes.BUBBLE,
                        this.getRandomX(0.5) - vec31.x * 1.5,
                        this.getRandomY() - vec31.y * 1.5,
                        this.getRandomZ(0.5) - vec31.z * 1.5,
                        0.0,
                        0.0,
                        0.0
                     );
               }
            }

            if (this.hasActiveAttackTarget()) {// 在攻击
               if (this.clientSideAttackTime < this.getAttackDuration()) {//攻击时间小于间隔时间
                  ++this.clientSideAttackTime;
               }

               LivingEntity livingentity = this.getActiveAttackTarget();//获得攻击对象
               if (livingentity != null) {//实体不null
                  this.getLookControl().setLookAt(livingentity, 90.0F, 90.0F);//实体视线锁定
                  this.getLookControl().tick();//调用控制器的tick
                  double d5 = (double)this.getAttackAnimationScale(0.0F);//获得攻击动画的缩放比例
                  double d0 = livingentity.getX() - this.getX();//获得 相对位置
                  double d1 = livingentity.getY(0.5) - this.getEyeY();
                  double d2 = livingentity.getZ() - this.getZ();
                  double d3 = Math.sqrt(d0 * d0 + d1 * d1 + d2 * d2);// 求距离
                  d0 /= d3;
                  d1 /= d3;
                  d2 /= d3;//向量归一化
                  double d4 = this.random.nextDouble()//获得一个随机数

                  while(d4 < d3) {//生成一个从自己到敌人的粒子效果
                     d4 += 1.8 - d5 + this.random.nextDouble() * (1.7 - d5);
                     this.level().addParticle(ParticleTypes.BUBBLE, this.getX() + d0 * d4, this.getEyeY() + d1 * d4, this.getZ() + d2 * d4, 0.0, 0.0, 0.0);//生成粒子效果
                  }
               }
            }
         }

         if (this.isInWaterOrBubble()) {//如果在水中或者气泡中
            this.setAirSupply(300);// 氧气值设置为300 .。。为什么你需要设置氧气值？
         } else if (this.onGround()) {//在地面
            this.setDeltaMovement(//设置移动速度
               this.getDeltaMovement()// 在当前位置上随机移动,y设置0.5，上下跳动
                  .add((double)((this.random.nextFloat() * 2.0F - 1.0F) * 0.4F), 0.5, (double)((this.random.nextFloat() * 2.0F - 1.0F) * 0.4F))
            );
            this.setYRot(this.random.nextFloat() * 360.0F);//随机设置旋转
            this.setOnGround(false);//设置为不再地面上
            this.hasImpulse = true;//是否有冲击力？
         }

         if (this.hasActiveAttackTarget()) {//如果有攻击敌人
            this.setYRot(this.yHeadRot);//转向
         }
      }

      super.aiStep(); //调用父类
   }

   protected SoundEvent getFlopSound() {// 返回跳跃声音
      return SoundEvents.GUARDIAN_FLOP;
   }

   public float getTailAnimation(float p_32864_) {// 返回尾巴动画
      return Mth.lerp(p_32864_, this.clientSideTailAnimationO, this.clientSideTailAnimation);//线性插值计算，根据传入的float和上一个动画和当前的动画
   }

   public float getSpikesAnimation(float p_32866_) {　
      return Mth.lerp(p_32866_, this.clientSideSpikesAnimationO, this.clientSideSpikesAnimation);//线性插值计算，根据传入的float和上一个动画和当前的动画
   }

   public float getAttackAnimationScale(float p_32813_) {// 获得攻击动画
      return ((float)this.clientSideAttackTime + p_32813_) / (float)this.getAttackDuration();// 返回
   }

   public float getClientSideAttackTime() {
      return (float)this.clientSideAttackTime;
   }

   @Override
   public boolean checkSpawnObstruction(LevelReader p_32829_) {//检测出生点
      return p_32829_.isUnobstructed(this);//是否可当前位置生成
   }
// 检查守卫的生成规则
   public static boolean checkGuardianSpawnRules(
      EntityType<? extends Guardian> p_218991_, LevelAccessor p_218992_, MobSpawnType p_218993_, BlockPos p_218994_, RandomSource p_218995_
   ) {
       // 随机数为0生成 或者这个位置能否看到天空
      return (p_218995_.nextInt(20) == 0 || !p_218992_.canSeeSkyFromBelowWater(p_218994_))
         && p_218992_.getDifficulty() != Difficulty.PEACEFUL // 不是和平
         && (p_218993_ == MobSpawnType.SPAWNER || p_218992_.getFluidState(p_218994_).is(FluidTags.WATER)) //刷怪笼 或者在水里
         && p_218992_.getFluidState(p_218994_.below()).is(FluidTags.WATER); // 
   }
// 处理守卫受到伤害的情况
   @Override
   public boolean hurt(DamageSource p_32820_, float p_32821_) {
      if (this.level().isClientSide) { // 是客户端返回
         return false;
      } else { // 是服务端
         if (!this.isMoving() && !p_32820_.is(DamageTypeTags.AVOIDS_GUARDIAN_THORNS) && !p_32820_.is(DamageTypes.THORNS)) {//不是移动 伤害源不是可避免 并且 伤害不是荆棘
            Entity entity = p_32820_.getDirectEntity(); // 获得造成伤害的实体 
            if (entity instanceof LivingEntity livingentity) { // 如果伤害源是活着的实体
               livingentity.hurt(this.damageSources().thorns(this), 2.0F);// 对实体造成伤害
            }
         }

         if (this.randomStrollGoal != null) {//如果不为空
            this.randomStrollGoal.trigger();//触发
         }

         return super.hurt(p_32820_, p_32821_);
      }
   }
// 获得实体头部旋转的最大角度
   @Override
   public int getMaxHeadXRot() {
      return 180;
   }
// 处理实体移动
   @Override
   public void travel(Vec3 p_32858_) {
      if (this.isControlledByLocalInstance() && this.isInWater()) {// 是本地控制 在水中
         this.moveRelative(0.1F, p_32858_); // 向当前方向移动
         this.move(MoverType.SELF, this.getDeltaMovement()); // 实体根据当前移动速度移动
         this.setDeltaMovement(this.getDeltaMovement().scale(0.9)); // 设置当前移动速度的90%
         if (!this.isMoving() && this.getTarget() == null) { // 不再移动并且没有目标
            this.setDeltaMovement(this.getDeltaMovement().add(0.0, -0.005, 0.0));// 设置移动速度
         }
      } else {
         super.travel(p_32858_);
      }
   }
// 获得乘客的附加数值
   @Override
   protected Vector3f getPassengerAttachmentPoint(Entity p_294655_, EntityDimensions p_294519_, float p_295088_) {
      return new Vector3f(0.0F, p_294519_.height + 0.125F * p_295088_, 0.0F);
   }
// 定义守护者的攻击目标
   static class GuardianAttackGoal extends Goal {
      private final Guardian guardian;// 守护者
      private int attackTime;// 攻击时间
      private final boolean elder;//是否elderGuadian

      public GuardianAttackGoal(Guardian p_32871_) {
         this.guardian = p_32871_;
         this.elder = p_32871_ instanceof ElderGuardian;
         this.setFlags(EnumSet.of(Goal.Flag.MOVE, Goal.Flag.LOOK));
      }

      @Override
      public boolean canUse() {// 此goal是否可用
         LivingEntity livingentity = this.guardian.getTarget();//获得攻击目标
         return livingentity != null && livingentity.isAlive();//如果攻击目标存在，不为空
      }
	
      @Override
      public boolean canContinueToUse() {// 是否年长，并且目标不为空，和敌人距离大于9
         return super.canContinueToUse() && (this.elder || this.guardian.getTarget() != null && this.guardian.distanceToSqr(this.guardian.getTarget()) > 9.0);
      }

      @Override
      public void start() {//开始攻击目标
         this.attackTime = -10;//攻击时间
         this.guardian.getNavigation().stop();//导航停止
         LivingEntity livingentity = this.guardian.getTarget();//获得目标
         if (livingentity != null) {//如果不为空
             //看向敌人
            this.guardian.getLookControl().setLookAt(livingentity, 90.0F, 90.0F);
         }
		// 设置为冲击为true
         this.guardian.hasImpulse = true;
      }

      @Override
      public void stop() {//停止攻击
          // 设计攻击目标为0 
         this.guardian.setActiveAttackTarget(0);
          // 设置目标为null
         this.guardian.setTarget(null);
          // 触发随机漫步
         this.guardian.randomStrollGoal.trigger();
      }

      @Override
      public boolean requiresUpdateEveryTick() {
          // 是否需要每tick更新
         return true;
      }

      @Override
      public void tick() {//每tick行为
         LivingEntity livingentity = this.guardian.getTarget();//获得守护者目标
         if (livingentity != null) {//不为空
            this.guardian.getNavigation().stop();//停止导航
            this.guardian.getLookControl().setLookAt(livingentity, 90.0F, 90.0F);//看向敌人
            if (!this.guardian.hasLineOfSight(livingentity)) {//检查是否有路线看到敌人
               this.guardian.setTarget(null);//看不到就设置目标为空
            } else {
               ++this.attackTime;//否则敌人存在，攻击时间++
               if (this.attackTime == 0) {//如果攻击时间为0
                  this.guardian.setActiveAttackTarget(livingentity.getId());//设置活跃攻击敌人的id
                  if (!this.guardian.isSilent()) {//如果不是静默
                      // 广播事件
                     this.guardian.level().broadcastEntityEvent(this.guardian, (byte)21);
                  }
               } else if (this.attackTime >= this.guardian.getAttackDuration()) {// 攻击时间大于持续时间
                  float f = 1.0F; // 根据游戏困难难度设置伤害
                  if (this.guardian.level().getDifficulty() == Difficulty.HARD) {
                     f += 2.0F;
                  }

                  if (this.elder) {
                     f += 2.0F;
                  }
//实体受到伤害
                  livingentity.hurt(this.guardian.damageSources().indirectMagic(this.guardian, this.guardian), f);
                  livingentity.hurt(this.guardian.damageSources().mobAttack(this.guardian), (float)this.guardian.getAttributeValue(Attributes.ATTACK_DAMAGE));// 实体受到伤害
                  this.guardian.setTarget(null);// 设置目标为null
               }

               super.tick();
            }
         }
      }
   }
// 攻击选择器
   static class GuardianAttackSelector implements Predicate<LivingEntity> {
      private final Guardian guardian;

      public GuardianAttackSelector(Guardian p_32879_) {
         this.guardian = p_32879_;
      }

      public boolean test(@Nullable LivingEntity p_32881_) {// 是否攻击
          // 是玩家 或者 是Squid  或者 是 Axolotl 并且 距离小于9
         return (p_32881_ instanceof Player || p_32881_ instanceof Squid || p_32881_ instanceof Axolotl) && p_32881_.distanceToSqr(this.guardian) > 9.0;
      }
   }
// 移动控制器
   static class GuardianMoveControl extends MoveControl {
      private final Guardian guardian;

      public GuardianMoveControl(Guardian p_32886_) {
         super(p_32886_);
         this.guardian = p_32886_;
      }

      @Override
      public void tick() { 
         if (this.operation == MoveControl.Operation.MOVE_TO && !this.guardian.getNavigation().isDone()) { // 是否移动到目标 并且导航没有完成
            Vec3 vec3 = new Vec3(this.wantedX - this.guardian.getX(), this.wantedY - this.guardian.getY(), this.wantedZ - this.guardian.getZ());//获得方向向量
            double d0 = vec3.length();//归一化
            double d1 = vec3.x / d0;
            double d2 = vec3.y / d0;
            double d3 = vec3.z / d0;
             // 计算偏航角
            float f = (float)(Mth.atan2(vec3.z, vec3.x) * 180.0F / (float)Math.PI) - 90.0F;
             //设置转向
            this.guardian.setYRot(this.rotlerp(this.guardian.getYRot(), f, 90.0F));
             // 设置身体转向
            this.guardian.yBodyRot = this.guardian.getYRot();
             // 设置守护者的速度
            float f1 = (float)(this.speedModifier * this.guardian.getAttributeValue(Attributes.MOVEMENT_SPEED));
             // 计算线性插值
            float f2 = Mth.lerp(0.125F, this.guardian.getSpeed(), f1);
             // 设置速度
            this.guardian.setSpeed(f2);
             // 
            double d4 = Math.sin((double)(this.guardian.tickCount + this.guardian.getId()) * 0.5) * 0.05;
            double d5 = Math.cos((double)(this.guardian.getYRot() * (float) (Math.PI / 180.0)));
            double d6 = Math.sin((double)(this.guardian.getYRot() * (float) (Math.PI / 180.0)));
            double d7 = Math.sin((double)(this.guardian.tickCount + this.guardian.getId()) * 0.75) * 0.05;
             // 设置移动方向
            this.guardian.setDeltaMovement(this.guardian.getDeltaMovement().add(d4 * d5, d7 * (d6 + d5) * 0.25 + (double)f2 * d2 * 0.1, d4 * d6));
            LookControl lookcontrol = this.guardian.getLookControl();
             // 获得视线的位置
            double d8 = this.guardian.getX() + d1 * 2.0;
            double d9 = this.guardian.getEyeY() + d2 / d0;
            double d10 = this.guardian.getZ() + d3 * 2.0;
             // 获得向移动的位置
            double d11 = lookcontrol.getWantedX();
            double d12 = lookcontrol.getWantedY();
            double d13 = lookcontrol.getWantedZ();
            if (!lookcontrol.isLookingAtTarget()) {//如果look控制器没处于看目标状态
               d11 = d8;
               d12 = d9;
               d13 = d10;
            }
			// 设置看的位置
            this.guardian.getLookControl().setLookAt(Mth.lerp(0.125, d11, d8), Mth.lerp(0.125, d12, d9), Mth.lerp(0.125, d13, d10), 10.0F, 40.0F);
            this.guardian.setMoving(true);//移动为true
         } else {//移动已完成
            this.guardian.setSpeed(0.0F);//设置速度为0
            this.guardian.setMoving(false);//移动false
         }
      }
   }
}

```

