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
   private static final int LIFETIME = 6000;//经验求生存时间 tick  300S 5min
   private static final int ENTITY_SCAN_PERIOD = 20;// 经验求每间隔多少tick扫描一次周围的玩家
   private static final int MAX_FOLLOW_DIST = 8;// 经验球最大的追随玩家的距离
   private static final int ORB_GROUPS_PER_AREA = 40; // 每个区域的最多经验球
   private static final double ORB_MERGE_DISTANCE = 0.5;// 每个经验求合并的最小距离
   private int age; // 经验球的年龄用于消失
   private int health = 5;// 经验球的生命
   public int value; // 经验球的数值
   private int count = 1;// 经验球的数值，当合并时候会增加，影响声音
   private Player followingPlayer; // 追随玩家
//创建一个经验求实体
   public ExperienceOrb(Level p_20776_, double p_20777_, double p_20778_, double p_20779_, int p_20780_) {
      this(EntityType.EXPERIENCE_ORB, p_20776_);// 另一个构造方法
      this.setPos(p_20777_, p_20778_, p_20779_);// 设置经验球的位置
      this.setYRot((float)(this.random.nextDouble() * 360.0));//设置经验球的旋转角度 0 到 360
      this.setDeltaMovement(//设置移动速度。 范围再-0.2到0.2 方向XYZ
         (this.random.nextDouble() * 0.2F - 0.1F) * 2.0, this.random.nextDouble() * 0.2 * 2.0, (this.random.nextDouble() * 0.2F - 0.1F) * 2.0
      );
      this.value = p_20780_; // 经验值大小
   }
// 从存档中加载实体
   public ExperienceOrb(EntityType<? extends ExperienceOrb> p_20773_, Level p_20774_) {
      super(p_20773_, p_20774_);
   }
// 重写父类的方法，根据运动发出声音和粒子效果，返回值是一个枚举。
   @Override
   protected Entity.MovementEmission getMovementEmission() {
      return Entity.MovementEmission.NONE;
   }
// 异步同步数据
   @Override
   protected void defineSynchedData() {
   }
// 每个tick回调方法，
   @Override
   public void tick() {
      super.tick();// 父类tick方法
      this.xo = this.getX(); // 记录xyz
      this.yo = this.getY();
      this.zo = this.getZ();
      if (this.isEyeInFluid(FluidTags.WATER)) { // 检查经验球是否在水中
         this.setUnderwaterMovement();// 设置为水中的移动
      } else if (!this.isNoGravity()) { // 如果受重力影响
          // 增加一个向下的加速度
         this.setDeltaMovement(this.getDeltaMovement().add(0.0, -0.03, 0.0));
      }
		// 检查是否在岩浆中
      if (this.level().getFluidState(this.blockPosition()).is(FluidTags.LAVA)) {
         this.setDeltaMovement( // 增加一个xz方向的随机加速度，模拟岩浆浮力 
            (double)((this.random.nextFloat() - this.random.nextFloat()) * 0.2F), 0.2F, (double)((this.random.nextFloat() - this.random.nextFloat()) * 0.2F)
         );
      }
// 检测经验求是否和其他方块实体发生碰撞，如果发生了就移动到最近的空间中
      if (!this.level().noCollision(this.getBoundingBox())) {
         this.moveTowardsClosestSpace(this.getX(), (this.getBoundingBox().minY + this.getBoundingBox().maxY) / 2.0, this.getZ());
      }
// 如果当前的tick计时为20的倍数
      if (this.tickCount % 20 == 1) {
          // 扫描周围的实体
         this.scanForEntities();
      }
// 如果追随玩家存在 玩家死亡 玩家处于观察者模式
      if (this.followingPlayer != null && (this.followingPlayer.isSpectator() || this.followingPlayer.isDeadOrDying())) {
         this.followingPlayer = null;
      }
// 
      if (this.followingPlayer != null) {
          // 获得玩家和当前实体的向量
         Vec3 vec3 = new Vec3(
            this.followingPlayer.getX() - this.getX(),
            this.followingPlayer.getY() + (double)this.followingPlayer.getEyeHeight() / 2.0 - this.getY(),
            this.followingPlayer.getZ() - this.getZ()
         );
          // 获得向量模长（距离）
         double d0 = vec3.lengthSqr();
         if (d0 < 64.0) {//如果距离小于64
            double d1 = 1.0 - Math.sqrt(d0) / 8.0;
            this.setDeltaMovement(this.getDeltaMovement().add(vec3.normalize().scale(d1 * d1 * 0.1)));// 给经验球设置一个该方向的加速度
         }
      }

      this.move(MoverType.SELF, this.getDeltaMovement());// 父类方法，根据加速度移动
      float f = 0.98F; // 经验球速度的衰减速度，每次减少2%
      if (this.onGround()) {// 判断是否在地面上
         BlockPos pos = getBlockPosBelowThatAffectsMyMovement();// 获得位置
         f = this.level().getBlockState(pos).getFriction(this.level(), pos, this) * 0.98F;//获得摩擦因素 和f 进行计算 得到新的f
      }
		// 根据f设置速度
      this.setDeltaMovement(this.getDeltaMovement().multiply((double)f, 0.98, (double)f));
      if (this.onGround()) {//如果在地面上就设置一个向上的速度
         this.setDeltaMovement(this.getDeltaMovement().multiply(1.0, -0.9, 1.0));
      }
	// 年龄++
      ++this.age;
      if (this.age >= 6000) {
         this.discard();// 如果超时，则消失
      }
   }
// 重写父类的方法，用于获得实体下面的方块
   @Override
   protected BlockPos getBlockPosBelowThatAffectsMyMovement() {
      return this.getOnPos(0.999999F);
   }
// 获得扫描附近实体，是否有实体和可以合并的经验求
   private void scanForEntities() {
       // 8 格范围内是否有玩家
      if (this.followingPlayer == null || this.followingPlayer.distanceToSqr(this) > 64.0) {
          // 有就设置为最近的玩家
         this.followingPlayer = this.level().getNearestPlayer(this, 8.0);
      }
// 获得周围的碰撞体积扩大0.5倍后的所有可合并经验求实体
      if (this.level() instanceof ServerLevel) {
         for(ExperienceOrb experienceorb : this.level()
            .getEntities(EntityTypeTest.forClass(ExperienceOrb.class), this.getBoundingBox().inflate(0.5), this::canMerge)) {
            this.merge(experienceorb);//合并
         }
      }
   }
// 公共静态方法，用于在给定世界的位置生成一定数量的经验球
   public static void award(ServerLevel p_147083_, Vec3 p_147084_, int p_147085_) {
      while(p_147085_ > 0) {
         int i = getExperienceValue(p_147085_);// 获得一个合适的数值
         p_147085_ -= i;// 更新剩下的数值
         if (!tryMergeToExisting(p_147083_, p_147084_, i)) {//尝试和已存在的合并
            p_147083_.addFreshEntity(new ExperienceOrb(p_147083_, p_147084_.x(), p_147084_.y(), p_147084_.z(), i));// 不能合并就生成新的
         }
      }
   }
// 私有静态的方法 尝试将经验添加到已存在的经验求中，
   private static boolean tryMergeToExisting(ServerLevel p_147097_, Vec3 p_147098_, int p_147099_) {
       // 创建一个轴对齐的包装盒，根据给定的位置和大小确定一个立体范围，位置，高度，宽度，深度
      AABB aabb = AABB.ofSize(p_147098_, 1.0, 1.0, 1.0);
      int i = p_147097_.getRandom().nextInt(40); // 获得一个随机数
      List<ExperienceOrb> list = p_147097_.getEntities(EntityTypeTest.forClass(ExperienceOrb.class), aabb, p_147081_ -> canMerge(p_147081_, i, p_147099_)); // 获得aabb内所有的可以合并经验求的经验求
      if (!list.isEmpty()) {// 如果不为空则可合并
         ExperienceOrb experienceorb = list.get(0);// 获得第一个
         ++experienceorb.count;// 增加count表示合并了一个经验求
         experienceorb.age = 0;// age 刷新
         return true;
      } else {
         return false;
      }
   }
// 判断是否可以和另一个经验求合并。
   private boolean canMerge(ExperienceOrb p_147087_) {
      return p_147087_ != this && canMerge(p_147087_, this.getId(), this.value);
   }
// 判断是否可以和另一个经验求合并，id%40为0.并且value数值相等
   private static boolean canMerge(ExperienceOrb p_147089_, int p_147090_, int p_147091_) {
      return !p_147089_.isRemoved() && (p_147089_.getId() - p_147090_) % 40 == 0 && p_147089_.value == p_147091_;
   }
//  合并
   private void merge(ExperienceOrb p_147101_) {
      this.count += p_147101_.count;
      this.age = Math.min(this.age, p_147101_.age);
      p_147101_.discard();// 另一个消失
   }
// 设置水下移动速度
   private void setUnderwaterMovement() {
      Vec3 vec3 = this.getDeltaMovement();
       // 设置水的阻力和水的浮力
      this.setDeltaMovement(vec3.x * 0.99F, Math.min(vec3.y + 5.0E-4F, 0.06F), vec3.z * 0.99F);
   }
//  溅起水花？
   @Override
   protected void doWaterSplashEffect() {
   }
// 受到伤害，伤害的来源和伤害的大小
   @Override
   public boolean hurt(DamageSource p_20785_, float p_20786_) {
       //如果是客户端，经验求已经移除，那么就返回false，这是forge的一个修复，处理客户端和服务器不同步的问题。
      if (this.level().isClientSide || this.isRemoved()) return false; //Forge: Fixes MC-53850
       // 是否对伤害源免疫
      if (this.isInvulnerableTo(p_20785_)) {
         return false;
      } else if (this.level().isClientSide) {//客户端受伤害没有逻辑
         return true;
      } else {
         this.markHurt();// 受到伤害标记
         this.health = (int)((float)this.health - p_20786_); // 生命值减少
         if (this.health <= 0) {// 如果生命值<0
            this.discard();
         }

         return true;
      }
   }
// 将经验求的数据存储到一个nbt中
   @Override
   public void addAdditionalSaveData(CompoundTag p_20796_) {
      p_20796_.putShort("Health", (short)this.health);
      p_20796_.putShort("Age", (short)this.age);
      p_20796_.putShort("Value", (short)this.value);
      p_20796_.putInt("Count", this.count);
   }
// 从nbt中读取数据
   @Override
   public void readAdditionalSaveData(CompoundTag p_20788_) {
      this.health = p_20788_.getShort("Health");
      this.age = p_20788_.getShort("Age");
      this.value = p_20788_.getShort("Value");
      this.count = Math.max(p_20788_.getInt("Count"), 1);
   }
// 重写的父类的方法。用于处理经验求和玩家接触
   @Override
   public void playerTouch(Player p_20792_) {
      if (!this.level().isClientSide) { // 处于服务器
         if (p_20792_.takeXpDelay == 0) {// 检测玩家拾取经验球的延迟是否为0，
            if (net.neoforged.neoforge.common.NeoForge.EVENT_BUS.post(new net.neoforged.neoforge.event.entity.player.PlayerXpEvent.PickupXp(p_20792_, this)).isCanceled()) return; // Forge拓展，如果玩家拾取经验求事件被取消则直接返回
            p_20792_.takeXpDelay = 2;// 设置拾取延迟
            p_20792_.take(this, 1);// 拾取经验求，以及拾取数量
            int i = this.repairPlayerItems(p_20792_, this.value);//  修复玩家身上的物品 剩余经验值i
            if (i > 0) { // i > 0 
               p_20792_.giveExperiencePoints(i); // 玩家获得经验值
            }

            --this.count;//  经验求count -- （为什么要在这种地方搞这样的细致的内容，没啥实质反馈。不亏需要一个mod来优化 经验值逻辑。 吐槽
            if (this.count == 0) { // 如果 count =0 
               this.discard(); // 消失
            }
         }
      }
   }
// 修复玩家身上的物品，玩家和经验球的数值
   private int repairPlayerItems(Player p_147093_, int p_147094_) {
      Entry<EquipmentSlot, ItemStack> entry = EnchantmentHelper.getRandomItemWith(Enchantments.MENDING, p_147093_, ItemStack::isDamaged); // 附魔助手的方法，随机获得一个玩家身上有的经验修补的物品，参数分别是附魔对象，玩家实体，一个是否损坏的判断器
      if (entry != null) { // 
         ItemStack itemstack = entry.getValue(); // 获得数值
         int i = Math.min((int) (this.value * itemstack.getXpRepairRatio()), itemstack.getDamageValue());// 经验球的数值*比例 和 物品损伤数值中较小的那个
          // 物品设置损坏值 为 当前的损坏值 - i
         itemstack.setDamageValue(itemstack.getDamageValue() - i); 
         int j = p_147094_ - this.durabilityToXp(i);//剩余经验值的数量
         return j > 0 ? this.repairPlayerItems(p_147093_, j) : 0;//递归调用修复物品
      } else {
         return p_147094_; // 没有可修复的返回
      }
   }
// 损伤值转为经验值
   private int durabilityToXp(int p_20794_) {
      return p_20794_ / 2;
   }
// 经验值转为损伤值
   private int xpToDurability(int p_20799_) {
      return p_20799_ * 2;
   }
// 获得数值
   public int getValue() {
      return this.value;
   }
// 根据value返回不同的icon图表
   public int getIcon() {
      if (this.value >= 2477) {
         return 10;
      } else if (this.value >= 1237) {
         return 9;
      } else if (this.value >= 617) {
         return 8;
      } else if (this.value >= 307) {
         return 7;
      } else if (this.value >= 149) {
         return 6;
      } else if (this.value >= 73) {
         return 5;
      } else if (this.value >= 37) {
         return 4;
      } else if (this.value >= 17) {
         return 3;
      } else if (this.value >= 7) {
         return 2;
      } else {
         return this.value >= 3 ? 1 : 0;
      }
   }
//获得合适经验值的方法
   public static int getExperienceValue(int p_20783_) {
      if (p_20783_ >= 2477) {
         return 2477;
      } else if (p_20783_ >= 1237) {
         return 1237;
      } else if (p_20783_ >= 617) {
         return 617;
      } else if (p_20783_ >= 307) {
         return 307;
      } else if (p_20783_ >= 149) {
         return 149;
      } else if (p_20783_ >= 73) {
         return 73;
      } else if (p_20783_ >= 37) {
         return 37;
      } else if (p_20783_ >= 17) {
         return 17;
      } else if (p_20783_ >= 7) {
         return 7;
      } else {
         return p_20783_ >= 3 ? 3 : 1;
      }
   }
	// 经验球是否被攻击
   @Override
   public boolean isAttackable() {
      return false;
   }
	// 经验球添加实体的数据包
   @Override
   public Packet<ClientGamePacketListener> getAddEntityPacket() {
       // 发包
      return new ClientboundAddExperienceOrbPacket(this);
   }

   @Override
   public SoundSource getSoundSource() {
       // 经验球的声音源是环境
      return SoundSource.AMBIENT;
   }
}

```

