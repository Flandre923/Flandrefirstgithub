---
title: Minecraft-源码-信标方块以及父类和接口
date: 2023-12-02 12:41:56
tags:
- 我的世界源码
- Minecraft
- Java
cover: https://w.wallhaven.cc/full/5g/wallhaven-5gpv25.jpg

---



# EntityBlock 接口

```java
// 定义了一个EntityBlock接口
// 该接口用了创建和管理方块实体
// 方块实体是一种特殊的方块，她可以存储额外的数据，并且在游戏中进行更新
public interface EntityBlock {
    // 方法用于在给定的位置和方块状态下创建一个新的方块实体对象，可以返回null
    
   BlockEntity newBlockEntity(BlockPos , BlockState );
// 获得一个方块实体的更新器，每个tick对方块实体进行更新
   // 默认方法，默认实现是返回null，T表示必须为BlockEntity或其子类。
    <T extends BlockEntity> BlockEntityTicker<T> getTicker(Level , BlockState , BlockEntityType<T> ) {
   }
    
// 获得用于方块实体的游戏事件监听器，查看该方块实体是否击沉了GameEvenListener。Holder，如果继承了就返回Listener或者返回null
    <T extends BlockEntity> GameEventListener getListener(ServerLevel , T ) {
   }
}
```





# BaseEntityBlock类

```java
class BaseEntityBlock extends Block implements EntityBlock {
    //构造方法
   protected BaseEntityBlock(BlockBehaviour.Properties ) {
   }
// 获得rendershape
    RenderShape getRenderShape(BlockState ) {
   }
    
    
//重写了父类的方法，处理方块的事件，事件id和事件参数
    //调用父类的处理事件方法
    //获得对于位置的blockentity
    //如果非空就调用方块实体的处理事件的方法，并将事件id和事件参数传入
   @Override
    boolean triggerEvent(BlockState , Level , BlockPos , int , int ) {
     
   }
    
    
//获得方块的menuprovider，获得方块实体如果方块实体继承于menprovider那么就将其转化为menuprovider

    MenuProvider getMenuProvider(BlockState , Level , BlockPos ) {
     
   }
    
    
// 创建方块实体的ticker，检测方块实体的类型是否相同，如果相同就返回A类型，否则的返回null
createTickerHelper(
      BlockEntityType<A> , BlockEntityType<E> , BlockEntityTickerp 
   ) {
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
 class BeaconBlock extends BaseEntityBlock implements BeaconBeamBlock {
    //构造方法
   public BeaconBlock() {
   }
	// 返回颜色
   @Override
   public DyeColor getColor() {
   }
     
	// 在给定位置创建一个新的方块实体
   @Override
    BlockEntity 
       newBlockEntity( ) {
   }
     
	// 获得一个方块实体的ticker，调用了createtickerhelper方法 返回一个ticker，每个tick对方块尸体进行更新，其中的调用方法是方块实体的tick

getTicker(Level , BlockState , BlockEntityType<T> ) {
   }
     
     
// 方块使用的处理
 use(BlockState p_49432_, Level p_49433_, BlockPos p_49434_, Player p_49435_, InteractionHand p_49436_, BlockHitResult p_49437_) {
             //如果是对应的方块实体 则打开menu
   }
     
     
// 方块实体使用模型渲染。
   @Override
   public RenderShape getRenderShape(BlockState ) {
   }
     
     
     
// 处理方块被放置的事件
   @Override
   public void setPlacedBy(Level , BlockPos , BlockState, LivingEntity , ItemStack ) {
       // 物品是否有自定义的名称
//设置自定义名称
         }
      }
   }
}
```



# BeaconBlockEntity 类

```java

public class BeaconBlockEntity extends BlockEntity implements MenuProvider, Nameable {
   int MAX_LEVELS = 4;//最大层数
  MobEffect[][] BEACON_EFFECTS = //信标效果
   Set<MobEffect> VALID_EFFECTS //信标效果的set集合

  Component DEFAULT_NAME // 名称
  String TAG_PRIMARY = "primary_effect";
  String TAG_SECONDARY = "secondary_effect";
   List<BeaconBlockEntity.BeaconBeamSection> beamSections = Lists.newArrayList();//信标光束部分的列表，颜色和高度？
  checkingBeamSections ;//颜色和高度
   int levels; // 层数
    int lastCheckY;// 上次检测的y坐标位置
   @Nullable
    primaryPower;// 主要效果
   @Nullable
    secondaryPower;// 次要效果
   @Nullable
    Component name;// 名字
    LockCode lockKey ; // 信标没有锁码
    // 信标数据的访问，该类用于传递方块实体的数据
    // get set getCount方法分别是获得，设置，和获得数据元素个数
 ContainerData dataAccess = new ContainerData() {
   //  0: 信标方块的层数
//1: 主效果的编码
//2: 次要效果的编码
   };
    
//检查modeeffect是否合法
   @Nullable
    MobEffect filterEffect(@Nullable MobEffect ) {
   }
    
// 构造方法
    BeaconBlockEntity(BlockPos , BlockState ) {
   }

   public static void tick() {
      //信标方块实体每 80 游戏刻检查一次光束段，如果光束被阻断，则检查结束。检查结束后，信标方块实体会更新其层数和光束段列表。如果层数大于 0，则会给附近的玩家施加相应的状态效果。以及播放相应声音等内容。
   }

    // 检查更新基座层数
     int updateBase() {

   }
//实体移除
   @Override
   public void setRemoved() {
   }

    ///更具层数以及玩家选中的效果给周围的玩家赋予药水效果
   private static void applyEffects() {
     
   
       
       

   public static void playSound(Level , BlockPos , SoundEvent ) {
   }
       
       
//获取信标方块实体的光束段列表，
   public List<BeaconBlockEntity.BeaconBeamSection> getBeamSections() {

   }
       


/// SL使用
   private static void storeEffect(CompoundTag , String , @Nullable MobEffect ) {
     
   }

   @Nullable
   private static MobEffect loadEffect(CompoundTag , String ) {
     
   }

   @Override
   public void load(CompoundTag ) {
     
   }

   @Override
   protected void saveAdditional(CompoundTag ) {
    
   }

   public void setCustomName(@Nullable Component ) {
   }

   @Nullable
   @Override
   public Component getCustomName() {
   }

   @Nullable
   @Override
   public AbstractContainerMenu createMenu(int , Inventory , Player ) {
       //判断玩家是否能解锁信标方块实体的菜单
       //如果是
       //创建一个新的BeaconMenu对象，它表示一个信标方块实体的菜单
       //否则，返回null，表示创建失败。
   }

   @Override
     getDisplayName() {
  
   }

   @Override
     getName() {
   
   }

   @Override
     setLevel(Level ) {
     
   }

   public static class BeaconBeamSection {
      final float[] color;
      private int height;

}
```



# BeaconRenderer 类

```java
@OnlyIn(Dist.CLIENT)

//类负责在游戏中渲染信标光束。 实现了BlockEntityRenderer接口
public class BeaconRenderer implements BlockEntityRenderer<BeaconBlockEntity> {
 BEAM_LOCATION 
 MAX_RENDER_Y

   public BeaconRenderer(BlockEntityRendererProvider.Context p_173529_) {
   }
    
    
//调用render（）方法来渲染每个刻度的信标
    //它从Beacon BlockEntity获取波束部分，并在其中循环。
   public void render() {

          //对于每个部分，它都会调用renderBeacon beam（）来渲染BeaconBeam的该部分
          //它传递PoseStack、缓冲区、动画时间/数据和颜色。
         
  
          //高度计数器j按截面的高度递增

      }
   }
    
    
//renderBeachBeam（）执行BeaconBeam渲染逻辑。
   private static void renderBeaconBeam(
   ) {

   }
    
    
//PoseStack、缓冲区、动画时间/值、颜色等参数来渲染光束。
   public static void renderBeaconBeam(

   ) {
     
   }

   private static void renderPart(

   ) {
     
   }

   private static void renderQuad(
    
   ) {

   }
    
    
//私有实用程序方法实际上处理用所有数据写入顶点的问题。
   private static void addVertex(
   
   ) {
      
   }
// 是否渲染偏差
   public boolean shouldRenderOffScreen(BeaconBlockEntity ) {
      
   }

    //返回视野范围
   @Override
   public int getViewDistance() {
      
//是否渲染
   public boolean shouldRender(BeaconBlockEntity , Vec3 ) {
      
   }
}
```

# BeaconMenu 类

```java

public class BeaconMenu extends AbstractContainerMenu {

Container beacon = new SimpleContainer(1) {//容量为1个插槽：
       // 能否存放物品
      public boolean canPlaceItem() {
          //检查item是否标记有 ItemTags.BEACON_PAYMENT_ITEMS 。
      }
 //该slot最大堆叠数量
   };
    
    
    //PaymentSlot对象引用
   private final BeaconMenu.PaymentSlot paymentSlot;
    // 访问是否合法的包装类
   private final ContainerLevelAccess access;

    

   public BeaconMenu(int p_39036_, Container p_39037_) {
   }

   public BeaconMenu(int p_39039_, Container p_39040_, ContainerData p_39041_, ContainerLevelAccess p_39042_) {
       //检查是否有正确的datacount
       //创建了slot
       //背包slot
   }
// 玩家关闭menu时候调用方法
   @Override
   public void removed(Player p_39049_) {
          // 移除paymentslot的内容
             //掉落在玩家附近
   }
//是否合法打开GUI
   @Override
   public boolean stillValid(Player p_39047_) {
   }
    
    
    // 获得层数
   public int getLevels() {
   }
    
    
// 返回药水效果的ID
   public static int encodeEffect(@Nullable MobEffect p_298586_) {
   }
    
    
// 通过药水效果ID返回药水效果
   @Nullable
   public static MobEffect decodeEffect(int p_298318_) {
   }

    // 获得第一种效果
   @Nullable
   public MobEffect getPrimaryEffect() {
   }
    
// 获得第二种效果
   @Nullable
   public MobEffect getSecondaryEffect() {
   }
    
// 更新效果
   public void updateEffects() {
          //设置两种效果并且移除slot的物品
          //更新blockentity
   }
//payslot是否有物品
   public boolean hasPayment() {
   }
    
    
// 自定义的payslot，只能放进去具有目标tag的物品
   class PaymentSlot extends Slot {
   }
}

```



# BeaconScreen类

```java

@OnlyIn(Dist.CLIENT)
public class BeaconScreen extends AbstractContainerScreen<BeaconMenu> {
   ResourceLocation BEACON_LOCATION 
 ResourceLocation BUTTON_DISABLED_SPRITE
 ResourceLocation BUTTON_SELECTED_SPRITE
 ResourceLocation BUTTON_HIGHLIGHTED_SPRITE 
 ResourceLocation BUTTON_SPRITE
 ResourceLocation CONFIRM_SPRITE 
 ResourceLocation CANCEL_SPRITE
   Component PRIMARY_EFFECT_LABE
   Component SECONDARY_EFFECT_LABEL 
 List<BeaconScreen.BeaconButton> beaconButtons
   @Nullable
   MobEffect primary;
   @Nullable
   MobEffect secondary;

   public BeaconScreen(final BeaconMenu p_97912_, Inventory p_97913_, Component p_97914_) {
       // menu添加slot监听器
// 数据变化
       //设置效果
   }
    
    
// 添加渲染按钮
    addBeaconButton(_) {
   }

   @Override
   protected void init() {
       //按钮清空
       //确认按钮
       //取消按钮
// 遍历信标的前三层效果,添加对应的按钮
     
      // 获取信标的第四层效果的数量，加一是因为有一个升级按钮
             // 计算该层的总宽度
      // 遍历该层的除了升级按钮外的每个效果
                   // 获取该效果的对象
                   // 创建一个信标能力按钮，参数为按钮的位置，效果的对象，是否为主要效果，和层级
                   // 设置该按钮为不激活状态
                   // 添加该按钮
      }
      // 创建一个信标升级按钮，参数为按钮的位置和一个默认的效果对象
             // 设置该按钮为不可见状态
             // 添加该按钮
   }

//tick每次更新按钮状态
   @Override
   public void containerTick() {
   }


// 获得信标层数，更新每个按钮
   void updateButtons() {
   }


//渲染labels
   @Override
   protected void renderLabels(GuiGraphics , int , int ) {
   }


   @Override
    void renderBg() {
       //绘制GUI
     
   }

   @Override
   public void render() {

   }

//信标接口
   @OnlyIn(Dist.CLIENT)
   interface BeaconButton {
   }


// 信标取消 按钮
   @OnlyIn(Dist.CLIENT)
   class BeaconCancelButton extends BeaconScreen.BeaconSpriteScreenButton {
   }
// 信标确认按钮
   @OnlyIn(Dist.CLIENT)
   class BeaconConfirmButton extends BeaconScreen.BeaconSpriteScreenButton {
   }
//信标效果升级按钮
   @OnlyIn(Dist.CLIENT)
   class BeaconPowerButton extends BeaconScreen.BeaconScreenButton {
   }


   @OnlyIn(Dist.CLIENT)
   abstract static class BeaconScreenButton extends AbstractButton implements BeaconScreen.BeaconButton {
   }

//精灵按钮
   @OnlyIn(Dist.CLIENT)
   abstract static class BeaconSpriteScreenButton extends BeaconScreen.BeaconScreenButton {
   }

//药水效果升级按钮
   @OnlyIn(Dist.CLIENT)
   class BeaconUpgradePowerButton extends BeaconScreen.BeaconPowerButton {
}

```

