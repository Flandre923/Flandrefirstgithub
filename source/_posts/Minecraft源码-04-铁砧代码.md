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

 class AnvilBlock extends FallingBlock {
    // 朝向
FACING 
    //碰撞box
 BASE
 X_LEG1 
 X_LEG2 
 X_TOPe Z_LEG1
 Z_LEG2 
 Z_TOP
    // x轴碰撞box
  X_AXIS_AABB 
    // z轴碰撞box
 Z_AXIS_AABB
    // gui的标题
Component CONTAINER_TITLE 
    // 掉落每block造成伤害
 FALL_DAMAGE_PER_DISTANCE
    // 最大造成伤害 
 FALL_DAMAGE_MAX

    AnvilBlock() {
       // 设置默认为北方
   }
//方块被放置时获取方块的状态。它接受一个BlockPlaceContext对象作为参数，并返回一个BlockState对象，该对象表示方块的状态。
     getStateForPlacement() {
   }
// 如果方块在客户端（即玩家的设备）上，它将返回InteractionResult.SUCCESS。否则，它将打开一个新的菜单，并返回InteractionResult.CONSUME。


     use() {
      if () {
      } else {
          // 打开一个menu
      }
   }
    
 //方法用于返回一个新的菜单提供者，该提供者在玩家与方块交互时打开一个新的菜单。
     getMenuProvider(_) {
      // 返回一个SimpleMenuProvider

   }
// 获取方块的形状。它接受一个BlockState对象、一个BlockGetter对象和一个BlockPos对象作为参数，并返回一个VoxelShape对象，该对象表示方块的形状。
   
     getShape() {
       //根据 x z 返回不同的碰撞box
   }
    
//falling方法用于处理方块下落时造成的伤害。它接受一个FallingBlockEntity对象作为参数，并设置该对象的伤害属性。
     falling( ) {
   }
    
//onLand和onBrokenAfterFall方法用于处理方块落地和破碎后的事件。它们接受一个Level对象、一个BlockPos对象和一个BlockState对象作为参数，并在方块落地或破碎后触发一个level事件。
   
     onLand(_) {
       // 播发声音
     
   }


     onBrokenAfterFall() {
       // 播放声音
     
   }
//获取方块下落时造成的伤害源。这个方法接受一个Entity对象作为参数，并返回一个DamageSource对象，该对象表示方块下落时造成的伤害源。
    getFallDamageSource( ) {
   }
    
//处理方块被损坏的情况。如果方块是ANVIL，那么它会变成破损的ANVIL。如果方块是破损的ANVIL，那么它会变成损坏的ANVIL。如果方块既不是ANVIL也不是破损的ANVIL，那么这个方法将返回null。
    BlockState damage( ) {
    
   
     
//rotate方法用于处理方块旋转的情况。这个方法接受一个BlockState对象和一个Rotation对象作为参数，并返回一个新的BlockState对象，该对象表示方块旋转后的状态。
   
     rotate() {
   }
//于定义方块的状态。这个方法接受一个StateDefinition.Builder对象作为参数，并添加方块的朝向状态。
   
     createBlockStateDefinition(.Buil) {
   }
        
//判断方块是否可以被路径查找。这个方法接受一个BlockState对象、一个BlockGetter对象、一个BlockPos对象和一个PathComputationType对象作为参数，并返回一个布尔值，表示方块是否可以被路径查找。在这个例子中，方块不可以被路径查找。
     isPathfindable( ,  ,  ,  ) {
   }
        
        
//获取方块的粉尘颜色。这个方法接受一个BlockState对象、一个BlockGetter对象和一个BlockPos对象作为参数，并返回一个整数，表示方块的粉尘颜色。
   
     getDustColor() {

   }
}

```

# AnvilMenu 类

```java

 class AnvilMenu extends ItemCombinerMenu {
    // 输入slot 0
INPUT_SLOT
    // 另一个输入slot 1 
ADDITIONAL_SLOT
    // 输出slot 2
 RESULT_SLOT 
    // 日志
   private static final Logger LOGGER = LogUtils.getLogger();
    // 表示是否开启调试模式。
DEBUG_COST 
    // 用于表示物品名称的最大长度。
MAX_NAME_LENGTH 
    // 用于表示修复物品的数量成本。
 repairItemCountCost;
    // 用于表示物品的名称。
itemName;
    // cost是一个DataSlot对象，用于表示修复的成本。
 cost
    // 定义的常量，用于表示修复的不同成本。
 COST_FAIL 
 COST_BASE 
 COST_ADDED_BASE 
 COST_REPAIR_MATERIAL 
 COST_REPAIR_SACRIFICE 
 COST_INCOMPATIBLE_PENALTY 
 COST_RENAME 
    // 是定义的常量，用于表示输入槽、额外槽和结果槽的X坐标。
INPUT_SLOT_X_PLACEMENT
 ADDITIONAL_SLOT_X_PLACEMENT
RESULT_SLOT_X_PLACEMENT
    // Y坐标
int SLOT_Y_PLACEMENT
	// 构造函数
   public AnvilMenu() {
   }


 	// 创建slot的定义。
   
     createInputSlotDefinitions() {

   }
//判断方块是否是铁砧。
 
     isValidBlock(BlockState ) {
   }
//判断玩家是否可以拿起物品。
     mayPickup(Player , boolean ) {
      
   }
     
//处理玩家拾取物品的事件。
    donTake(Player , ItemStack) {
       // 生存模式
      if () {
          //扣除玩家cost的经验
      }
       //定义了一个名为breakChance的浮点数变量，用于表示破损的概率,这个概率是通过调用net.neoforged.neoforge.common.CommonHooks.onAnvilRepair方法计算得出的
        
//清空第一个输入槽的物品。
        
       //检查repairItemCountCost的值。如果repairItemCountCost大于0，那么获取第二个输入槽的物品，并检查它的数量是否大于repairItemCountCost。如果是，那么将repairItemCountCost从物品的数量中减去，并将减少后的物品重新放入第二个输入槽。否则，清空第二个输入槽的物品。
      if () {
       
      } else {
        
      }
        
//将cost的值设置为0。
        
       //执行一个lambda表达式，该表达式用于处理铁砧的破损。如果玩家生存模式，并且铁砧的状态是铁砧，并且随机数小于breakChance，那么调用AnvilBlock.damage方法破损铁砧，并触发相应的破损事件。否则，触发破损事件。
    
   }
//用于计算修复物品的结果。
  createResult() {
       //获取第一个输入槽的物品，并将cost的值设置为1。
       
      
       //初始化三个整数变量i、j和k，用于存储修复的成本。


       //检查第一个输入槽的物品是否为空。如果为空，那么清空结果槽的物品，并将cost的值设置为0。
      if ()) {
          

      } else {
          //一个输入槽的物品不为空，那么复制第一个输入槽的物品，并获取第二个输入槽的物品。然后，获取第一个输入槽的物品的附魔列表，并计算修复的成本。
          
       
//然后，检查是否需要修复物品。如果不需要，那么返回。
        
          
          //果需要修复物品，那么检查第二个输入槽的物品是否为空。如果不为空，那么检查第一个输入槽的物品是否可以被修复，并计算修复的成本。
         if () {
           
            } else {
                //如果第一个输入槽的物品不可以被修复，那么检查第二个输入槽的物品是否可以用来修复第一个输入槽的物品。如果可以，那么计算修复的成本。
               if () {
                 
               }

               if () {
                
                  if () {
                   
                  }

                  if () {
                    
                  }
               }
//如果第二个输入槽的物品不能用来修复第一个输入槽的物品，那么检查第二个输入槽的物品的附魔列表。
//如果第二个输入槽的物品的附魔列表不为空，那么遍历附魔列表，并对每个附魔进行处理。如果附魔不为空，那么获取第一个输入槽的物品的附魔列表中的该附魔的等级，或者如果第一个输入槽的物品的附魔列表中没有该附魔，那么获取第二个输入槽的物品的附魔列表中的该附魔的等级。然后，如果附魔的等级不相等，那么将较大的等级设置为新的等级。如果新的等级超过了附魔的最大等级，那么将新的等级设置为附魔的最大等级。然后，将新的等级添加到第一个输入槽的物品的附魔列表中。
              
//如果所有的附魔都不能添加到第一个输入槽的物品的附魔列表中，那么清空结果槽的物品，并将cost的值设置为0。
               if () {
                
               }
            }
         }
//第一个输入槽的物品的名称不为空，并且不是空格，那么检查第一个输入槽的物品的名称是否与第二个输入槽的物品的名称相同。如果不相同，那么将k的值设置为1，并将i的值增加k的值。然后，将第一个输入槽的物品的名称设置为itemName。
        
      
      
          //如果flag为真，并且第一个输入槽的物品不可以被附魔，那么将itemstack1的值设置为空。
       
      
//然后，将cost的值设置为j和i的和。
        
      
          //如果i的值小于等于0，那么将itemstack1的值设置为空。
         
      
//如果k的值等于i的值，并且k的值大于0，并且cost的值大于等于40，那么将cost的值设置为39。
         
      
//如果cost的值大于等于40，并且玩家没有无敌模式，那么将itemstack1的值设置为空。
        
      
//如果itemstack1的值不为空，那么获取第一个输入槽的物品的基础修复成本，并检查第二个输入槽的物品是否为空。如果不为空，并且第二个输入槽的物品的基础修复成本大于第一个输入槽的物品的基础修复成本，那么将第二个输入槽的物品的基础修复成本设置为新的修复成本。
        
      
      
//如果k的值不等于i的值，或者k的值等于0，那么将新的修复成本计算为增加的修复成本。
          
      
      
//然后，将新的修复成本设置为itemstack1的修复成本，并将itemstack1的附魔列表设置为map。
         
      
//最后，将itemstack1的值设置为结果槽的物品，并广播更改。
       

      
//它接受一个整数作为参数 ( p_39026_ )，这可能代表该项目的基本成本。然后，该方法返回基本成本乘以 2，然后再增加 1。这表明基本成本每增加一个单位，维修成本就会增加 100%
  }
      
      
// 它接受一个字符串作为参数 ( )，这是该物品的新名称。该方法首先使用 validateName 方法验证名称。如果名称有效且与当前项目名称不同，则该方法设置新名称，更新slot 2 中项目的悬停名称（如果存在），然后调用createResult方法。如果名称无效或与当前项目名称相同，则该方法返回 
     setItemName(String ) {

   }
//它接受一个字符串作为参数 (  )，这是要验证的名称。该方法使用 SharedConstants.filterText 过滤文本，并检查过滤后文本的长度是否小于或等于50。如果是，则返回过滤后的文本；否则，返回过滤后的文本。否则，返回 null 

      validateName(String ) {

   }
//此方法返回该项目的当前成本。它通过调用 cost 对象上的 get 方法来完成此操作。
     getCost() {
     
   }
//此方法设置项目的最大成本。它接受一个整数作为参数 ( value )，这是新的最大成本。然后，该方法通过调用 cost 对象 ayokoding.com 上的 set 方法将 cost 对象设置为新的最大成本。
     setMaximumCost(int ) {

   }
}

```

# AnvilScreen类

```java
//用于为游戏中的铁砧创建图形用户界面 (GUI)。
//类只能在游戏的客户端加载
@OnlyIn(...)
//它扩展了 ItemCombinerScreen<AnvilMenu> ，这表明它是一种允许玩家以某种方式组合项目的屏幕。
 class AnvilScreen extends ItemCombinerScreen<AnvilMenu> {
    //用于存储 GUI 中使用的各种精灵的位置。这些精灵可能用于在屏幕上绘制 GUI 元素。
 TEXT_FIELD_SPRITE
   [....]
     
     
    // name 是一个 EditBox ，玩家可以使用它来输入项目的名称。
    EditBox ;
    //。 player 是一个 Player 对象，代表正在使用 anvil 的玩家。
     Player ;
//它需要三个参数： AnvilMenu 、 Inventory 和 Component 。
     
    //AnvilMenu 可能是玩家使用铁砧时显示的Menu。 Inventory 可能是玩家的背包， Component 可能是屏幕上显示的标题
    AnvilScreen(AnvilMenu , Inventory , Component ) {
   }
     
//屏幕初始化时被调用。它设置屏幕的 GUI 元素，包括用于输入项目名称的 EditBox 和初始焦点
  
     subInit() {
     ....
   }
     
//调整屏幕大小时会调用此方法。它保存 EditBox 的当前值，重新初始化屏幕，然后恢复 EditBox 的值。
resize(,  ,  ) {
   ....
   }
     
     
//当按下某个键时会调用此方法。它检查该键是否为转义键，如果是，则关闭容器。否则，它会检查 EditBox 是否可以使用输入，如果不能，则将按键传递给超类
     keyPressed() {
      ....
   }
     
//当item名称更改时调用此方法。它检查菜单槽 0 中的项目是否有项目，如果有，它会向服务器发送一个数据包以将该项目重命名该item
    void onNameChanged( ) {
     .....
   }
     
     
//调用此方法以在屏幕上呈现标签。它检查物品的成本，如果成本大于0，它会根据成本以及玩家是否负担得起来设置文本和文本本身的颜色。如果成本大于或等于 40 并且玩家不是创造模式，则会将文本设置为“太昂贵”。如果成本小于 40，则会将文本设置为“维修成本：[成本]”。如果result slot没有item，则会将文本设置为空。然后它在屏幕上绘制文本
   @Override
   protected void renderLabels() {
     ......
   }
     
     
//调用该方法来渲染屏幕背景。它根据插槽 0 是否有项目来 blit（绘制）精灵 
     renderBg() {
   
   }
     
     
//调用该方法来渲染屏幕的前景。它呈现用于输入item名称 的 EditBox 。
     renderFg() {
   }
     
     
//调用此方法以在屏幕上呈现错误图标。它检查插槽 0 和 1 是否有item，以及结果插槽是否没有item，如果满足这些条件，它会为错误图标生成一个 sprite。
     renderErrorIcon() {
      
   }

 slotChanged() {
    
}
```

# ServerGamePacketListenerImpl 类

```java
   
     handleRenameItem(ServerboundRenameItemPacket ) {

       /// 服务器处理改变命名的发包
   }

```



# ItemCombinerMenu 类

```java

public abstract class ItemCombinerMenu extends AbstractContainerMenu {
    // 玩家背包行
NVENTORY_SLOTS_PER_ROW 
    // 玩家背包列
INVENTORY_SLOTS_PER_COLUMN
    // 
ContainerLevelAccess ;
 Player ;
//输入 slot
nputSlots;
    // 输入 slot indexinputSlotIndexes;
    // result slot
resultSlots
resultSlotIndex;
// 子类实现方法
    // 能否被拿起
abstract boolean mayPickup(Player , boolean );
// 能否拿出
 abstract void onTake(Player , ItemStack );
// 是否合法方块
 abstract boolean isValidBlock(BlockState );
// 构造函数
    ItemCombinerMenu(@Nullable MenuType<?> , int , Inventory , ContainerLevelAccess p_39776_) {
     
   }
// 创建输入slot
     createInputSlots(ItemCombinerMenuSlotDefinition ) {
      
   }
    
// 创建输出的slot
     createResultSlot() {
、
          // 能否被放置
// 能否被拿起，调用抽象方法，由子类实现
// 能否被拿出，调用抽象方法，由子类实现

      });
   }


// 创建玩家的背包
     createInventorySlots( ) {
      
   }

// 合成结果，子类实现
    abstract void createResult();

// 输入的slot创建函数，子类实现
    abstract ItemCombinerMenuSlotDefinition createInputSlotDefinitions();

// 创建simplecontainer
    SimpleContainer createContainer(int ) {
      return{
          // 重写方法，当内容改变时候，设置赃位
         
         
      };
   }


//slots改变时候，如果不是输入slot则调用createresult方法。
   @Override
   public void slotsChanged( ) {
    
   }

// 当移除当前的menu时候
   @Override
   public void removed(Player ) {
     
   }
// 判断是否合法位置打开menu
   @Override
   public boolean stillValid(Player ) {
      
   }

// shift的快速移动
    ItemStack quickMoveStack(Player , int ) {

   }


}

```



# ItemCombinerScreen类

```java

@OnlyIn(Dist.CLIENT)
 abstract class ItemCombinerScreen<T extends ItemCombinerMenu> extends AbstractContainerScreen<T> implements ContainerListener {
    // menuResource
     menuResource;
// 
   public ItemCombinerScreen(T , Inventory , Component , ResourceLocation ) {

   }
// 子类实现
   protected void subInit() {
   }
     
// init函数
   @Override
   protected void init() {
    
// 关闭scrren时候移除 SlotListener
   
   public  removed() {
   
   }
       
// 渲染
   @Override
   public void render() {
    
   }
       
// 渲染文字 子类实现
   protected void renderFg(GuiGraphics p_283399_, int p_98928_, int p_98929_, float p_98930_) {
   }
       
// 渲染背景图片，以及error图片
   @Override
   protected void renderBg() {
     

   protected abstract void renderErrorIcon(GuiGraphics p_281990_, int p_266822_, int p_267045_);
       
// 两个接口的方法

   public void dataChanged() {
   }


   public void slotChanged() {
   }
}
```

