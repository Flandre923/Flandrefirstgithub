---
title: 【Minecraft-1.20.4-NeoForge-模组教程】15附魔
date: 2024-01-30 15:56:54
tags:
- 我的世界
- Neoforge
cover:
---



# 

# 与附魔相关的类

Enchantment 类是附魔系统的抽象基类，用于定义和创建游戏中的附魔。每个附魔都会影响游戏中的实体或物品，例如提供额外的伤害、防御或特殊效果。该类实现了 IEnchantmentExtension 接口，以支持扩展附魔功能。



slots：附魔适用的装备槽数组。
rarity：附魔的稀有度。
category：附魔的类别，通常控制附魔可以应用到的物品类型以及在哪个创造模式标签下显示。
descriptionId：附魔的描述ID，通常用于创建本地化键。
builtInRegistryHolder：内置注册表持有者，用于存储附魔实例。



byId(int pId)：静态方法，通过内部数字ID从注册表中获取附魔。
getSlotItems(LivingEntity pEntity)：获取实体装备在附魔关心slots中的所有物品。
getRarity()：获取附魔的稀有度。
getMinLevel()：获取附魔的正常情况下的最小等级。
getMaxLevel()：获取附魔的正常情况下的最大等级。
getMinCost(int pLevel)：返回附魔在给定等级下的最小附魔能力值。
getMaxCost(int pLevel)：返回附魔在给定等级下的最大附魔能力值。
getDamageProtection(int pLevel, DamageSource pSource)：计算使用附魔时提供的额外伤害保护量。
getDamageBonus(int pLevel, MobType pType)：计算使用附魔攻击时提供的额外伤害量（已弃用）。
isCompatibleWith(Enchantment pOther)：检查附魔是否与另一个附魔兼容。
checkCompatibility(Enchantment pOther)：检查附魔能否与另一个附魔共存。
getOrCreateDescriptionId()：懒加载初始化附魔的描述ID。
getDescriptionId()：获取附魔的描述ID。
getFullname(int pLevel)：获取附魔的完整名称，包括等级。
canEnchant(ItemStack pStack)：检查附魔是否可以应用于给定的物品堆。
doPostAttack(LivingEntity pAttacker, Entity pTarget, int pLevel)：实体使用附魔攻击另一个实体后调用的钩子方法。
doPostHurt(LivingEntity pTarget, Entity pAttacker, int pLevel)：实体被另一个实体攻击后调用的钩子方法。
isTreasureOnly()：检查附魔是否只能作为宝藏获得。
isCurse()：检查附魔是否是诅咒。
isTradeable()：检查附魔是否可以由NPC交易，如村民。

isDiscoverable()：检查附魔是否可以通过游戏机制从附魔注册表中随机发现。默认返回 true，表示可以被发现。
canApplyAtEnchantingTable(ItemStack stack)：特定于附魔台应用的检查。此方法检查给定的物品堆是否可以在附魔台上应用当前附魔。它与 canEnchant(ItemStack) 方法不同，后者适用于所有可能的附魔。
isAllowedOnBooks()：检查附魔是否允许通过附魔台施加在书上。默认返回 true，表示允许这一 vanilla 特性。
builtInRegistryHolder()：已弃用的方法，返回内置注册表持有者。这个持有者用于存储附魔实例。



构造方法摘要
Enchantment(Enchantment.Rarity pRarity, EnchantmentCategory pCategory, EquipmentSlot[] pApplicableSlots)：创建一个新的附魔实例，并初始化其稀有度、类别和适用的装备槽。



枚举摘要
Rarity：附魔的稀有度枚举，用于表示附魔的稀有程度和权重。
Rarity 枚举字段和方法
COMMON：普通附魔，权重为 10。
UNCOMMON：不常见附魔，权重为 5。
RARE：稀有附魔，权重为 2。
VERY_RARE：非常稀有附魔，权重为 1。
getWeight()：获取稀有度的权重。权重用于决定附魔在附魔池中的出现频率。



# 原版附魔注册的位置

Enchantments 类是一个静态类，用于注册和存储游戏中可用的各种附魔。它提供了每种附魔的静态实例，这些实例在游戏的其他部分被引用以应用附魔效果。附魔可以应用于不同的装备槽，如头盔、胸甲、护腿、鞋子、主手武器等，并且每种附魔都有其特定的作用和稀有度。



ARMOR_SLOTS：一个包含所有盔甲装备槽的数组，用于定义哪些附魔可以应用于盔甲。
ALL_DAMAGE_PROTECTION：提供全方位伤害保护的附魔。
FIRE_PROTECTION：提供火焰伤害保护的附魔。
FALL_PROTECTION：减少跌落伤害的附魔。
BLAST_PROTECTION：提供爆炸伤害保护的附魔。
PROJECTILE_PROTECTION：提供弹射物伤害保护的附魔。
RESPIRATION：增加水下呼吸时间的附魔。
AQUA_AFFINITY：提高水下工作速度的附魔。
THORNS：反弹伤害给攻击者的附魔。
DEPTH_STRIDER：提高在水中的移动速度的附魔。
FROST_WALKER：在冰面上生成霜的附魔。
BINDING_CURSE：阻止装备脱落的诅咒附魔。
SOUL_SPEED：提高灵魂沙块上移动速度的附魔。
SWIFT_SNEAK：提高潜行速度的附魔。
SHARPNESS：增加近战武器伤害的附魔。
SMITE：对亡灵生物造成额外伤害的附魔。
BANE_OF_ARTHROPODS：对节肢生物造成额外伤害的附魔。
KNOCKBACK：增加击退效果的附魔。
FIRE_ASPECT：使攻击带有火焰效果的附魔。
MOB_LOOTING：增加从怪物掉落战利品的附魔。
SWEEPING_EDGE：增加横扫攻击伤害的附魔。
BLOCK_EFFICIENCY：增加挖掘速度的附魔。
SILK_TOUCH：使挖掘矿石不破坏它们的附魔。
UNBREAKING：增加工具耐久度的附魔。
BLOCK_FORTUNE：增加矿石掉落幸运值的附魔。
POWER_ARROWS：增加箭矢伤害的附魔。
PUNCH_ARROWS：增加箭矢击退效果的附魔。
FLAMING_ARROWS：使箭矢着火的附魔。
INFINITY_ARROWS：使箭矢无限使用的附魔。
FISHING_LUCK：增加钓鱼时获得宝藏的附魔。
FISHING_SPEED：增加钓鱼速度的附魔。
LOYALTY：使三叉戟返回到投掷者的附魔。
IMPALING：增加三叉戟对水生生物的伤害的附魔。
RIPTIDE：使三叉戟有几率触发激流效果的附魔。
CHANNELING：使三叉戟在雷暴天气中有几率召唤闪电的附魔。
MULTISHOT：使弩一次射出多支箭矢的附魔。
QUICK_CHARGE：减少弩拉弦时间的附魔。
PIERCING：增加箭矢穿透力的附魔。
MENDING：修复装备的附魔。
VANISHING_CURSE：阻止装备被修复的诅咒附魔。

# 添加一个自己的附魔

## 添加一个自己的power附魔

继承Enchantment类，重写其中的几个方法

```java
public class MyPowerfulEnchantment extends Enchantment {
    protected MyPowerfulEnchantment(Enchantment.Rarity pRarity, EquipmentSlot... pApplicableSlots) {
        // rarity 代表了这个附魔的稀有程度，可以是 COMMON、UNCOMMON、RARE 或 VERY_RARE。
        // type 代表了这个附魔可以加在什么工具/武器/装备上。
        // slots 代表了“这个附魔加在什么格子里装的工具/武器/装备上才有效果”，例如荆棘只在盔甲四件套上有效。
        // slots 会影响 getEnchantedItem（func_92099_a）的返回值，这个方法用于获取某个实体上有指定附魔的物品。

        super(pRarity, EnchantmentCategory.BOW, pApplicableSlots);
    }
    /**
     *返回所传递附魔等级所需的最小附魔能力值。
     */
    @Override
    public int getMinCost(int pEnchantmentLevel) {
        return 1 + (pEnchantmentLevel - 1) * 10;
    }
    @Override
    public int getMaxCost(int pEnchantmentLevel) {
        return this.getMinCost(pEnchantmentLevel) + 15;
    }

    @Override
    public int getMinLevel() {
        return super.getMinLevel();
    }

    @Override
    public int getMaxLevel() {
        return 5;
    }


}
```

## 添加一个注册用的类ModEnchantments

将自己的附魔添加到注册表中

```java
public class ModEnchantments {

    public static final DeferredRegister<Enchantment> ENCHANTMENTS = DeferredRegister.create(Registries.ENCHANTMENT, ExampleMod.MODID);

    public static final Supplier<Enchantment> POWER_ARROWS = ENCHANTMENTS.register("my_power", ()-> new MyPowerfulEnchantment(Enchantment.Rarity.COMMON, EquipmentSlot.MAINHAND));

    public static void register(IEventBus bus) {
        ENCHANTMENTS.register(bus);
    }
}
```

## 注册到modeventbus

```java
public ExampleMod(IEventBus modEventBus)
{
    modEventBus.addListener(this::commonSetup);
    NeoForge.EVENT_BUS.register(this);
    ModEnchantments.register(modEventBus);
}
```

## 可以在附魔台中获得

![image-20240130165652752](../images/%E3%80%90Minecraft-1-20-3-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9115%E9%99%84%E9%AD%94/image-20240130165652752.png)

## 可以在铁砧上敲

不是同一类型不能添加上去

![image-20240130170343795](../images/%E3%80%90Minecraft-1-20-3-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9115%E9%99%84%E9%AD%94/image-20240130170343795.png)

是同一类型才能添加附魔

![image-20240130170351753](../images/%E3%80%90Minecraft-1-20-3-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9115%E9%99%84%E9%AD%94/image-20240130170351753.png)

# 我的附魔如何生效

通常，附魔的效果和附魔本身的类没有关系。举个例子，精准采集和时运的逻辑实际上是在 `Block` 里的。还记得 Forge patch 后的那个获得掉落的方法 `getDrops` 吗？它最后一个 `int` 参数就是当前使用工具的时运等级。
这也意味着，你的附魔的具体效果需要通过覆写 `Block` 或 `Item` 类下的某些方法及事件订阅等方式实现。`EnchantmentHelper` 类提供了一些帮助确定某物品的附魔等级的方法，比如 `getEnchantmentLevel`（`func_77506_a`）和 `getEnchantments`（`func_82781_a`）

> 参考:[4.2 附魔 · Harbinger (covertdragon.team)](https://harbinger.covertdragon.team/chapter-04/enchantment.html)



例如这里我想让我的my_power附魔生效，我就需要在ArrowLooseEvent事件中获得弓箭实体，根据自己的附魔添加效果。

在这个类中，我将原版弓箭的逻辑搬了过来，然后取消了后续执行，在这个shot逻辑中添加了自己的附魔的power获得伤害，为了直观直接赋值了20点基础攻击伤害。

```java
@Mod.EventBusSubscriber
public class MyPowerEnchantmentEvent {
    @SubscribeEvent
    public static void MyPowerEnchantmentShot(ArrowLooseEvent event){
        ItemStack bow = event.getBow();
        Player player = event.getEntity();
        int i = event.getCharge();
        Level level = event.getLevel();
        ItemStack itemstack = player.getProjectile(bow);
        boolean flag = player.getAbilities().instabuild || EnchantmentHelper.getItemEnchantmentLevel(Enchantments.INFINITY_ARROWS, bow) > 0;

        shot(itemstack,flag,i,player,level,bow);
        // 取消掉原版的弓的射箭逻辑
        event.setCanceled(true);

    }

    public static void shot(ItemStack itemstack, Boolean flag, int i, Player player, Level pLevel, ItemStack pStack){
        if (!itemstack.isEmpty() || flag) {
            if (itemstack.isEmpty()) {
                itemstack = new ItemStack(Items.ARROW);
            }

            float f = getPowerForTime(i);
            if (!((double)f < 0.1)) {
                boolean flag1 = player.getAbilities().instabuild || (itemstack.getItem() instanceof ArrowItem && ((ArrowItem)itemstack.getItem()).isInfinite(itemstack, pStack, player));
                if (!pLevel.isClientSide) {
                    ArrowItem arrowitem = (ArrowItem)(itemstack.getItem() instanceof ArrowItem ? itemstack.getItem() : Items.ARROW);
                    AbstractArrow abstractarrow = arrowitem.createArrow(pLevel, itemstack, player);
                    abstractarrow = customArrow(abstractarrow, itemstack);
                    abstractarrow.shootFromRotation(player, player.getXRot(), player.getYRot(), 0.0F, f * 3.0F, 1.0F);
                    if (f == 1.0F) {
                        abstractarrow.setCritArrow(true);
                    }

                    int j = EnchantmentHelper.getItemEnchantmentLevel(Enchantments.POWER_ARROWS, pStack);
                    if (j > 0) {
                        abstractarrow.setBaseDamage(abstractarrow.getBaseDamage() + (double)j * 0.5 + 0.5);
                    }

                    int k = EnchantmentHelper.getItemEnchantmentLevel(Enchantments.PUNCH_ARROWS, pStack);
                    if (k > 0) {
                        abstractarrow.setKnockback(k);
                    }

                    if (EnchantmentHelper.getItemEnchantmentLevel(Enchantments.FLAMING_ARROWS, pStack) > 0) {
                        abstractarrow.setSecondsOnFire(100);
                    }
                    // 这里是我们的附魔，这里为了效果直接不按照等级，基于了20点基础伤害
                    if(EnchantmentHelper.getItemEnchantmentLevel(ModEnchantments.POWER_ARROWS.get(),pStack) > 0){
                        abstractarrow.setBaseDamage(20);
                    }

                    pStack.hurtAndBreak(1, player, p_311711_ -> p_311711_.broadcastBreakEvent(player.getUsedItemHand()));
                    if (flag1 || player.getAbilities().instabuild && (itemstack.is(Items.SPECTRAL_ARROW) || itemstack.is(Items.TIPPED_ARROW))) {
                        abstractarrow.pickup = AbstractArrow.Pickup.CREATIVE_ONLY;
                    }

                    pLevel.addFreshEntity(abstractarrow);
                }

                pLevel.playSound(
                        null,
                        player.getX(),
                        player.getY(),
                        player.getZ(),
                        SoundEvents.ARROW_SHOOT,
                        SoundSource.PLAYERS,
                        1.0F,
                        1.0F / (pLevel.getRandom().nextFloat() * 0.4F + 1.2F) + f * 0.5F
                );
                if (!flag1 && !player.getAbilities().instabuild) {
                    itemstack.shrink(1);
                    if (itemstack.isEmpty()) {
                        player.getInventory().removeItem(itemstack);
                    }
                }

            }
        }
    }

    public static float getPowerForTime(int pCharge) {
        float f = (float)pCharge / 20.0F;
        f = (f * f + f * 2.0F) / 3.0F;
        if (f > 1.0F) {
            f = 1.0F;
        }

        return f;
    }


    public static AbstractArrow customArrow(AbstractArrow arrow, ItemStack stack) {
        return arrow;
    }


}
```

# EnchantmentHelper类

EnchantmentHelper 类提供了一系列用于处理和操作附魔的工具方法。这些方法包括从物品堆栈中检索、设置和执行附魔。此外，该类还提供了计算附魔提供保护的伤害减免和攻击伤害加成等功能。



可以通过查找原版附魔的使用方法去了解和学习EnchantmentHelper类的使用。



例如我们上面使用的：

在这里通过EnchantmentHelper 获得了对于附魔的附魔等级。如果等级>0则给arrow添加一个20的基础伤害。

```java
 if(EnchantmentHelper.getItemEnchantmentLevel(ModEnchantments.POWER_ARROWS.get(),pStack) > 0){
                        abstractarrow.setBaseDamage(20);
                    }
```

# 想让附魔只给自己的物品

## EnchantmentCategory 枚举

EnchantmentCategory 枚举是 net.neoforged.neoforge.common.IExtensibleEnum 的实现，它定义了不同类型的物品可以接受哪些类型的附魔。每个子类代表一种物品类型，如盔甲、武器、可破坏物品等，并提供了检查该类型物品是否可以接受附魔的方法。

**字段摘要**
delegate：一个 java.util.function.Predicate<Item> 类型的委托，用于测试物品是否可以接受该附魔类型的附魔。

**方法摘要**
canEnchant(Item pItem)：检查给定的物品是否可以接受该附魔类型的附魔。如果 delegate 不是 null，则使用委托来测试；否则返回 false。
create(String name, java.util.function.Predicate<Item> delegate)：创建一个新的 EnchantmentCategory 实例，但这是不允许的，因为枚举不允许扩展。



ARMOR：检查物品是否为盔甲物品。
ARMOR_FEET：检查物品是否为脚部盔甲。
ARMOR_LEGS：检查物品是否为腿部盔甲。
ARMOR_CHEST：检查物品是否为胸部盔甲。
ARMOR_HEAD：检查物品是否为头部盔甲。
WEAPON：检查物品是否为武器。
DIGGER：检查物品是否为挖掘工具。
FISHING_ROD：检查物品是否为钓鱼竿。
TRIDENT：检查物品是否为三叉戟。
BREAKABLE：检查物品是否可以被破坏。
BOW：检查物品是否为弓。
WEARABLE：检查物品是否为可穿戴物品。
CROSSBOW：检查物品是否为弩。
VANISHABLE：检查物品是否为可消失物品。

## 创建自己的EnchantmentCategory

在create中应该传入一个lammabd的表达式用于判断是否可以进行附魔，这里传入的是是否是EnderIEyeItem.class类的实例。

其他的方法，你可以参考EnchantmentCategory中的内容

```java
package net.flandre923.examplemod.api;

import net.minecraft.world.item.BowItem;
import net.minecraft.world.item.EnderEyeItem;
import net.minecraft.world.item.enchantment.EnchantmentCategory;

public class ExampleModReference {
    public static final EnchantmentCategory MY_POWER = EnchantmentCategory.create("MY_POWER", EnderEyeItem.class::isInstance);

}

```



## 创建附魔使用自己的EnchantmentCategory

在这个类中直接使用了自己的EnchantmentCategory

```java
protected MyPowerfulEnchantment(Enchantment.Rarity pRarity, EquipmentSlot... pApplicableSlots) {
    // rarity 代表了这个附魔的稀有程度，可以是 COMMON、UNCOMMON、RARE 或 VERY_RARE。
    // type 代表了这个附魔可以加在什么工具/武器/装备上。
    // slots 代表了“这个附魔加在什么格子里装的工具/武器/装备上才有效果”，例如荆棘只在盔甲四件套上有效。
    // slots 会影响 getEnchantedItem（func_92099_a）的返回值，这个方法用于获取某个实体上有指定附魔的物品。

    super(pRarity, ExampleModReference.MY_POWER, pApplicableSlots);
}
```

## 现在附魔仅仅可以在自己的设置的物品上面了。

可以看到这个附魔书可以附魔在自己的设置的物品末影珍珠上。而不能附魔在弓上了。

![image-20240130180644853](../images/%E3%80%90Minecraft-1-20-3-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9115%E9%99%84%E9%AD%94/image-20240130180644853.png)

![image-20240130180653617](../images/%E3%80%90Minecraft-1-20-3-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9115%E9%99%84%E9%AD%94/image-20240130180653617.png)

## 如何让自己的EnchantmentCategory下的附魔能在附魔台中出现

![image-20240130181135206](../images/%E3%80%90Minecraft-1-20-3-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9115%E9%99%84%E9%AD%94/image-20240130181135206.png)

现在是不可以找到

经过我们从附魔台的代码中寻找发现

![image-20240130182619330](../images/%E3%80%90Minecraft-1-20-3-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9115%E9%99%84%E9%AD%94/image-20240130182619330.png)

我们可以找到最后是看物品的这个getEnchanementValue的数值决定是否附魔的

![image-20240130182636197](../images/%E3%80%90Minecraft-1-20-3-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9115%E9%99%84%E9%AD%94/image-20240130182636197.png)

所以你的物品需要重写该方法



例如这样：

```java
public class MyItem extends Item {

    public MyItem(Properties pProperties) {
        super(pProperties);
    }

    /**
     * Checks isDamagable and if it cannot be stacked
     */
    @Override
    public boolean isEnchantable(ItemStack pStack) {
        return pStack.getCount() == 1;
    }

    /**
     * Return the enchantability factor of the item, most of the time is based on material.
     */
    @Override
    public int getEnchantmentValue() {
        return 1;
    }
}
```

记得修改你的category的判断的条件

![image-20240130183720027](../images/%E3%80%90Minecraft-1-20-3-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9115%E9%99%84%E9%AD%94/image-20240130183720027.png)





# 参考：

[VReference.java - Vampirism [GitHub\] - Visual Studio Code - GitHub](https://github.dev/TeamLapen/Vampirism)

[4.2 附魔 · Harbinger (covertdragon.team)](https://harbinger.covertdragon.team/chapter-04/enchantment.html)
