---
title: Minecraft我的世界-如何做弓BowItem
date: 2023-12-09 20:04:04
tags:
- 我的世界
- 弓箭
cover: https://view.moezx.cc/images/2020/10/13/46556fd32bbbcd923d7cf81c99dcac0e.jpg
---



# RangedWeaponItem

**1. 类定义:**

- 定义了一个名为`RangedWeaponItem`的抽象类，该类扩展了`Item`类。

**2. 静态predicates**

- 定义了两个static predicates: 
  - `BOW_PROJECTILES`: 此谓词使用`ItemTags`检查`itemstack`是否为“Arrows”类型。`ARROWS`tag。
  - 此谓词将`BOW_PROJECTILES'谓词与对“Firework Rocket”项的附加检查相结合。这使得弩既可以射箭，也可以发射烟花火箭。

**3. 构造函数:**

- 构造函数接受`Item。Settings”对象作为参数，并调用“Item”类的父类构造函数来初始化该项。

**4. getHeldProjectiles:**

- 此方法返回“getProjectiles”方法的结果。这本质上允许子类定义自己的射弹类型，同时保持与现有“getHeldProjectle”方法的兼容性。

**5. getProjectiles:**

- 这是一种抽象方法，子类必须实现它来定义它们可以发射的特定类型的投射物。

**6. getHeldProjectile:**

- 此静态方法以“LivingEntity”和“Predicate＜ItemStack＞”作为参数，并返回 entity's OFF HAND  和 MAIN HAND 中与给定 predicate匹配的第一个itemstack。
- T方法同时检查实体 entity的Main Hand  和OFF hand 。

**7. getEnchantability:**

- 此方法覆盖“Item”类的“getEnchantatibility”方法，并返回值1。这意味着“远程武器”实例可以被附魔附魔。

**8. getRange:**

- 这是一个抽象的方法，子类必须实现它来定义武器的最大射程。

# BowItem

**1. Class Definition:**

- 此代码定义了一个名为“BowItem”的类，该类扩展了“RangedWeaponItem”类并实现了“Vanishable”接口。
- T这意味着“弓”是一种远程武器，可以被附魔、vanishable 和射箭。

**2. Static Fields:**

- Two static fields are defined:
  - `TICKS_PER_SECOND`: 此常数定义一秒钟内的刻度数（20）。
  - `RANGE`: 此常数定义弓的最大范围（15）。

**3. Constructor:**

- 构造函数接受`Item.Settings`对象作为参数，并调用`RangedWeaponItem`类的父类构造函数来初始化该item。

**4. onStoppedUsing:**

- 当玩家停止使用弓时，会调用此方法。
- 它检查玩家是否是“玩家实体”，以及他们是否有必要的arrows。
- 然后计算弓的拉动进度并创建一个箭头实体。
- 箭头实体随后以适当的速度、伤害和附魔产生。
- 最后，更新 player's inventory并播放适当的声音。

**5. getPullProgress:**

- 此静态方法将use ticks作为参数，并返回弓的拉动进度。
- 拉动进度用于计算箭头的速度。

**6. getMaxUseTime:**

- 此方法返回弓的最大使用时间（72000个刻度）。

**7. getUseAction:**

- T此方法返回弓（bow）的使用动作。

**8. use:**

- 当玩家开始使用弓时，就会调用此方法。
- 它检查玩家是否有必要的箭头，然后将玩家当前的手设置为持有弓的手。

**9. getProjectiles:**

- 此方法返回弓可以发射的投射物类型的predicate 。

**10. getRange:**

- 此方法返回弓的最大范围。





# 自定义物品模型ModelPredicateProviderRegistry

**1. Registration:**

- Three`ModelPredicateProviderRegistry.register`

  调用用于注册特定item和identifiers：

  - `Items.BOW`, identifier "pull": 此提供程序根据剩余使用时间计算弓的“pull（拉动）”进度。
  - `Items.BRUSH`, identifier "brushing": 此提供程序基于笔刷的当前使用时间模10提供“brushing”动画。
  - `Items.BOW`, identifier "pulling"：此提供程序检查实体是否实际使用了BOW，如果是则返回1.0f，否则返回0.0f。

**2. ClampedModelPredicateProvider:**

- The `ClampedModelPredicateProvider` interface 定义了用于计算模型动画进度的行为。
- 它需要四个参数:
  - `stack`: 正在使用的ItemStack。
  - `world`: 当前的世界。
  - `entity`: 使用项的实体。
  - `seed`: 动画中变化的随机种子。

**3. register method:**

- The`register`方法需要三个参数:
  - `item`: 要为其注册提供程序的item。
  - `id`: 模型动画的标识符。
  - `provider`: ClampedModelPredicateProvider实例。
- 它将提供程序存储在地图中，以便将来在模型渲染期间进行检索。
- 该映射使用项目和标识符作为有效访问的密钥。

**4. ITEM_SPECIFIC map:**

- 此静态映射存储所有已注册的模型predicate 提供程序
- 它使用项作为关键字，并存储一个嵌套映射，其中包含标识符及其相应的predicate 提供程序。







