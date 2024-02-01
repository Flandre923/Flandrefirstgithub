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
PortalShape类 // 处理游戏中的传送门的形状
MIN_WIDTH // 最小宽度
MAX_WIDTH  // 最大宽度
MIN_HEIGHT // 最小高度
MAX_HEIGHT // 最大高度
IBlockStateExtension::isPortalFrame; // 传送门框架判断函数

// 用于给定位置和方向上寻找符合条件的传送门形状
findEmptyPortalShape()
// 在给定的环境中寻找一个符合条件的传送门的形状
// 创建一个传送门形状，调用谓词进行过滤。根据条件返回对应的Optional包装的PortalShape对象
findPortalShap()
// 初始化一个新的传送门的形状实例
PortalShape()  
// 计算传送门底部左边的位置
calculateBottomLeft(BlockPos)
// 计算宽度
calculateWidth() 
// 计算传送门到边缘的位置
getDistanceUntilEdgeAboveFrame()
// 计算高度
calculateHeight() 
// 计算传送门顶部的位置
getDistanceUntilTop() 
// 判断一个位置是否是空的
isEmpty()
// 判断传送门是否是合法的
isValid()
// 创建传送门方块（紫色那个）
createPortalBlocks() 
// 传送门是否完整
isComplete()
// 获得相对位置
getRelativePosition()
// 创建传送门信息
PortalInfo createPortalInfo()
// 用于找到碰撞免疫的位置
findCollisionFreePosition()

```



# BaseFrieBlock类

// 火焰类型方块的基类

```java
 BaseFireBlock// 火焰类型方块的基类

    // 根据放置的位置和level决定状态
getStateForPlacement
	// 根据getter和blockpos决定火焰方块的状态
getState() 
	// 返回火焰方块的形状，用于碰撞检测
getShape() 
	// 每个tick会调用的方法，处理火焰动画的效果，例如播放火焰的环境音效，添加火焰的粒子效果
animateTick() 
	// 抽象方法，用于判断方块是否可以燃烧
canBurn();
	// 处理实体进入火焰的状况，例如添加火焰伤害和燃烧效果
entityInside()
	// 处理方块被放置的情况，例如在特定的维度中，如果方块上方有足够的空间，就会尝试创建一个传送门
    onPlace()
	// 判断当前维度世界是否可以创建传送门
inPortalDimension()
	// 处理方块被破坏时候的粒子效果
spawnDestroyParticles()
	// 处理玩家破坏方块时候的情况
playerWillDestroy()
	//它用于判断方块是否可以被放置在给定的位置。
 canBePlacedAt()
	// 静态方法，用于判断给定的位置是否可以创建传送门
isPortal()

```

