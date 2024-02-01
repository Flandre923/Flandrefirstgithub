---
title: Minecraft-world相关内容
date: 2023-12-11 22:22:58
tags:
- 我的世界
- Minecraft
- Java
cover:
---



Fabric的源码，对应的反混淆比较多。

forge和neoforge大家就自己对应下把

# World

字段

- CODEC用于将RegistryKey<World>类型对象转为二进制数据或者反之。RegistryKey用于表示注册表的类，World是一个注册表。
- OVERWORLD表示世界中的主世界。
- NETHER表示下届
- END表示末地
- HORIZONTAL_LIMIT水平边界，玩家可达到的最远坐标
- MAX_UPDATE_LIMIT 最大更新深度，即每个tick最多更新的方块数量
- field_30967 => 32数值的是每个区块的大小
- field_30968 => 15的是最大关照等级
- field_30969 => 24000 一天的长度tick
-  MAX_Y 玩家能达到的最大高度
- MIN_Y 玩家能达到的最低高度
- blockEntityTickers List类型，存储每个tick更新的方块实体。
- neighborUpdater 邻居更新器，用于更新周围方块
- pendingBlockEntityTickers  List类型 用于存储下一个tick要更新的方块
- iteratingTickingBlockEntities boolean值 表示是否在遍历方块实体列表，避免在遍历中添加或者删除元素
- thread 线程：表示该世界的线程，
- debugWorld 该世界是否是一个调试世界。boolean值
- ambientDarkness int 表示世界的环境暗程度，影响光照的因素
- lcgBlockSeed int，随机数算法，影响是随机行为，火蔓延，冰融化，植物生长
- lcgBlockSeedIncrement 用于随机数算法，同上
- rainGradientPrev 上一个tick的雨水渐变，受雨水影响的因素。取决时间和天气
- rainGradient 当前游戏tick的雨水渐变
- thunderGradientPrev  雷电
- tunderGradient 雷电 同上
- Random random 随机数生成器
- Random threadSafeRandom 线程安全随机数生成器，可能用于方块更新，地形生成等，已过时
- RegistryKey《DimensionType》dimension 表示世界的维度类型，定义世界的特性的类，例如天空的颜色，重力强度等。
- RegistryEntry<DimensionType> dimensionEntry，用于表示该世界的维度类型在注册表中的条目。
- MutableWorldProperties  properties 世界的属性 ，名称、难度、游戏模式、时间、天气等
- profiler 监控游戏性能的类，它可以记录游戏中的各种事件的耗时和占用率，例如渲染、更新、网络、声音等。
- isClient 该世界是否是一个客户端世界。它用于显示游戏的画面和声音，以及接收玩家的输入。客户端世界通常需要与服务器世界进行同步，以保证游戏的一致性。
- border 它是一个WorldBorder类型的对象，用于表示该世界的边界。 世界边界是一个用于限制玩家活动范围的机制，它可以由玩家或命令控制，它可以改变大小、形状、颜色等。
- biomeAccess访问该世界的生物群系。BiomeAccess是一个用于获取或缓存生物群系的接口，它可以根据不同的算法或来源提供生物群系。
- registryKey RegistryKey<World>类型，表示该世界在注册表中的键
- registryManager DynamicRegistryManager类型，管理该世界的动态注册表。动态注册表是一种可以在游戏运行时改变的注册表，它用于存储一些可以由玩家或数据包自定义的资源，例如生物群系、结构、特性等。
- damageSources。伤害来源是一个用于定义造成伤害的原因和效果的类，例如火焰、爆炸、魔法等。DamageSources是一个用于存储或获取伤害来源的类，它包含了一些常用的伤害来源的实例。
- tickOrder 它是一个长整数，表示该世界的更新顺序。

方法

- World构造方法
- isClient（）返回是否是客户端世界
- getServer（）返回MinecraftServer为Null
- isInBuildLimit（） 判断给出的pos位置是否具有限制
- isValid（） 判断给出的pos位置的天气是否是合法
- isValidHorizontally（）给出的pos 是否是在合法的范围内
- isInvalidVertically（）给出的y数值方向是否是合法的范围
- getWorldChunk（）给出的pos位置的区块
- getChunk（）给出x，z坐标位置的区块
- getChunk（）给出x，z 最低的区块状态，以及是否创建区块 返回x，z的区块
- setBlockState（）设置pos位置方块的状态为state，flags
- setBlockState（） 重载方法，给出方块的pos，方块的state，方块的flag和方块的最大更新深度，跟pos位置的方块为state，flag，设置最大更新深度，并更新和它周围方块的状体。flag是是一个用于控制方块的更新和通知的整数，它包含了不同的位，例如移动、通知监听器、通知邻居、强制状态等。Block.MOVED是一个常量，表示方块的移动位，如果这一位为真，表示方块是被移动的，例如被活塞推动。
- onBlockChanged（）空方法，当blockstate发生改变时候调用，传入的是方块pos，oldstate，newstate
- removeBlock（）传入pos，move（boolean）获得流体的状态，将当前方块设置为流体方块
- breakBlock（）

