---
title: 【Minecraft-1.20.4-NeoForge-模组教程】01环境搭建
date: 2024-01-30 14:47:54
tags:
- Minecraft
- 模组
- Java
cover: 
---

# 下载MDK

![image-20240130093157409](../images/%E3%80%90Minecraft-1-20-4-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9101%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/image-20240130093157409.png)

2.点击code，下载zip

![image-20240130093214264](../images/%E3%80%90Minecraft-1-20-4-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9101%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/image-20240130093214264.png)



# 解压

解压压缩包，我这里还换了个文件夹的名称

![image-20240130093310691](../images/%E3%80%90Minecraft-1-20-4-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9101%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/image-20240130093310691.png)

# 构建

用Idea打开文件夹。等待build完成。

![image-20240130093628830](../images/%E3%80%90Minecraft-1-20-4-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9101%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/image-20240130093628830.png)

构建完成

![image-20240130100459153](../images/%E3%80%90Minecraft-1-20-4-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9101%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/image-20240130100459153.png)



# 运行

运行此task获得idea的配置

![image-20240130100658872](../images/%E3%80%90Minecraft-1-20-4-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9101%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/image-20240130100658872.png)

选中run client 启动

![image-20240130100811683](../images/%E3%80%90Minecraft-1-20-4-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9101%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/image-20240130100811683.png)

正常启动游戏

![image-20240130101121979](../images/%E3%80%90Minecraft-1-20-4-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9101%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/image-20240130101121979.png)

# 配置模组信息

## 调整包结构

![image-20240130150845575](../images/%E3%80%90Minecraft-1-20-4-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9101%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/image-20240130150845575.png)

## 调整类的内容

![image-20240130151102183](../images/%E3%80%90Minecraft-1-20-4-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9101%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/image-20240130151102183.png)

ExampleMod

```java
package net.flandre923.examplemod;

import com.mojang.logging.LogUtils;
import net.neoforged.api.distmarker.Dist;
import net.neoforged.bus.api.IEventBus;
import net.neoforged.bus.api.SubscribeEvent;
import net.neoforged.fml.common.Mod;
import net.neoforged.fml.event.lifecycle.FMLClientSetupEvent;
import net.neoforged.fml.event.lifecycle.FMLCommonSetupEvent;
import net.neoforged.neoforge.common.NeoForge;
import net.neoforged.neoforge.event.server.ServerStartingEvent;
import org.slf4j.Logger;

@Mod(ExampleMod.MODID)
public class ExampleMod
{
    public static final String MODID = "examplemod";
    private static final Logger LOGGER = LogUtils.getLogger();
    public ExampleMod(IEventBus modEventBus)
    {
        modEventBus.addListener(this::commonSetup);
        NeoForge.EVENT_BUS.register(this);
    }

    private void commonSetup(final FMLCommonSetupEvent event)
    {
    }

    @SubscribeEvent
    public void onServerStarting(ServerStartingEvent event)
    {
    }

}

```

## 配置模组显示信息

```properties
# 设置gradle命令默认使用的内存。可以通过用户或命令行属性覆盖。
#org.gradle.jvmargs=
org.gradle.daemon=false
org.gradle.debug=false

#更多关于这个的信息可以在 https://github.com/neoforged/NeoGradle/blob/NG_7.0/README.md#apply-parchment-mappings 查看
#你也可以在以下链接找到最新版本：https://parchmentmc.org/docs/getting-started
neogradle.subsystems.parchment.minecraftVersion=1.20.3
neogradle.subsystems.parchment.mappingsVersion=2023.12.31
# 环境属性
# 你可以在以下链接找到最新版本：https://projects.neoforged.net/neoforged/neoforge
# Minecraft版本必须与Neo版本一致才能获得有效的构件
minecraft_version=1.20.4
# Minecraft版本范围可以使用任何版本的Minecraft作为边界
# 快照、预发行版和候选发行版不保证能正确排序，因为它们不遵循标准的版本命名规则。
minecraft_version_range=[1.20.4,1.21)
#Neo版本必须与Minecraft版本一致才能获得有效的构件
neo_version=20.4.80-beta
# Neo版本范围可以使用任何版本的Neo作为边界
neo_version_range=[20.4,)
#加载器版本范围只能使用FML的主要版本作为边界
loader_version_range=[2,)

## 模组属性

# 模组的唯一标识符。必须使用英文小写。必须符合正则表达式 [a-z][a-z0-9_]{1,63}
# 必须与主模组类中用@Mod注解的字符串常量相匹配。
mod_id=examplemod
# 模组的易于阅读的显示名称。
mod_name=Example Mod
# 模组的许可协议。在 https://choosealicense.com/ 查阅你的选项。默认为“所有权利保留”。
mod_license=All Rights Reserved
# 模组的版本。请参阅 https://semver.org/
mod_version=1.0.0
# 模组的组ID。在发布为Maven仓库的构件时才重要。
# 这应该与用于模组源的基础包相匹配。
# 请参阅 https://maven.apache.org/guides/mini/guide-naming-conventions.html
mod_group_id=net.flandre923.examplemod
# 模组的作者。这是一个简单的文本字符串，用于在模组列表中显示。
mod_authors=YourNameHere, OtherNameHere
# 模组的描述。这是一个简单的多行文本字符串，用于在模组列表中显示。
mod_description=Example mod description.\nNewline characters can be used and will be replaced properly.
```

## 重新构建项目

![image-20240130151455512](../images/%E3%80%90Minecraft-1-20-4-NeoForge-%E6%A8%A1%E7%BB%84%E6%95%99%E7%A8%8B%E3%80%9101%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/image-20240130151455512.png)
