---
title: forge 和 NeoForge 配置Mixin-1.20.4版本
date: 2024-01-29 17:01:30
tags:
- Java
- 我的世界
- Minecraft
cover: https://view.moezx.cc/images/2019/06/09/66873012_p0_master1200.jpg
---

# Forge配置Mixins

## 方法1.使用Minecraft Development 插件

创建项目时候勾选上Use Mixins即可。

![image-20240129171003107](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240129171003107.png)

点击create后等待构建完成。

这里我使用这个插件构建项目导致启动游戏会报错找不到main类，不怎么怎么解决。

## 方法2 在Forge开发环境中配置Mixins

### 配置forge

下载Forge的MDK

![image-20240129222749767](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240129222749767.png)

等待广告后下载

![image-20240129222809167](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240129222809167.png)

解压后使用IDEA打开这个文件夹

![image-20240129222905734](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240129222905734.png)

![image-20240129222938521](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240129222938521.png)

等待项目构建完成

![image-20240129222955992](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240129222955992.png)

运行该Task

![image-20240129223805156](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240129223805156.png)

完成后可以选择run client 

![image-20240129223839945](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240129223839945.png)

正常启动，就说明forge可以正常运行了。

![image-20240129224116787](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240129224116787.png)

### 配置mixin

1.将MixinGradle插件添加到Gradle构建脚本中

```groovy
buildscript {
    repositories {
        // These repositories are only for Gradle plugins, put any other repositories in the repository block further below
        maven { url = 'https://repo.spongepowered.org/repository/maven-public/' }
        mavenCentral()
        jcenter()
    }
    dependencies {
        classpath 'org.spongepowered:mixingradle:0.7-SNAPSHOT'
    }
}


apply plugin: 'org.spongepowered.mixin'


```

2.配置一个refmap的生成，使用它来映射到不同的Minecraft环境。

```groovy

mixin {
    add sourceSets.main, "${mod_id}.refmap.json"

    config "${mod_id}.mixins.json"
    debug.export = true

}

```

3.添加一个Mixin作为注释处理器

```groovy
dependencies {

    annotationProcessor 'org.spongepowered:mixin:0.8.5:processor'
}
```

4.添加一个Mixin配置文件

modid.mixins.json

```json
{
  "required": true,
  "minVersion": "0.8",
  "package": "com.example.examplemod.mixin",
  "compatibilityLevel": "JAVA_17",
  "refmap": "examplemod.refmap.json",
  "mixins": [
    "ExampleMixin",
    "LivingEntityMixin"
  ],
  "client": [
  ],
  "injectors": {
    "defaultRequire": 1
  }
}

```

5.最后，您必须使用 `genIntellijRuns` 或 `genEcliipseRuns` 重新生成运行配置，具体取决于您的IDE，以允许MixinGradle配置它们

### 测试Mixin

![image-20240130003103185](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240130003103185.png)

```java
package com.example.examplemod.mixin;

import net.minecraft.server.MinecraftServer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(MinecraftServer.class)
public class ExampleMixin {
    @Inject(at = @At("HEAD"),method = "loadLevel")
    private void init(CallbackInfo info){
        System.out.println("LoadLevel---");
    }
}

```

添加ExampleMinxin类到mixins的json文件中

```java
{
  "required": true,
  "minVersion": "0.8",
  "package": "com.example.examplemod.mixin",
  "compatibilityLevel": "JAVA_17",
  "refmap": "examplemod.refmap.json",
  "mixins": [
    "ExampleMixin",
    "LivingEntityMixin"
  ],
  "client": [
  ],
  "injectors": {
    "defaultRequire": 1
  }
}

```



启动游戏进入世界会打印LoadLevel的内容

![image-20240130003208233](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240130003208233.png)

# NeoForge配置Mixins

## 配置Neoforge开发环境

1.进入官网点击MDK

![image-20240130093157409](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240130093157409.png)

2.点击code，下载zip

![image-20240130093214264](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240130093214264.png)

解压压缩包，我这里还换了个文件夹的名称

![image-20240130093310691](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240130093310691.png)

用Idea打开文件夹。等待build完成。

![image-20240130093628830](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240130093628830.png)

构建完成

![image-20240130100459153](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240130100459153.png)

运行此task获得idea的配置

![image-20240130100658872](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240130100658872.png)

选中run client 启动

![image-20240130100811683](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240130100811683.png)

正常启动游戏

![image-20240130101121979](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240130101121979.png)

## 配置mixin

在build.gradle中配置以下的内容

```groovy
plugins {
    id 'java-library'
    id 'eclipse'
    id 'idea'
    id 'maven-publish'
    id 'net.neoforged.gradle.userdev' version '7.0.80'
    id 'net.neoforged.gradle.mixin' version '7.0.80'
}

mixin {
    config("${mod_id}.mixins.json")
}

//configurations {
//    annotationProcessor.exclude group: "org.spongepowered", module: "mixin"
//}
```

2.添加mixins的json文件

![image-20240130112958690](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240130112958690.png)

ExampleMixin

```java
package com.example.examplemod.mixin;

import net.minecraft.server.MinecraftServer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(MinecraftServer.class)
public class ExampleMixin {
    @Inject(at = @At("HEAD"),method = "loadLevel")
    private void init(CallbackInfo info){
        System.out.println("LoadLevel---");
    }
}
```

examplemod.mixins.json

```json
{
  "required": true,
  "minVersion": "0.8",
  "package": "com.example.examplemod.mixin",
  "compatibilityLevel": "JAVA_17",
  "refmap": "examplemod.refmap.json",
  "mixins": [
    "ExampleMixin",
    "LivingEntityMixin"
  ],
  "client": [
  ],
  "injectors": {
    "defaultRequire": 1
  }
}
```

# 启动游戏测试

创建一个世界，进入游戏看查看到mixin成功运行了。

![image-20240130113046700](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240130113046700.png)



# Neoforge默认已经支持mixin，直接在toml中配置下json文件即可使用

mods.toml

```toml
[[mixins]]
    config="examplemod.mixins.json"
```

examplemod.mixins.json

```json
{
  "required": true,
  "minVersion": "0.8",
  "package": "net.flandre923.examplemod.mixin",
  "compatibilityLevel": "JAVA_17",
  "refmap": "examplemod.refmap.json",
  "mixins": [
    "ExampleMixin"
  ],
  "client": [
  ],
  "injectors": {
    "defaultRequire": 1
  }
}
```

添加一个测试用的类

```java
package net.flandre923.examplemod.mixin;

import net.minecraft.server.MinecraftServer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(MinecraftServer.class)
public class ExampleMixin {
    @Inject(at = @At("HEAD"),method = "loadLevel")
    private void init(CallbackInfo info){
        System.out.println("LoadLevel---");
    }
}
```

## 启动测试

![image-20240201160454120](../images/forge%E9%85%8D%E7%BD%AEMixin-1-20-4%E7%89%88%E6%9C%AC/image-20240201160454120.png)

# 参考

[Dark的网站|Mixin简介- Forge --- Dark's Site | Mixin Introduction - Forge (darkhax.net)](https://darkhax.net/2020/07/mixins)

[build.gradle - BetterCompatibilityChecker [GitHub\] - Visual Studio Code - GitHub](https://github.dev/nanite/BetterCompatibilityChecker)
