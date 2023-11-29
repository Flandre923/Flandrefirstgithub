---
title: Minecraft 以及 Minecraft mod 反混淆实现
date: 2023-11-25 16:40:52
tags:
- 我的世界
- Mincraft
- 反混淆
- Java
cover: https://view.moezx.cc/images/2017/11/24/PhantasmPID52654660by212a.jpg
---





## 下载

[Recaf - modern bytecode editor (coley.software)](https://www.coley.software/Recaf/)

直接打开

## 使用Recaf对Minecraft进行反编译反混淆

找到Minecraft jar打开,从json中找到client mapping地址,下载保存

![image-20231126105120674](https://s2.loli.net/2023/11/26/EDnaRvK1wN6exkA.png)

![image-20231126105210776](https://s2.loli.net/2023/11/26/94vxRqnNl8GW5P7.png)

![image-20231126105354166](https://s2.loli.net/2023/11/26/t8IfZvdG1MqJoPg.png)

将Minecraft jar拖入到窗口,使用mappin-proguard对其进行反混淆

![image-20231126105141070](https://s2.loli.net/2023/11/26/7Pmrj6VtHMIhqDi.png)

![image-20231126105531477](https://s2.loli.net/2023/11/26/JiDSbyp3K5FtX97.png)

![image-20231126105603427](https://s2.loli.net/2023/11/26/kVqK6rUpLDXEbx1.png)

稍作等待即可完成

![image-20231126105808422](https://s2.loli.net/2023/11/26/IWEhfwxQlBmeFCu.png)

## 使用recaf对mod进行反编译反混淆

下载你的需要反混淆的模组jar,以及混淆表,例如新游戏版本forge的就是官方的,所以用上面的即可.

流程还是和上面一样,jar拖入后,加载反混淆表即可.

## 参考

[Minecraft模组加载器开发教程#2使用Recaf反混淆_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1gF411p7wf/?vd_source=f5ab73e8b88cb4cb94d904126cdfeb27)

[利用创可贴和recaf对模组进行修改 - 创可贴 (Bansōkō) - MC百科|最大的Minecraft中文MOD百科 (mcmod.cn)](https://www.mcmod.cn/post/3012.html)

[1fxe/Recaf4Forge: A Recaf plugin which applies mappings to forge mods, you can also export the forge workspace. (github.com)](https://github.com/1fxe/Recaf4Forge)

[如何对一个插件进行反编译修改 - 联机问答 - Minecraft(我的世界)中文论坛 - (mcbbs.net)](https://www.mcbbs.net/thread-1254952-1-1.html)

[Minecraft任意版本Forge模组反混淆教程_浩绪的博客-CSDN博客](https://blog.csdn.net/m0_74075298/article/details/128517915)
