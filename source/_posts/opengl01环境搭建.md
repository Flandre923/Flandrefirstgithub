---
title: opengl01环境搭建
date: 2023-12-03 10:25:27
tags:
- opengl
- java
cover: https://view.moezx.cc/images/2022/02/24/f2c7ce0594ab77fa8dc4c70d5379c8ea.png
---



# 为什么要了解openGL

我的世界Minecraft，使用的是的OpenGL图形库，也意味着你如果需要进行一些图形的操作就会需要用的OpenGL处理。

# openGL简介

- OpenGL一般被认为什么，具有什么功能
- OpenGL真正是一个什么东西。
- OpenGL规范的作用是什么，真正由谁实现
- 规范在哪里可以找到。如何理解规范的作用。

## 核心模式和立即渲染模式

- 什么是立即渲染模式，为什么又立即渲染模式的情况下还需要核心模式
- 核心模式和立即渲染模式的比较区别，为什么学习和新模式
- 为什么不学习最新的版本
- 为什么游戏开发不使用最新的opengl规范版本

## 拓展

- 拓展是什么？
- 拓展能流行起来？
- 为什么拓展可以成为OpenGL的规范？
- 如何区分机器支持不支持拓展？

## 状态机

- OpenGL的设计思想？如何工作的？
- OpenGL的状态是指什么？
- 如何更改OpenGL的状态
- 怎么使用OpenGL渲染
- 我们的主要手段是什么

## 对象

- 为什么OpenGL需要做抽象
- 如何理解OpenGL中的对象
- OpenGL中的工作流
- OpenGL中对象思想，如何实现多对象的渲染

# LWJGL简介

- LWJGL什么？又什么作用
- LWJGL的特点

## GLFW

- GLFW作用是什么

其他的辅助阅读材料：[LWJGL入门指南：序章（我的世界(Minecraft)java原版同款游戏库） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/367160621)

# 环境搭建

 ## 模块依赖

- 如何获得对应的模块依赖
  - lwjgl提供的依赖及项目在线生成工具：[https://www.lwjgl.org/customize](https://link.zhihu.com/?target=https%3A//www.lwjgl.org/customize)

## 功能实现

1. 简单使用opengl创建和渲染窗口
2. 使用glfw进行处理窗口事件监听

## 代码实现

官方demo

执行main方法即可看到结果

```java

import org.lwjgl.*;
import org.lwjgl.glfw.*;
import org.lwjgl.opengl.*;
import org.lwjgl.system.*;

import java.nio.*;

import static org.lwjgl.glfw.Callbacks.*;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL11.*;
import static org.lwjgl.system.MemoryStack.*;
import static org.lwjgl.system.MemoryUtil.*;

public class Main {

    private long window;
    public static void main(String[] args) {
        System.out.println("Hello LWJGL" + Version.getVersion() + "!");
        new Main().run();
    }
    public void run(){
        init();
        loop();
        //释放了窗口的所有回调函数，比如键盘输入、鼠标移动、窗口大小变化等
        glfwFreeCallbacks(window);
        //销毁了窗口和它的OpenGL上下文，它们的句柄被存储在 window 变量中
        glfwDestroyWindow(window);


        //这一行终止了GLFW的运行，它会释放GLFW占用的所有资源，比如错误回调、显示器列表、游戏手柄等。
        glfwTerminate();
        //这一行设置了错误回调函数为null，然后释放了它占用的内存
        glfwSetErrorCallback(null).free();
    }

    private void init() {
        //创建了一个错误回调函数，它会将GLFW的错误信息打印到标准错误输出中
        GLFWErrorCallback.createPrint(System.err).set();
        //尝试初始化GLFW库，如果失败了，就抛出一个异常
        if ( !glfwInit() )
            throw new IllegalStateException("Unable to initialize GLFW");
//        这一行设置了窗口的创建提示为默认值，这一步是可选的，因为默认值已经被设置好了。
        glfwDefaultWindowHints(); // optional, the current window hints are already the default
        //设置了窗口的可见性提示为假
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // the window will stay hidden after creation
        //设置了窗口的可调整大小提示为真，这意味着窗口可以被用户拖动边框来改变大小
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // the window will be resizable
        //这一行创建了一个窗口和一个OpenGL上下文
        //里使用了300x300的大小，"Hello World!"的标题，NULL的显示器（表示使用窗口模式而不是全屏模式），NULL的共享窗口（表示不与其他窗口共享上下文）
        window = glfwCreateWindow(300, 300, "Hello World!", NULL, NULL);
        // 检查了窗口是否创建成功，如果失败了，就抛出一个异常
        if ( window == NULL )
            throw new RuntimeException("Failed to create the GLFW window");
        // 这一行设置了一个键盘输入的回调函数，它会在每次按下、重复或者释放一个键时被调用。这个函数的参数分别是窗口、键、扫描码、动作和修饰键。
        glfwSetKeyCallback(window, (window, key, scancode, action, mods) -> {
            if ( key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE )
                //它判断了如果按下了Esc键并且释放了，就设置窗口的关闭标志为真,这样就可以在渲染循环中检测到并退出程序。
                glfwSetWindowShouldClose(window, true); // We will detect this in the rendering loop
        });
        //使用了一个 MemoryStack 对象来分配一些临时的缓冲区
        try ( MemoryStack stack = stackPush() ) {
            //分配了两个整数型的缓冲区，用来存储窗口的宽度和高度。然后调用了 glfwGetWindowSize 函数，它会将窗口的大小写入到缓冲区中。
            IntBuffer pWidth = stack.mallocInt(1); // int*
            IntBuffer pHeight = stack.mallocInt(1); // int*

            glfwGetWindowSize(window, pWidth, pHeight);
//获取了主显示器的视频模式，它是一个 GLFWVidMode 对象，包含了显示器的分辨率、色深、刷新率等信息。这里使用了 glfwGetPrimaryMonitor 函数来获取主显示器的句柄。
            GLFWVidMode vidmode = glfwGetVideoMode(glfwGetPrimaryMonitor());
//设置了窗口的位置，它的参数分别是窗口、X坐标和Y坐标。这里使用了一些简单的计算，使得窗口在显示器的中央显示。
            glfwSetWindowPos(
                    window,
                    (vidmode.width() - pWidth.get(0)) / 2,
                    (vidmode.height() - pHeight.get(0)) / 2
            );
            //结束了 try 块，同时也释放了之前分配的缓冲区。
        }
//将窗口的OpenGL上下文设置为当前线程的上下文，这样就可以使用OpenGL的函数来绘制图形了。在调用任何OpenGL的函数之前，必须要有一个当前的上下文。
        glfwMakeContextCurrent(window);
        //这一行设置了交换间隔为1，这意味着启用了垂直同步，也就是每次交换缓冲区时会等待显示器的刷新信号。这样可以避免画面的撕裂和闪烁现象。
        glfwSwapInterval(1);
//这一行将窗口显示出来，因为之前设置了窗口的可见性提示为假，所以需要手动调用这个函数来显示窗口。
        glfwShowWindow(window);
    }

    private void loop() {
        //创建了OpenGL的功能对象，它会根据当前的上下文检测可用的OpenGL版本和扩展，并将它们映射到Java的方法中。这一步是在使用OpenGL的函数之前必须要做的。
        GL.createCapabilities();
//设置了清除颜色缓冲区时使用的颜色，它的参数分别是红色、绿色、蓝色和透明度的分量，范围都是0.0到1.0。这里使用了1.0, 0.0, 0.0, 0.0，表示纯红色。
        glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
//这一行开始了一个循环，它的条件是窗口的关闭标志为假，也就是窗口还没有被关闭。这个标志可以通过 glfwSetWindowShouldClose 函数来设置，或者通过用户点击窗口的关闭按钮来触发。
        while ( !glfwWindowShouldClose(window) ) {
            //这一行清除了帧缓冲区，它的参数是一个位掩码，表示要清除的缓冲区的类型。这里使用了 GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT，表示清除颜色缓冲区和深度缓冲区。颜色缓冲区存储了像素的颜色信息，深度缓冲区存储了像素的深度信息。清除缓冲区时，会使用之前设置的清除值，比如颜色缓冲区会被填充为红色。
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the framebuffer
//这一行交换了窗口的颜色缓冲区，也就是将后台缓冲区的内容显示到屏幕上，同时将前台缓冲区的内容移动到后台
            glfwSwapBuffers(window); // swap the color buffers
//            这一行处理了窗口的事件，比如键盘输入、鼠标移动、窗口大小变化等。
            glfwPollEvents();
        }
    }
}
```











# 参考文章

[LWJGL入门指南：序章（我的世界(Minecraft)java原版同款游戏库） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/367160621)

[LWJGL入门指南：第一行LWJGL代码，如何安装LWJGL或生成maven或gradle依赖 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/367164750)

[OpenGL - LearnOpenGL CN (learnopengl-cn.github.io)](https://learnopengl-cn.github.io/01 Getting started/01 OpenGL/)
