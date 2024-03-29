---
title: opengl02创建窗口
date: 2023-12-03 14:14:19
tags:
- opengl
- java
cover: https://view.moezx.cc/images/2022/02/24/21072e30a955e2aa314d4b879b95ebf7.png

---

# 创建窗口

- 第一步我们需要做什么？
- 为什么我们需要自己创建窗口和定义OpenGL上下文以及用户处理
- 幸运的是？GLFW库作用是什么，帮助我们省去了什么问题。

## GLFW

- GLFW是什么，我们需要他做什么？

## 构建GLFW

由于我们使用的是java，并且在上一部分已经整理好了环境，所以这里不再赘述C的内容，想了解的可以到参考文献中参看详情。

### CMake

### 编译

## 我们的第一个工程

## 链接

### Windows上的OpenGL库

### Linux上的OpenGL库

## GLAD

## 附加资源

- [GLFW: Window Guide](http://www.glfw.org/docs/latest/window_guide.html)：GLFW官方的配置GLFW窗口的指南。
- [Building applications](http://www.opengl-tutorial.org/miscellaneous/building-your-own-c-application/)：提供了很多编译或链接相关的信息和一大列错误及对应的解决方案。
- [GLFW with Code::Blocks](http://wiki.codeblocks.org/index.php?title=Using_GLFW_with_Code::Blocks)：使用Code::Blocks IDE编译GLFW。
- [Running CMake](http://www.cmake.org/runningcmake/)：简要的介绍如何在Windows和Linux上使用CMake。
- [Writing a build system under Linux](http://learnopengl.com/demo/autotools_tutorial.txt)：Wouter Verholst写的一个autotools的教程，讲的是如何在Linux上编写构建系统，尤其是针对这些教程。
- [Polytonic/Glitter](https://github.com/Polytonic/Glitter)：一个简单的样板项目，它已经提前配置了所有相关的库；如果你想要很方便地搞到一个LearnOpenGL教程的范例工程，这也是很不错的。

# 你好窗口

接下来我们创建main函数，在这个函数中我们将会实例化GLFW窗口：

```JAVA

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

    public static void main(String[] args) {
        //glfwInit函数来初始化GLFW
        if(!glfwInit()){
            throw new IllegalStateException("Unable to initialize GLFW");
        }
        //使用glfwWindowHint函数来配置GLFW
        //glfwWindowHint函数的第一个参数代表选项的名称，我们可以从很多以GLFW_开头的枚举值中选择；第二个参数接受一个整型，用来设置这个选项的值
        //我们将主版本号(Major)和次版本号(Minor)都设为3
        //我们同样明确告诉GLFW我们使用的是核心模式(Core-profile)
        //明确告诉GLFW我们需要使用核心模式意味着我们只能使用OpenGL功能的一个子集（没有我们已不再需要的向后兼容特性）
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
        glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);


    }

}
```

该函数的所有的选项以及对应的值都可以在 [GLFW’s window handling](http://www.glfw.org/docs/latest/window.html#window_hints) 这篇文档中找到。

接下来我们创建一个窗口对象，这个窗口对象存放了所有和窗口相关的数据，而且会被GLFW的其他函数频繁地用到。

```java
//创建窗口
        //glfwCreateWindow函数需要窗口的宽和高作为它的前两个参数。第三个参数表示这个窗口的名称（标题），这里我们使用"LearnOpenGL"，
        //最后两个参数我们暂时忽略。
        //。这个函数将会返回一个GLFWwindow对象
        //我们会在其它的GLFW操作中使用到
        long window = glfwCreateWindow(800,600,"LearnOpenGL",NULL,NULL);

        if(window == NULL){
            System.out.println("Failed to create GLFW window");
            glfwTerminate();
            return ;
        }
        //创建完窗口我们就可以通知GLFW将我们窗口的上下文设置为当前线程的主上下文了。
        glfwMakeContextCurrent(window);
```

- 如何初始化GLFW以及怎么配置内容
- 怎么创建窗口，设置上下文

## 视口

```java
        // 在我们开始渲染之前还有一件重要的事情要做，我们必须告诉OpenGL渲染窗口的尺寸大小，即视口(Viewport)，这样OpenGL才只能知道怎样根据窗口大小显示数据和坐标。
        //我们可以通过调用glViewport函数来设置窗口的维度(Dimension)：
        //glViewport函数前两个参数控制窗口左下角的位置。第三个和第四个参数控制渲染窗口的宽度和高度（像素）。
        glViewport(0,0,800,600);
```



我们实际上也可以将视口的维度设置为比GLFW的维度小，这样子之后所有的OpenGL渲染将会在一个更小的窗口中显示，这样子的话我们也可以将一些其它元素显示在OpenGL视口之外。

然而，当用户改变窗口的大小的时候，视口也应该被调整。我们可以对窗口注册一个回调函数(Callback Function)，它会在每次窗口大小被调整的时候被调用。这个回调函数的原型如下：

这个帧缓冲大小函数需要一个GLFWwindow作为它的第一个参数，以及两个整数表示窗口的新维度。每当窗口改变大小，GLFW会调用这个函数并填充相应的参数供你处理。

```java
    public void framebufferSizeCallback(long window,int width,int height){
        glViewport(0,0,width,height);
    }
```

```java
//        我们还需要注册这个函数，告诉GLFW我们希望每当窗口调整大小的时候调用这个函数：
        glfwSetFramebufferSizeCallback(window, Main::framebufferSizeCallback);
```

## 准备好你的引擎

我们可不希望只绘制一个图像之后我们的应用程序就立即退出并关闭窗口。我们希望程序在我们主动关闭它之前不断绘制图像并能够接受用户输入。

因此，我们需要在程序中添加一个while循环，我们可以把它称之为渲染循环(Render Loop)，它能在我们让GLFW退出前一直保持运行。下面几行的代码就实现了一个简单的渲染循环：

```java

        while(!glfwWindowShouldClose(window)){
            glfwSwapBuffers(window);
            glfwPollEvents();
        }

```

- glfwWindowShouldClose函数在我们每次循环的开始前检查一次GLFW是否被要求退出，如果是的话该函数返回`true`然后渲染循环便结束了，之后为我们就可以关闭应用程序了。
- glfwPollEvents函数检查有没有触发什么事件（比如键盘输入、鼠标移动等）、更新窗口状态，并调用对应的回调函数（可以通过回调方法手动设置）。
- glfwSwapBuffers函数会交换颜色缓冲（它是一个储存着GLFW窗口每一个像素颜色值的大缓冲），它在这一迭代中被用来绘制，并且将会作为输出显示在屏幕上。

> **双缓冲(Double Buffer)**
>
> 应用程序使用单缓冲绘图时可能会存在图像闪烁的问题。 这是因为生成的图像不是一下子被绘制出来的，而是按照从左到右，由上而下逐像素地绘制而成的。最终图像不是在瞬间显示给用户，而是通过一步一步生成的，这会导致渲染的结果很不真实。为了规避这些问题，我们应用双缓冲渲染窗口应用程序。**前**缓冲保存着最终输出的图像，它会在屏幕上显示；而所有的的渲染指令都会在**后**缓冲上绘制。当所有的渲染指令执行完毕后，我们**交换**(Swap)前缓冲和后缓冲，这样图像就立即呈显出来，之前提到的不真实感就消除了。

## 最后一件事

当渲染循环结束后我们需要正确释放/删除之前的分配的所有资源。我们可以在main函数的最后调用glfwTerminate函数来完成。

```java
glfwTerminate();
```

这样便能清理所有的资源并正确地退出应用程序。现在你可以尝试编译并运行你的应用程序了，如果没做错的话，你将会看到如下的输出：

## 输入

我们同样也希望能够在GLFW中实现一些输入控制，这可以通过使用GLFW的几个输入函数来完成。我们将会使用GLFW的glfwGetKey函数，它需要一个窗口以及一个按键作为输入。这个函数将会返回这个按键是否正在被按下。我们将创建一个processInput函数来让所有的输入代码保持整洁。

```java

    private static void processInput(long window) {
        if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
    }

```

这里我们检查用户是否按下了返回键(Esc)（如果没有按下，glfwGetKey将会返回GLFW_RELEASE。如果用户的确按下了返回键，我们将通过glfwSetwindowShouldClose使用把`WindowShouldClose`属性设置为 `true`的方法关闭GLFW。下一次while循环的条件检测将会失败，程序将会关闭。

我们接下来在渲染循环的每一个迭代中调用processInput：

```java
        while(!glfwWindowShouldClose(window)){
            processInput(window);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
```

这就给我们一个非常简单的方式来检测特定的键是否被按下，并在每一帧做出处理。



## 渲染

我们要把所有的渲染(Rendering)操作放到渲染循环中，因为我们想让这些渲染指令在每次渲染循环迭代的时候都能被执行。代码将会是这样的：

```Java
            // 我们使用一个自定义的颜色清空屏幕
            //我们还调用了glClearColor来设置清空屏幕所用的颜色
glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
while(!glfwWindowShouldClose(window)){
            processInput(window);
            //渲染


            //在每个新的渲染迭代开始的时候我们总是希望清屏，否则我们仍能看见上一次迭代的渲染结果
            //我们可以通过调用glClear函数来清空屏幕的颜色缓冲,它接受一个缓冲位(Buffer Bit)来指定要清空的缓冲
            // 可能的缓冲位有GL_COLOR_BUFFER_BIT，GL_DEPTH_BUFFER_BIT和GL_STENCIL_BUFFER_BIT\
            //由于现在我们只关心颜色值，所以我们只清空颜色缓冲。
            //整个颜色缓冲都会被填充为glClearColor里所设置的颜色
            glClear(GL_COLOR_BUFFER_BIT);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
```



你应该能够回忆起来我们在 *OpenGL* 这节教程的内容，glClearColor函数是一个**状态设置**函数，而glClear函数则是一个**状态使用**的函数，它使用了当前的状态来获取应该清除为的颜色。



## 以下是全部代码

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
    private static final int SCR_WIDTH = 800;
    private static final int SCR_HEIGHT = 600;

    private long window;


    public static void main(String[] args) {
        new Main().run();

    }
    public void run() {
        // Initialize and configure GLFW
        init();
        // Create the window and the OpenGL context
        createWindow();
        // Load all OpenGL function pointers
        loadGL();
        // Render loop
        loop();
        // Terminate GLFW and free resources
        terminate();
    }

    private void init() {
        // Set up an error callback to print to System.err
        GLFWErrorCallback.createPrint(System.err).set();

        // Initialize GLFW
        if (!glfwInit()) {
            throw new IllegalStateException("Unable to initialize GLFW");
        }

        // Set window hints for OpenGL version and profile
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        // Uncomment this line for MacOS compatibility
        // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    }

    private void createWindow() {
        // Create the window
        window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
        if (window == NULL) {
            throw new RuntimeException("Failed to create the GLFW window");
        }

        // Make the window's OpenGL context current
        glfwMakeContextCurrent(window);

        // Set up a window size callback to adjust the viewport
        glfwSetWindowSizeCallback(window, new GLFWWindowSizeCallbackI() {
            @Override
            public void invoke(long window, int width, int height) {
                // Make sure the viewport matches the new window dimensions
                glViewport(0, 0, width, height);
            }
        });
    }

    private void loadGL() {
        // Create the OpenGL capabilities object
        GLCapabilities caps = GL.createCapabilities();

        // Check if the capabilities object is null
        if (caps == null) {
            throw new IllegalStateException("Failed to create GL capabilities");
        }
    }

    private void loop() {
        // Set the clear color to red
        glClearColor(1.0f, 0.0f, 0.0f, 0.0f);

        // Loop until the user closes the window
        while (!glfwWindowShouldClose(window)) {
            // Process input
            processInput();

            // Clear the color buffer
            glClear(GL_COLOR_BUFFER_BIT);

            // Swap the front and back buffers
            glfwSwapBuffers(window);

            // Poll for window events
            glfwPollEvents();
        }
    }

    private void processInput() {
        // Check if the user pressed the Esc key
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            // Set the window's close flag to true
            glfwSetWindowShouldClose(window, true);
        }
    }

    private void terminate() {
        // Free the window callbacks and destroy the window
        glfwFreeCallbacks(window);
        glfwDestroyWindow(window);

        // Terminate GLFW and free the error callback
        glfwTerminate();
        glfwSetErrorCallback(null).free();
    }

}
```

# 参考文献

[创建窗口 - LearnOpenGL CN (learnopengl-cn.github.io)](https://learnopengl-cn.github.io/01 Getting started/02 Creating a window/)

[你好，窗口 - LearnOpenGL CN (learnopengl-cn.github.io)](https://learnopengl-cn.github.io/01 Getting started/03 Hello Window/)
