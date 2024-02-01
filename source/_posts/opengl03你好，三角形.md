---
title: opengl03你好，三角形
date: 2023-12-04 17:44:09
tags:
- opengl
- java
cover: https://view.moezx.cc/images/2022/02/24/0357efa6c36996b3fd0edc744c3d0ba8.png

---

# 你好，三角形

> 在学习此节之前，建议将这三个单词先记下来：
>
> - 顶点数组对象：Vertex Array Object，VAO
> - 顶点缓冲对象：Vertex Buffer Object，VBO
> - 元素缓冲对象：Element Buffer Object，EBO 或 索引缓冲对象 Index Buffer Object，IBO
>
> 当指代这三个东西的时候，可能使用的是全称，也可能用的是英文缩写，翻译的时候和原文保持的一致。由于没有英文那样的分词间隔，中文全称的部分可能不太容易注意。但请记住，缩写和中文全称指代的是一个东西。

在OpenGL中，事物都存在于3D空间中，但屏幕和窗口是2D像素数组。因此，OpenGL的主要任务就是将3D坐标转换为适应屏幕的2D像素。图形渲染管线是管理这个转换过程的一系列阶段。它可以分为两个主要部分：将3D坐标转换为2D坐标，然后将2D坐标转换为有颜色的像素。

- 图形渲染管线作用

- 2D坐标 和2D像素的意思

图形渲染管线可以被划分为几个阶段，每个阶段将会把前一个阶段的输出作为输入。

图形渲染管线的每个阶段都是高度专门化的，并且可以并行执行。因此，现代的显卡具有成千上万个小处理核心，每个核心在GPU上为管线的每个阶段运行自己的小程序，以快速处理数据。这些小程序被称为着色器（Shaders）。

着色器可以由开发者编写，并且可以替代默认的着色器，从而更精细地控制渲染管线的特定部分。由于它们在GPU上运行，所以可以节省宝贵的CPU时间。OpenGL着色器使用OpenGL着色器语言（GLSL）编写。

- 什么是着色器

- 着色器语言是什么

蓝色部分代表的是我们可以注入自定义的着色器的部分。

![image-20231207221731214](https://s2.loli.net/2023/12/07/tKN5xbi9d1oTm68.png)

- 什么是顶点，顶点数据，顶点属性
- 什么是图元

图形渲染管线的每个阶段都有特定的功能。顶点着色器将单个顶点作为输入，将3D坐标转换为另一种3D坐标，并对顶点属性进行处理。图元装配阶段将顶点装配成指定图元的形状。几何着色器可以生成新的顶点，构造出其他形状。光栅化阶段将图元映射为屏幕上的像素，并执行裁剪以提高效率。片段着色器计算像素的最终颜色，包括光照、阴影等高级效果。最后的阶段是Alpha测试和混合阶段，用于判断像素是否应该丢弃，并对物体进行混合。

- 顶点着色器的作用
- 图元装配(Primitive Assembly)作用
- 几何着色器(Geometry Shader)作用
- 光栅化阶段(Rasterization Stage)作用
- 片段是什么？
- 片段着色器的作用
- Alpha测试和混合(Blending)作用

图形渲染管线非常复杂，包含许多可配置的部分。但对于大多数情况，只需要配置顶点着色器和片段着色器就足够了。几何着色器是可选的，通常可以使用默认的着色器。

## 顶点输入

在使用OpenGL绘制图形之前，需要给OpenGL提供一些顶点数据。顶点数据是指图形的坐标信息，通常是三维坐标（x、y和z）。OpenGL只处理在-1.0到1.0范围内的坐标，这些坐标被称为标准化设备坐标（Normalized Device Coordinates），超出范围的坐标将不会显示在屏幕上。

- openGL处理什么范围的坐标，这些坐标被称为什么

为了渲染一个三角形，需要指定三个顶点的坐标。在OpenGL中，顶点坐标以标准化设备坐标的形式定义，并且将z坐标设置为0.0，这样三角形的每个顶点的深度（与观察者的距离）都相同，使其看起来像是二维的。

标准化设备坐标是一个范围为-1.0到1.0的坐标空间，与常规屏幕坐标不同，其中y轴的正方向向上，坐标原点位于图像中心。为了将标准化设备坐标转换为屏幕坐标，需要进行视口变换（Viewport Transform）。

- 标准化设备坐标到屏幕坐标过程

顶点数据通过顶点缓冲对象（Vertex Buffer Objects, VBO）进行管理。VBO是在显卡内存中存储大量顶点数据的缓冲区。通过一次性发送大批数据到显卡，可以提高性能。

顶点缓冲对象是OpenGL中的一个对象，具有唯一的ID。通过生成和绑定VBO对象，可以配置OpenGL如何解释顶点数据并发送给显卡。

使用glBufferData函数将顶点数据复制到VBO的内存中。函数的参数包括目标缓冲类型、数据大小、实际数据以及数据管理方式等。

最后，需要创建顶点着色器和片段着色器来处理顶点数据。这些着色器将在显卡上创建内存用于储存顶点数据，并进行相应的处理。

- 顶点数据在哪里处理，提高性能的手段是？

- 如何创建顶点缓冲区
- 如何将顶点缓冲区绑定到opengl上
- 如何将顶点数据复制到缓冲中

```c++
// glBufferData是一个专门用来把用户定义的数据复制到当前绑定缓冲的函数。
glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
//它的第一个参数是目标缓冲的类型：顶点缓冲对象当前绑定到GL_ARRAY_BUFFER目标上
//第二个参数指定传输数据的大小(以字节为单位)；
//第三个参数是我们希望发送的实际数据。
//第四个参数指定了我们希望显卡如何管理给定的数据。
//GL_STATIC_DRAW ：数据不会或几乎不会改变。
//GL_DYNAMIC_DRAW：数据会被改变很多。
//GL_STREAM_DRAW ：数据每次绘制时都会改变。
//三角形的位置数据不会改变，每次渲染调用时都保持原样，所以它的使用类型最好是GL_STATIC_DRAW。如果，比如说一个缓冲中的数据将频繁被改变，那么使用的类型就是GL_DYNAMIC_DRAW或GL_STREAM_DRAW，这样就能确保显卡把数据放在能够高速写入的内存部分。
```

## 顶点着色器

顶点着色器是在渲染过程中起作用的一种可编程着色器。在现代OpenGL中，至少需要设置一个顶点着色器和一个片段着色器才能进行渲染。这段内容简要介绍了着色器，以及如何配置两个简单的着色器来绘制第一个三角形。首先，需要使用OpenGL着色器语言GLSL编写顶点着色器，并对其进行编译以在程序中使用。

- 做渲染至少需要什么的着色器
- 怎么编写顶点着色器

非常基础的GLSL顶点着色器的源代码，有点像C语言 

```glsl
#version 330 core
layout (location = 0) in vec3 aPos;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
}
```

第一行是版本号和core模式，330指3.3，420指4.2类推

第二行 in关键字声明了输入顶点属性（Input Vertex Attribute)，本例中，只关心顶点位置数据，因此只声明了一个顶点属性。使用layout (location = 0) 设置输入变量的位置量。

为了设置顶点着色器的输出，必须将位置数据赋值给预定义的 `gl_Position` 变量，该变量在幕后是一个 `vec4` 类型。在 `main` 函数的末尾，将 `gl_Position` 的值设置为该顶点着色器的输出。由于输入是一个3分量向量，必须将其转换为4分量向量。可以通过将 `vec3` 数据作为 `vec4` 构造函数的参数，并将 `w` 分量设置为 `1.0` 来完成此任务。

这个顶点着色器可能是最简单的形式，因为它直接将输入数据传递到着色器的输出。在实际的程序中，输入数据通常不是标准化设备坐标，因此首先需要将它们转换为OpenGL可视区域内的坐标。

## 编译着色器

- 怎么创建顶点着色器
- 怎么把编写的着色器代码附加到顶点着色器上

- 怎么编译着色器
- 怎么查看编译是否错误

```c++
glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
glCompileShader(vertexShader);
```

glShaderSource函数把要编译的着色器对象作为第一个参数。第二参数指定了传递的源码字符串数量，这里只有一个。第三个参数是顶点着色器真正的源码，第四个参数我们先设置为`NULL`。



检查错误的代码

```c++
int  success;
char infoLog[512];
glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);

if(!success)
{
    glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
    std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
}
```

## 片段着色器

片段着色器是用于渲染三角形的最后一个着色器，它负责计算像素最终的颜色输出。在这里，我们的片段着色器会持续输出橘黄色。

> 在计算机图形中，颜色通常由红、绿、蓝和透明度四个元素组成，通常缩写为RGBA。在OpenGL或GLSL中，颜色的每个部分强度通常设置在0.0到1.0之间。例如，设置红色为1.0、绿色为1.0会得到两种颜色的混合，即黄色。通过调整这三种颜色分量的不同强度，可以生成超过1600万种不同的颜色！

```glsl
#version 330 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
} 
```

片段着色器只需要一个输出变量，即表示最终输出颜色的4分量向量。通过声明 `out` 关键字，我们定义了一个名为 `FragColor` 的输出变量。接下来，我们将一个具有1.0 Alpha 值（完全不透明）的橘黄色 `vec4` 赋值给颜色输出。

```c++
unsigned int fragmentShader;
fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
glCompileShader(fragmentShader);
```

编译片段着色器的过程与顶点着色器类似，只不过使用了 `GL_FRAGMENT_SHADER` 常量作为着色器类型。创建着色器对象、将源码附加到着色器对象上并编译。

- 怎么编写片段着色器代码
- 怎么创建片段着色器
- 怎么将着色器代码附加到片段着色器
- 怎么编译
- 怎么处理错误

### 着色器程序

着色器程序对象是将多个着色器合并并最终链接完成的版本。为了使用先前编译的着色器，我们需要将它们链接成一个着色器程序对象，然后在渲染对象时激活这个着色器程序。已激活的着色器程序会在我们发送渲染调用时被使用。

将着色器链接至程序时，它会将每个着色器的输出连接到下一个着色器的输入。如果输出和输入不匹配，就会产生连接错误。

```c++
unsigned int shaderProgram;
shaderProgram = glCreateProgram();
```

创建程序对象很简单，使用 `glCreateProgram()` 函数创建一个程序，并返回一个新的程序对象的ID引用。接着，需要将之前编译的着色器附加到程序对象上，然后用 `glLinkProgram()` 进行链接。

```c++
glAttachShader(shaderProgram, vertexShader);
glAttachShader(shaderProgram, fragmentShader);
glLinkProgram(shaderProgram);
```
> 就像着色器的编译一样，我们也可以检测链接着色器程序是否失败，并获取相应的日志。与上面不同，我们不会调用glGetShaderiv和glGetShaderInfoLog，现在我们使用：
>
> ```c++
> glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
> if(!success) {
>     glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
>     ...
> }
> ```

一旦链接成功，可以使用 `glUseProgram()` 函数激活这个程序对象，从而使每个着色器调用和渲染调用都使用这个程序对象（之前定义的着色器）。

在链接着色器对象至程序对象之后，记得删除着色器对象，因为它们不再需要了。

```c++
glDeleteShader(vertexShader);
glDeleteShader(fragmentShader);
```

现在，我们已经将输入顶点数据发送给了GPU，并告诉了GPU如何处理顶点和片段着色器中的数据。不过，OpenGL还需要了解如何解释内存中的顶点数据，并将其与顶点着色器的属性进行链接。需要告诉OpenGL怎么做才能继续。

- 着色器程序对象如何创建
- 如何使用着色器程序对象，以及如何删除着色器对象
- 如何看是否有错误

## 链接顶点属性

- **链接顶点属性：**
  - 顶点着色器允许指定以顶点属性形式的输入，但需要手动指定输入数据的哪一部分对应顶点着色器的哪个属性。使用 `glVertexAttribPointer` 来告诉OpenGL如何解析顶点数据到顶点属性。
  - 使用 `glEnableVertexAttribArray` 启用顶点属性，默认情况下是禁用的。
  - 重复这个过程来配置所有顶点属性，并在渲染前设置。

```c++
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
glEnableVertexAttribArray(0);
```

- 第一个参数指定我们要配置的顶点属性。还记得我们在顶点着色器中使用`layout(location = 0)`定义了position顶点属性的位置值(Location)吗？它可以把顶点属性的位置值设置为`0`。因为我们希望把数据传递到这一个顶点属性中，所以这里我们传入`0`。
- 第二个参数指定顶点属性的大小。顶点属性是一个`vec3`，它由3个值组成，所以大小是3。
- 第三个参数指定数据的类型，这里是GL_FLOAT(GLSL中`vec*`都是由浮点数值组成的)。
- 下个参数定义我们是否希望数据被标准化(Normalize)。如果我们设置为GL_TRUE，所有数据都会被映射到0（对于有符号型signed数据是-1）到1之间。我们把它设置为GL_FALSE。
- 第五个参数叫做步长(Stride)，它告诉我们在连续的顶点属性组之间的间隔。由于下个组位置数据在3个`float`之后，我们把步长设置为`3 * sizeof(float)`。要注意的是由于我们知道这个数组是紧密排列的（在两个顶点属性之间没有空隙）我们也可以设置为0来让OpenGL决定具体步长是多少（只有当数值是紧密排列时才可用）。一旦我们有更多的顶点属性，我们就必须更小心地定义每个顶点属性之间的间隔，我们在后面会看到更多的例子（译注: 这个参数的意思简单说就是从这个属性第二次出现的地方到整个数组0位置之间有多少字节）。
- 最后一个参数的类型是`void*`，所以需要我们进行这个奇怪的强制类型转换。它表示位置数据在缓冲中起始位置的偏移量(Offset)。由于位置数据在数组的开头，所以这里是0。我们会在后面详细解释这个参数。

> 每个顶点属性从一个VBO管理的内存中获得它的数据，而具体是从哪个VBO（程序中可以有多个VBO）获取则是通过在调用glVertexAttribPointer时绑定到GL_ARRAY_BUFFER的VBO决定的。由于在调用glVertexAttribPointer之前绑定的是先前定义的VBO对象，顶点属性`0`现在会链接到它的顶点数据。

```c++
// 0. 复制顶点数组到缓冲中供OpenGL使用
glBindBuffer(GL_ARRAY_BUFFER, VBO);
glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
// 1. 设置顶点属性指针
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
glEnableVertexAttribArray(0);
// 2. 当我们渲染一个物体时要使用着色器程序
glUseProgram(shaderProgram);
// 3. 绘制物体
someOpenGLFunctionThatDrawsOurTriangle();
```

- 如何连接顶点属性

**顶点数组对象（VAO）：**

- VAO像VBO一样被绑定，用于存储属性配置，可以简化顶点属性的配置。一旦配置完成，可以在渲染时只需绑定对应的VAO即可。
- VAO会储存关于顶点属性的配置和关联的VBO对象。

![image-20231208115605259](C:\Users\flan\AppData\Roaming\Typora\typora-user-images\image-20231208115605259.png)

- 创建vao

```c++
unsigned int VAO;
glGenVertexArrays(1, &VAO);
```

- 设置vao
- 使用vao

```c++
// ..:: 初始化代码（只运行一次 (除非你的物体频繁改变)） :: ..
// 1. 绑定VAO
glBindVertexArray(VAO);
// 2. 把顶点数组复制到缓冲中供OpenGL使用
glBindBuffer(GL_ARRAY_BUFFER, VBO);
glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
// 3. 设置顶点属性指针
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
glEnableVertexAttribArray(0);

[...]

// ..:: 绘制代码（渲染循环中） :: ..
// 4. 绘制物体
glUseProgram(shaderProgram);
glBindVertexArray(VAO);
someOpenGLFunctionThatDrawsOurTriangle();
```

**使用顶点数组对象：**

- 初始化代码中，配置VAO和VBO以及相关的属性。
- 在渲染代码中，绑定VAO并使用 `glDrawArrays` 函数绘制图元。

一般当你打算绘制多个物体时，你首先要生成/配置所有的VAO（和必须的VBO及属性指针)，然后储存它们供后面使用。当我们打算绘制物体的时候就拿出相应的VAO，绑定它，绘制完物体后，再解绑VAO。

### 我们一直期待的三角形

**绘制图元：**

- 使用 `glDrawArrays` 函数来绘制图元，指定图元类型、起始索引和顶点数量。

```c++
glUseProgram(shaderProgram);
glBindVertexArray(VAO);
glDrawArrays(GL_TRIANGLES, 0, 3);
```

glDrawArrays函数第一个参数是我们打算绘制的OpenGL图元的类型。由于我们在一开始时说过，我们希望绘制的是一个三角形，这里传递GL_TRIANGLES给它。第二个参数指定了顶点数组的起始索引，我们这里填`0`。最后一个参数指定我们打算绘制多少个顶点，这里是`3`（我们只从我们的数据中渲染一个三角形，它只有3个顶点长）。

## 元素缓冲对象

元素缓冲对象（EBO）是OpenGL中的索引缓冲区，解决了绘制重复顶点的问题。举例来说，绘制一个矩形需要两个三角形，但会有顶点重复。使用 EBO 可以只储存不同的顶点并设定绘制顺序，以减少资源浪费。

**EBO的作用：**

- 对于一个矩形，定义4个不同的顶点，然后定义6个顶点的绘制顺序索引。
- 创建 EBO 对象并将索引复制到缓冲区，类似于 VBO 的操作。

```c++
float vertices[] = {
    0.5f, 0.5f, 0.0f,   // 右上角
    0.5f, -0.5f, 0.0f,  // 右下角
    -0.5f, -0.5f, 0.0f, // 左下角
    -0.5f, 0.5f, 0.0f   // 左上角
};

unsigned int indices[] = {
    // 注意索引从0开始! 
    // 此例的索引(0,1,2,3)就是顶点数组vertices的下标，
    // 这样可以由下标代表顶点组合成矩形

    0, 1, 3, // 第一个三角形
    1, 2, 3  // 第二个三角形
};
```

```c++
unsigned int EBO;
glGenBuffers(1, &EBO)
```

```java
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
```

```java
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
```

第一个参数指定了我们绘制的模式，这个和glDrawArrays的一样。第二个参数是我们打算绘制顶点的个数，这里填6，也就是说我们一共需要绘制6个顶点。第三个参数是索引的类型，这里是GL_UNSIGNED_INT。最后一个参数里我们可以指定EBO中的偏移量（或者传递一个索引数组，但是这是当你不在使用索引缓冲对象的时候），但是我们会在这里填写0。


元素缓冲对象（EBO）是OpenGL中的索引缓冲区，解决了绘制重复顶点的问题。举例来说，绘制一个矩形需要两个三角形，但会有顶点重复。使用 EBO 可以只储存不同的顶点并设定绘制顺序，以减少资源浪费。

- **使用 EBO：**
  - 使用 `glDrawElements` 函数替换 `glDrawArrays` 函数，指定绘制模式、顶点数量和索引类型。
  - 绘制时，指定 `GL_ELEMENT_ARRAY_BUFFER` 的绑定，告诉 OpenGL 从 EBO 中获取索引进行绘制。

```c++
// ..:: 初始化代码 :: ..
// 1. 绑定顶点数组对象
glBindVertexArray(VAO);
// 2. 把我们的顶点数组复制到一个顶点缓冲中，供OpenGL使用
glBindBuffer(GL_ARRAY_BUFFER, VBO);
glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
// 3. 复制我们的索引数组到一个索引缓冲中，供OpenGL使用
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
// 4. 设定顶点属性指针
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
glEnableVertexAttribArray(0);

[...]

// ..:: 绘制代码（渲染循环中） :: ..
glUseProgram(shaderProgram);
glBindVertexArray(VAO);
glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
glBindVertexArray(0);
```

## 附加资源

- [antongerdelan.net/hellotriangle](http://antongerdelan.net/opengl/hellotriangle.html)：Anton Gerdelan的渲染第一个三角形教程。
- [open.gl/drawing](https://open.gl/drawing)：Alexander Overvoorde的渲染第一个三角形教程。
- [antongerdelan.net/vertexbuffers](http://antongerdelan.net/opengl/vertexbuffers.html)：顶点缓冲对象的一些深入探讨。
- [调试](https://learnopengl.com/#!In-Practice/Debugging)：这个教程中涉及到了很多步骤，如果你在哪卡住了，阅读一点调试的教程是非常值得的（只需要阅读到调试输出部分）。

# 练习

为了更好的掌握上述概念，我准备了一些练习。建议在继续下一节的学习之前先做完这些练习，确保你对这些知识有比较好的理解。

1. 添加更多顶点到数据中，使用glDrawArrays，尝试绘制两个彼此相连的三角形：[参考解答](https://learnopengl.com/code_viewer_gh.php?code=src/1.getting_started/2.3.hello_triangle_exercise1/hello_triangle_exercise1.cpp)
2. 创建相同的两个三角形，但对它们的数据使用不同的VAO和VBO：[参考解答](https://learnopengl.com/code_viewer_gh.php?code=src/1.getting_started/2.4.hello_triangle_exercise2/hello_triangle_exercise2.cpp)
3. 创建两个着色器程序，第二个程序使用一个不同的片段着色器，输出黄色；再次绘制这两个三角形，让其中一个输出为黄色：[参考解答](https://learnopengl.com/code_viewer_gh.php?code=src/1.getting_started/2.5.hello_triangle_exercise3/hello_triangle_exercise3.cpp)





# Java代码 三角形

```java
package org.example;

import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.opengl.GL;

import static org.lwjgl.opengl.GL11.glViewport;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.*;

public class Main {
    private static final int SCR_WIDTH = 800;
    private static final int SCR_HEIGHT = 600;

    static final String vertexShaderSource = "#version 330 core\n" +
            "layout (location = 0) in vec3 aPos;\n" +
            "void main()\n" +
            "{\n" +
            "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n" +
            "}\0";
    static final String fragmentShaderSource = "#version 330 core\n" +
            "out vec4 FragColor;\n" +
            "void main()\n" +
            "{\n" +
            "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n" +
            "}\n\0";




    public static void main(String[] args) {
        // glfw: initialize and configure
        // ------------------------------
        GLFWErrorCallback.createPrint(System.err).set();
        if (!GLFW.glfwInit()) {
            throw new IllegalStateException("Unable to initialize GLFW");
        }
        GLFW.glfwDefaultWindowHints();
        GLFW.glfwWindowHint(GLFW.GLFW_CONTEXT_VERSION_MAJOR, 3);
        GLFW.glfwWindowHint(GLFW.GLFW_CONTEXT_VERSION_MINOR, 3);
        GLFW.glfwWindowHint(GLFW.GLFW_OPENGL_PROFILE, GLFW.GLFW_OPENGL_CORE_PROFILE);

//        GLFW.glfwWindowHint(GLFW.GLFW_OPENGL_FORWARD_COMPAT, GLFW.GLFW_TRUE);

        // glfw window creation
        // --------------------
        long window = GLFW.glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", 0, 0);
        if(window==0){
            System.err.println("Failed to create GLFW window");
            GLFW.glfwTerminate();
            return;
        }
        GLFW.glfwMakeContextCurrent(window);
        GLFW.glfwSetFramebufferSizeCallback(window,(window1, width, height) -> glViewport(0,0,width,height));
        GL.createCapabilities();


        // build and compile our shader program
        // ------------------------------------
        int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader,vertexShaderSource);
        glCompileShader(vertexShader);

        int success = glGetShaderi(vertexShader, GL_COMPILE_STATUS);
        if (success==0){
            String log = glGetShaderInfoLog(vertexShader);
            System.err.println("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + log);
        }
        // fragment shader
        int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader,fragmentShaderSource);
        glCompileShader(fragmentShader);

        success = glGetShaderi(fragmentShader, GL_COMPILE_STATUS);
        if (success == 0) {
            String log = glGetShaderInfoLog(fragmentShader);
            System.err.println("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + log);
        }
        // link shaders
        int shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram,vertexShader);
        glAttachShader(shaderProgram,fragmentShader);
        glLinkProgram(shaderProgram);
        // check for linking errors
        success = glGetProgrami(shaderProgram,GL_LINK_STATUS);
        if (success == 0){
            String log = glGetProgramInfoLog(shaderProgram);
            System.err.println("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + log);
        }
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------

        float[] vertices = {
                -0.5f, -0.5f, 0.0f, // left
                0.5f, -0.5f, 0.0f, // right
                0.0f,  0.5f, 0.0f  // top
        };
        int VBO = glGenBuffers();
        int VAO =  glGenVertexArrays();
        // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER,VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, false, 3 * Float.BYTES, 0);
        glEnableVertexAttribArray(0);

        // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // uncomment this call to draw in wireframe polygons.
//        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        while(!GLFW.glfwWindowShouldClose(window)){
            // input
            // -----
            if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_ESCAPE) == GLFW.GLFW_PRESS) {
                GLFW.glfwSetWindowShouldClose(window, true);
            }

            // render
            // -----
            glClearColor(0.2f,0.2f,0.2f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            // draw our first triangle
            glUseProgram(shaderProgram);
            glBindVertexArray(VAO);
            glDrawArrays(GL_TRIANGLES,0,3);

            // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
            // -------------------------------------------------------------------------------
            GLFW.glfwSwapBuffers(window);
            GLFW.glfwPollEvents();
        }
        // optional: de-allocate all resources once they've outlived their purpose:
        // ------------------------------------------------------------------------
        glDeleteVertexArrays(VAO);
        glDeleteBuffers(VBO);
        glDeleteProgram(shaderProgram);

        // glfw: terminate, clearing all previously allocated GLFW resources.
        // ------------------------------------------------------------------
        GLFW.glfwTerminate();
    }

}
```



# Java代码 元素缓冲对象



```java

import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.opengl.GL;

import static org.lwjgl.opengl.GL11.glViewport;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.*;
import static org.lwjgl.opengl.GL30.glBindVertexArray;

public class Main {
    private static final int SCR_WIDTH = 800;
    private static final int SCR_HEIGHT = 600;

    static final String vertexShaderSource = "#version 330 core\n" +
            "layout (location = 0) in vec3 aPos;\n" +
            "void main()\n" +
            "{\n" +
            "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n" +
            "}\0";
    static final String fragmentShaderSource = "#version 330 core\n" +
            "out vec4 FragColor;\n" +
            "void main()\n" +
            "{\n" +
            "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n" +
            "}\n\0";




    public static void main(String[] args) {
        // glfw: initialize and configure
        // ------------------------------
        GLFWErrorCallback.createPrint(System.err).set();
        if (!GLFW.glfwInit()) {
            throw new IllegalStateException("Unable to initialize GLFW");
        }
        GLFW.glfwDefaultWindowHints();
        GLFW.glfwWindowHint(GLFW.GLFW_CONTEXT_VERSION_MAJOR, 3);
        GLFW.glfwWindowHint(GLFW.GLFW_CONTEXT_VERSION_MINOR, 3);
        GLFW.glfwWindowHint(GLFW.GLFW_OPENGL_PROFILE, GLFW.GLFW_OPENGL_CORE_PROFILE);

//        GLFW.glfwWindowHint(GLFW.GLFW_OPENGL_FORWARD_COMPAT, GLFW.GLFW_TRUE);

        // glfw window creation
        // --------------------
        long window = GLFW.glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", 0, 0);
        if(window==0){
            System.err.println("Failed to create GLFW window");
            GLFW.glfwTerminate();
            return;
        }
        GLFW.glfwMakeContextCurrent(window);
        GLFW.glfwSetFramebufferSizeCallback(window,(window1, width, height) -> glViewport(0,0,width,height));
//        creates the necessary function pointers for OpenGL's functions, making them accessible and usable within your Java code.
        GL.createCapabilities();


        // build and compile our shader program
        // ------------------------------------
        int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader,vertexShaderSource);
        glCompileShader(vertexShader);

        int success = glGetShaderi(vertexShader, GL_COMPILE_STATUS);
        if (success==0){
            String log = glGetShaderInfoLog(vertexShader);
            System.err.println("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + log);
        }
        // fragment shader
        int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader,fragmentShaderSource);
        glCompileShader(fragmentShader);

        success = glGetShaderi(fragmentShader, GL_COMPILE_STATUS);
        if (success == 0) {
            String log = glGetShaderInfoLog(fragmentShader);
            System.err.println("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + log);
        }
        // link shaders
        int shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram,vertexShader);
        glAttachShader(shaderProgram,fragmentShader);
        glLinkProgram(shaderProgram);
        // check for linking errors
        success = glGetProgrami(shaderProgram,GL_LINK_STATUS);
        if (success == 0){
            String log = glGetProgramInfoLog(shaderProgram);
            System.err.println("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + log);
        }
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------

        float[] vertices = {
                0.5f, 0.5f, 0.0f,   // 右上角
                0.5f, -0.5f, 0.0f,  // 右下角
                -0.5f, -0.5f, 0.0f, // 左下角
                -0.5f, 0.5f, 0.0f   // 左上角
        };

         int indices[] = {
                // 注意索引从0开始!
                // 此例的索引(0,1,2,3)就是顶点数组vertices的下标，
                // 这样可以由下标代表顶点组合成矩形

                0, 1, 3, // 第一个三角形
                1, 2, 3  // 第二个三角形
        };

        int VBO = glGenBuffers();
        int VAO =  glGenVertexArrays();
        int EBO = glGenBuffers();
        // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER,VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,indices,GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, false, 3 * Float.BYTES, 0);
        glEnableVertexAttribArray(0);

        // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO; keep the EBO bound.
//        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
        // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
        glBindVertexArray(0);
        // uncomment this call to draw in wireframe polygons.
//        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        while(!GLFW.glfwWindowShouldClose(window)){
            // input
            // -----
            if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_ESCAPE) == GLFW.GLFW_PRESS) {
                GLFW.glfwSetWindowShouldClose(window, true);
            }

            // render
            // -----
            glClearColor(0.2f,0.2f,0.2f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            // draw our first triangle
            glUseProgram(shaderProgram);
            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_INT,0);

            // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
            // -------------------------------------------------------------------------------
            GLFW.glfwSwapBuffers(window);
            GLFW.glfwPollEvents();
        }
        // optional: de-allocate all resources once they've outlived their purpose:
        // ------------------------------------------------------------------------
        glDeleteVertexArrays(VAO);
        glDeleteBuffers(VBO);
        glDeleteBuffers(EBO);
        glDeleteProgram(shaderProgram);

        // glfw: terminate, clearing all previously allocated GLFW resources.
        // ------------------------------------------------------------------
        GLFW.glfwTerminate();
    }
}
```



# 练习

## 1 

```java
 float[] vertices = {
                -0.25f, -0.5f, 0.0f,
                0.25f, -0.5f, 0.0f,
                0.0f, 0.0f, 0.0f,
                -0.25f, 0.5f, 0.0f,
                0.25f, 0.5f, 0.0f,
        };

         int indices[] = {
                // 注意索引从0开始!
                // 此例的索引(0,1,2,3)就是顶点数组vertices的下标，
                // 这样可以由下标代表顶点组合成矩形

                0, 1, 2, // 第一个三角形
                2, 3, 4,  // 第二个三角形
        };
```



## 2

```java

        float[] vertices = {
                -0.25f, -0.5f, 0.0f,
                0.25f, -0.5f, 0.0f,
                0.0f, 0.0f, 0.0f,
        };

         int indices[] = {
                // 注意索引从0开始!
                // 此例的索引(0,1,2,3)就是顶点数组vertices的下标，
                // 这样可以由下标代表顶点组合成矩形
                0, 1, 2, // 第一个三角形
        };

        int VBO = glGenBuffers();
        int VBO2 = glGenBuffers();
        int VAO =  glGenVertexArrays();
        int VAO2 =  glGenVertexArrays();
        int EBO = glGenBuffers();
        // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER,VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,indices,GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, false, 3 * Float.BYTES, 0);
        glEnableVertexAttribArray(0);

        // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO; keep the EBO bound.
//        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
        // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
        glBindVertexArray(0);
//==================VAO2==============

        glBindVertexArray(VAO2);

        glBindBuffer(GL_ARRAY_BUFFER,VBO2);
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW);


        glVertexAttribPointer(0, 3, GL_FLOAT, false, 3 * Float.BYTES, 0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, 0);


        glBindVertexArray(0);
```



 ## 3

```java
package org.example;

import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.opengl.GL;

import static org.lwjgl.opengl.GL11.glViewport;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.*;
import static org.lwjgl.opengl.GL30.glBindVertexArray;

public class Main {
    private static final int SCR_WIDTH = 800;
    private static final int SCR_HEIGHT = 600;

    static final String vertexShaderSource = "#version 330 core\n" +
            "layout (location = 0) in vec3 aPos;\n" +
            "void main()\n" +
            "{\n" +
            "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n" +
            "}\0";
    static final String fragmentShaderSource = "#version 330 core\n" +
            "out vec4 FragColor;\n" +
            "void main()\n" +
            "{\n" +
            "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n" +
            "}\n\0";


    static final String fragmentShaderSource2 = "#version 330 core\n" +
            "out vec4 FragColor;\n" +
            "void main()\n" +
            "{\n" +
            "   FragColor = vec4(1.0f, 0.8f, 0.5f, 1.0f);\n" +
            "}\n\0";


    public static void main(String[] args) {
        // glfw: initialize and configure
        // ------------------------------
        GLFWErrorCallback.createPrint(System.err).set();
        if (!GLFW.glfwInit()) {
            throw new IllegalStateException("Unable to initialize GLFW");
        }
        GLFW.glfwDefaultWindowHints();
        GLFW.glfwWindowHint(GLFW.GLFW_CONTEXT_VERSION_MAJOR, 3);
        GLFW.glfwWindowHint(GLFW.GLFW_CONTEXT_VERSION_MINOR, 3);
        GLFW.glfwWindowHint(GLFW.GLFW_OPENGL_PROFILE, GLFW.GLFW_OPENGL_CORE_PROFILE);

//        GLFW.glfwWindowHint(GLFW.GLFW_OPENGL_FORWARD_COMPAT, GLFW.GLFW_TRUE);

        // glfw window creation
        // --------------------
        long window = GLFW.glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", 0, 0);
        if(window==0){
            System.err.println("Failed to create GLFW window");
            GLFW.glfwTerminate();
            return;
        }
        GLFW.glfwMakeContextCurrent(window);
        GLFW.glfwSetFramebufferSizeCallback(window,(window1, width, height) -> glViewport(0,0,width,height));
//        creates the necessary function pointers for OpenGL's functions, making them accessible and usable within your Java code.
        GL.createCapabilities();


        // build and compile our shader program
        // ------------------------------------
        int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader,vertexShaderSource);
        glCompileShader(vertexShader);

        int success = glGetShaderi(vertexShader, GL_COMPILE_STATUS);
        if (success==0){
            String log = glGetShaderInfoLog(vertexShader);
            System.err.println("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + log);
        }
        // fragment shader
        int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader,fragmentShaderSource);
        glCompileShader(fragmentShader);

        success = glGetShaderi(fragmentShader, GL_COMPILE_STATUS);
        if (success == 0) {
            String log = glGetShaderInfoLog(fragmentShader);
            System.err.println("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + log);
        }

        // ===== fragmentshader2
        int fragmentShader2 = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader2,fragmentShaderSource2);
        glCompileShader(fragmentShader2);

        success = glGetShaderi(fragmentShader2, GL_COMPILE_STATUS);
        if (success == 0) {
            String log = glGetShaderInfoLog(fragmentShader2);
            System.err.println("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + log);
        }


        // link shaders
        int shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram,vertexShader);
        glAttachShader(shaderProgram,fragmentShader);
        glLinkProgram(shaderProgram);
        // check for linking errors
        success = glGetProgrami(shaderProgram,GL_LINK_STATUS);
        if (success == 0){
            String log = glGetProgramInfoLog(shaderProgram);
            System.err.println("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + log);
        }

        // link shaders2
        int shaderProgram2 = glCreateProgram();
        glAttachShader(shaderProgram2,vertexShader);
        glAttachShader(shaderProgram2,fragmentShader2);
        glLinkProgram(shaderProgram2);
        // check for linking errors
        success = glGetProgrami(shaderProgram2,GL_LINK_STATUS);
        if (success == 0){
            String log = glGetProgramInfoLog(shaderProgram2);
            System.err.println("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + log);
        }
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        glDeleteShader(fragmentShader2);

        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------

        float[] vertices = {
                -0.25f, -0.5f, 0.0f,
                0.25f, -0.5f, 0.0f,
                0.0f, 0.0f, 0.0f,
                -0.25f, 0.5f, 0.0f,
                0.25f, 0.5f, 0.0f,
        };

         int indices[] = {
                // 注意索引从0开始!
                // 此例的索引(0,1,2,3)就是顶点数组vertices的下标，
                // 这样可以由下标代表顶点组合成矩形
                 0, 1, 2, // 第一个三角形
                 2, 3, 4,  // 第二个三角形
        };

        int VBO = glGenBuffers();
        int VAO =  glGenVertexArrays();
        int EBO = glGenBuffers();
        // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER,VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,indices,GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, false, 3 * Float.BYTES, 0);
        glEnableVertexAttribArray(0);

        // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO; keep the EBO bound.
//        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
        // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
        glBindVertexArray(0);


        // uncomment this call to draw in wireframe polygons.
//        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        while(!GLFW.glfwWindowShouldClose(window)){
            // input
            // -----
            if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_ESCAPE) == GLFW.GLFW_PRESS) {
                GLFW.glfwSetWindowShouldClose(window, true);
            }

            // render
            // -----
            glClearColor(0.2f,0.2f,0.2f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            // draw our first triangle
            glUseProgram(shaderProgram);
            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_INT,0);
            glUseProgram(shaderProgram2);
            glDrawElements(GL_TRIANGLES,3,GL_UNSIGNED_INT,0);

//            glDrawArrays(GL_TRIANGLES,0,3);
            // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
            // -------------------------------------------------------------------------------
            GLFW.glfwSwapBuffers(window);
            GLFW.glfwPollEvents();
        }
        // optional: de-allocate all resources once they've outlived their purpose:
        // ------------------------------------------------------------------------
        glDeleteVertexArrays(VAO);
        glDeleteBuffers(VBO);
        glDeleteBuffers(EBO);
        glDeleteProgram(shaderProgram);

        // glfw: terminate, clearing all previously allocated GLFW resources.
        // ------------------------------------------------------------------
        GLFW.glfwTerminate();
    }
}
```







