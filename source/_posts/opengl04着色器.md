---
title: opengl04着色器
date: 2023-12-08 20:35:05
tags:
- opengl
- java
cover: https://view.moezx.cc/images/2018/03/14/18097514_p0.jpg
---

# 着色器

着色器是指在GPU上运行的小程序，专门负责图形渲染管线中的特定部分。这些程序的基本功能是将输入转换为输出。它们是高度独立的程序，无法直接相互通信，唯一的沟通方式是通过输入和输出传递信息。

# GLSL

着色器是用类似于C语言的GLSL编写的。GLSL专为图形计算而设计，它包含了许多有用的特性，尤其是针对向量和矩阵操作的功能。

每个着色器都有一个固定的结构：它们以声明版本开始，然后包括输入和输出变量、uniform变量，以及一个main函数作为入口点。在main函数中，处理所有的输入，进行所需的计算，并将结果输出到输出变量中。如果不了解uniform变量，不必担心，我们会在后续进行详细解释。

```c++
#version version_number
in type in_variable_name;
in type in_variable_name;

out type out_variable_name;

uniform type uniform_name;

int main()
{
  // 处理输入并进行一些图形操作
  ...
  // 输出处理过的结果到输出变量
  out_variable_name = weird_stuff_we_processed;
}
```

查询GL_MAX_VERTEX_ATTRIBS来获取具体的上限：

```c++
int nrAttributes;
glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &nrAttributes);
std::cout << "Maximum nr of vertex attributes supported: " << nrAttributes << std::endl;
```

- 着色器用GLSL语言编写，这是一种专门为图形计算设计的语言，它包含对向量和矩阵操作的特性。
- 着色器的结构是固定的：版本声明，输入和输出变量声明，uniform声明以及主函数main。主函数处理输入变量，输出结果到输出变量。
- 顶点着色器中的输入变量也被称为顶点属性(Vertex Attribute)。OpenGL规定至少有16个包含4个分量的顶点属性可用，但具体数量由硬件决定，可以用GL_MAX_VERTEX_ATTRIBS查询具体上限。通常情况下，返回的值至少是16，对大多数情况来说足够了。

## 数据类型

基础数据类型：`int`、`float`、`double`、`uint`和`bool`。两种容器类型，分别是向量(Vector)和矩阵(Matrix)

### 向量

| 类型    | 含义                            |
| :------ | :------------------------------ |
| `vecn`  | 包含`n`个float分量的默认向量    |
| `bvecn` | 包含`n`个bool分量的向量         |
| `ivecn` | 包含`n`个int分量的向量          |
| `uvecn` | 包含`n`个unsigned int分量的向量 |
| `dvecn` | 包含`n`个double分量的向量       |

一个向量的分量可以通过`vec.x`这种方式获取.你可以分别使用`.x`、`.y`、`.z`和`.w`来获取它们的第1、2、3、4个分量。GLSL也允许你对颜色使用`rgba`，或是对纹理坐标使用`stpq`访问相同的分量。

向量这一数据类型也允许一些有趣而灵活的分量选择方式，叫做重组(Swizzling)。重组允许这样的语法：

```glsl
vec2 someVec;
vec4 differentVec = someVec.xyxx;
vec3 anotherVec = differentVec.zyw;
vec4 otherVec = someVec.xxxx + anotherVec.yxzy;
```

```glsl
vec2 vect = vec2(0.5, 0.7);
vec4 result = vec4(vect, 0.0, 0.0);
vec4 otherResult = vec4(result.xyz, 1.0);
```

## 输入与输出

这段内容指出了着色器之间的数据传递和通信方式。尽管每个着色器是独立的小程序，但作为渲染管线的一部分，它们需要有输入和输出来进行数据传递和交流。GLSL通过in和out关键字定义了这种输入和输出。只要输出变量与下一个着色器阶段的输入匹配，数据就可以传递下去。

顶点着色器和片段着色器在输入和输出方面有所不同。顶点着色器从顶点数据中直接接收输入，并通过使用location元数据定义输入变量来配置顶点属性。这有助于在CPU上管理顶点数据属性。另一方面，片段着色器需要一个vec4颜色输出变量，因为它负责生成最终的颜色输出。如果片段着色器没有定义输出颜色，OpenGL会将物体渲染为黑色或白色。

要在着色器之间发送数据，需要在发送方着色器中声明一个输出，然后在接收方着色器中声明一个相匹配的输入。当这两个变量类型和名称一致时，OpenGL会将它们链接在一起，实现数据传递（这在链接程序对象时完成）。这种方法的优势是使数据传递更直观、减少了OpenGL调用，并提供了更好的代码可读性。

**顶点着色器**

```glsl
#version 330 core
layout (location = 0) in vec3 aPos; // 位置变量的属性位置值为0

out vec4 vertexColor; // 为片段着色器指定一个颜色输出

void main()
{
    gl_Position = vec4(aPos, 1.0); // 注意我们如何把一个vec3作为vec4的构造器的参数
    vertexColor = vec4(0.5, 0.0, 0.0, 1.0); // 把输出变量设置为暗红色
}
```

**片段着色器**

```c++
#version 330 core
out vec4 FragColor;

in vec4 vertexColor; // 从顶点着色器传来的输入变量（名称相同、类型相同）

void main()
{
    FragColor = vertexColor;
}
```

## Uniform


Uniform是一种让CPU应用程序向GPU着色器发送数据的方式。它与顶点属性不同：首先，uniform是全局的，每个着色器程序对象中的uniform必须是唯一的，可以被任意着色器在任意阶段访问。其次，无论设置什么值，uniform都会一直保存数据，直到重置或更新。

```c++
#version 330 core
out vec4 FragColor;

uniform vec4 ourColor; // 在OpenGL程序代码中设定这个变量

void main()
{
    FragColor = ourColor;
}
```



在GLSL着色器中，我们通过在类型和变量名前加上uniform关键字来声明uniform。这样就可以在着色器中使用它们了。我们可以通过uniform来设置三角形的颜色。

要更新uniform的值，首先要获取uniform的位置值，然后使用特定的函数设置它们的值。在OpenGL中，这些函数根据数据类型使用不同的后缀，比如glUniform4f用于设置四个浮点数的uniform。

```c++
float timeValue = glfwGetTime();
float greenValue = (sin(timeValue) / 2.0f) + 0.5f;
int vertexColorLocation = glGetUniformLocation(shaderProgram, "ourColor");
glUseProgram(shaderProgram);
glUniform4f(vertexColorLocation, 0.0f, greenValue, 0.0f, 1.0f);
```

| 后缀 | 含义                                 |
| :--- | :----------------------------------- |
| `f`  | 函数需要一个float作为它的值          |
| `i`  | 函数需要一个int作为它的值            |
| `ui` | 函数需要一个unsigned int作为它的值   |
| `3f` | 函数需要3个float作为它的值           |
| `fv` | 函数需要一个float向量/数组作为它的值 |

在渲染循环中，我们可以在每次迭代中更新uniform的值，让颜色随时间改变。这种方式可以让三角形颜色在渲染过程中变化。使用uniform对于在渲染迭代中变化的属性非常有用，也可以方便程序和着色器之间的数据交互。

```c++
while(!glfwWindowShouldClose(window))
{
    // 输入
    processInput(window);

    // 渲染
    // 清除颜色缓冲
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // 记得激活着色器
    glUseProgram(shaderProgram);

    // 更新uniform颜色
    float timeValue = glfwGetTime();
    float greenValue = sin(timeValue) / 2.0f + 0.5f;
    int vertexColorLocation = glGetUniformLocation(shaderProgram, "ourColor");
    glUniform4f(vertexColorLocation, 0.0f, greenValue, 0.0f, 1.0f);

    // 绘制三角形
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // 交换缓冲并查询IO事件
    glfwSwapBuffers(window);
    glfwPollEvents();
}
```





Java代码

```java
package org.example;

import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.opengl.GL;

import static org.lwjgl.glfw.GLFW.glfwGetTime;
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
            "uniform vec4 ourColor;\n" +
            "void main()\n" +
            "{\n" +
            "   FragColor = ourColor;\n" +
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

        // link shaders2
        int shaderProgram2 = glCreateProgram();
        glAttachShader(shaderProgram2,vertexShader);
        glLinkProgram(shaderProgram2);
        // check for linking errors
        success = glGetProgrami(shaderProgram2,GL_LINK_STATUS);
        if (success == 0){
            String log = glGetProgramInfoLog(shaderProgram2);
            System.err.println("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + log);
        }
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------

        float[] vertices = {
                0.5f, -0.5f, 0.0f,  // bottom right
                -0.5f, -0.5f, 0.0f,  // bottom left
                0.0f,  0.5f, 0.0f   // top
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

        // remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO; keep the EBO bound.
        // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

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
            glBindVertexArray(VAO);
            glUseProgram(shaderProgram);

            double timeValue = glfwGetTime();
            float greenValue = (float) Math.sin(timeValue/ 2.0 + 0.5);
            int vertexColorLocation = glGetUniformLocation(shaderProgram, "ourColor");
            glUniform4f(vertexColorLocation, 0.0f, greenValue, 0.0f, 1.0f);

            glDrawArrays(GL_TRIANGLES, 0, 3);
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

## 更多属性！

这段教程主要介绍了如何将颜色数据加入顶点数据，然后更新顶点着色器和片段着色器来处理这些新的数据。原先的三角形只有位置信息，现在我们将三个角分别指定为红色、绿色和蓝色。

```c++
float vertices[] = {
    // 位置              // 颜色
     0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f,   // 右下
    -0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,   // 左下
     0.0f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f    // 顶部
};
```



添加颜色数据后，我们需要更新顶点着色器以接收颜色值作为顶点属性输入，并更新片段着色器来使用新的输出变量来指定颜色。然后，重新配置顶点属性指针以便让OpenGL知道新的数据布局。

```glsl
#version 330 core
layout (location = 0) in vec3 aPos;   // 位置变量的属性位置值为 0 
layout (location = 1) in vec3 aColor; // 颜色变量的属性位置值为 1

out vec3 ourColor; // 向片段着色器输出一个颜色

void main()
{
    gl_Position = vec4(aPos, 1.0);
    ourColor = aColor; // 将ourColor设置为我们从顶点数据那里得到的输入颜色
}
```



```glsl
#version 330 core
out vec4 FragColor;  
in vec3 ourColor;

void main()
{
    FragColor = vec4(ourColor, 1.0);
}
```



最后，解释了片段插值的概念。在渲染过程中，光栅化阶段会创建比顶点更多的片段。这些片段在图形形状上的位置不同，片段着色器的输入属性会根据位置进行插值，比如线段上的端点颜色会在两端之间插值。在三角形渲染时，这些插值颜色导致了我们看到的不同颜色的效果。

```c++
// 位置属性
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
glEnableVertexAttribArray(0);
// 颜色属性
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3* sizeof(float)));
glEnableVertexAttribArray(1);
```

Java代码

```java
package org.example;

import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.opengl.GL;

import static org.lwjgl.glfw.GLFW.glfwGetTime;
import static org.lwjgl.opengl.GL11.glViewport;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.*;
import static org.lwjgl.opengl.GL30.glBindVertexArray;

public class Main {
    private static final int SCR_WIDTH = 800;
    private static final int SCR_HEIGHT = 600;

    static final String vertexShaderSource = "#version 330 core\n" +
            "layout (location = 0) in vec3 aPos;\n" +
            "layout (location = 1) in vec3 aColor;\n" +
            "out vec3 ourColor;\n" +
            "void main()\n" +
            "{\n" +
            "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n" +
            "   ourColor=aColor;\n" +
            "}\0";
    static final String fragmentShaderSource = "#version 330 core\n" +
            "in vec3 ourColor;\n" +
            "out vec4 FragColor;\n" +
            "void main()\n" +
            "{\n" +
            "   FragColor = vec4(ourColor,1.0);\n" +
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

        // link shaders2
        int shaderProgram2 = glCreateProgram();
        glAttachShader(shaderProgram2,vertexShader);
        glLinkProgram(shaderProgram2);
        // check for linking errors
        success = glGetProgrami(shaderProgram2,GL_LINK_STATUS);
        if (success == 0){
            String log = glGetProgramInfoLog(shaderProgram2);
            System.err.println("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + log);
        }
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------

        float[] vertices = {
                0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f,   // 右下
                -0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,   // 左下
                0.0f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f    // 顶部
        };


        int VBO = glGenBuffers();
        int VAO =  glGenVertexArrays();
        // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER,VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW);


        glVertexAttribPointer(0, 3, GL_FLOAT, false, 6 * Float.BYTES, 0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, false, 6 * Float.BYTES, 3*Float.BYTES);
        glEnableVertexAttribArray(1);

        // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO; keep the EBO bound.
        // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

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
            glBindVertexArray(VAO);
            glUseProgram(shaderProgram);

            double timeValue = glfwGetTime();
            float greenValue = (float) Math.sin(timeValue/ 2.0 + 0.5);
            int vertexColorLocation = glGetUniformLocation(shaderProgram, "ourColor");
            glUniform4f(vertexColorLocation, 0.0f, greenValue, 0.0f, 1.0f);

            glDrawArrays(GL_TRIANGLES, 0, 3);
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

# 我们自己的着色器类

这部分介绍了如何创建一个着色器类来简化着色器的编写、编译和管理。这个类可以从硬盘读取着色器源代码文件，编译并链接它们，并提供了一些工具函数来处理uniform变量。

在头文件中，使用了预处理指令来避免链接冲突。着色器类包含了着色器程序的ID，构造函数需要顶点和片段着色器源代码的文件路径。此外，类中还包含了用来激活着色器程序的use函数，以及一些用于设置uniform变量的工具函数。

```c++
#ifndef SHADER_H
#define SHADER_H

#include <glad/glad.h>; // 包含glad来获取所有的必须OpenGL头文件

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>


class Shader
{
public:
    // 程序ID
    unsigned int ID;

    // 构造器读取并构建着色器
    Shader(const char* vertexPath, const char* fragmentPath);
    // 使用/激活程序
    void use();
    // uniform工具函数
    void setBool(const std::string &name, bool value) const;  
    void setInt(const std::string &name, int value) const;   
    void setFloat(const std::string &name, float value) const;
};

#endif
```



## 从文件读取

这部分内容讲了如何使用C++文件流来读取着色器的内容，并把它们存储到字符串对象中。这个过程包括打开和读取文件，将文件内容转换为字符串，并关闭文件处理器。

接着介绍了编译和链接着色器的过程。顶点着色器和片段着色器都需要经历类似的编译流程，并且在编译和链接出错时会打印相关的错误信息。之后，将着色器附加到着色器程序对象上，并链接这些着色器。

着色器类提供了使用着色器和设置uniform变量的函数。`use()` 函数激活着色器程序，而 `setBool()`, `setInt()`, `setFloat()` 函数用于设置不同类型的uniform变量。

最后，通过示例展示了如何使用这个着色器类。在创建着色器对象之后，可以在程序中使用它们，设置uniform变量，然后绘制场景。

整个过程中的代码示例和文件路径提供了完整的指导，帮助理解如何使用新的着色器类。

### MyShader类

```java
package org.example;

import java.io.*;

import static org.lwjgl.opengl.GL20.*;

public class MyShader {
    public int ID;
    MyShader(String vertexPath,String fragmentPath){
        String vertexCode;
        String fragmentCode;
        try {
            BufferedReader vertexReader = new BufferedReader(new FileReader(getClass().getClassLoader().getResource(vertexPath).getFile()));
            BufferedReader fragmentReader = new BufferedReader(new FileReader(getClass().getClassLoader().getResource(fragmentPath).getFile()));

            String line;
            StringBuilder vertexBuilder  = new StringBuilder();
            StringBuilder fragmentBuilder = new StringBuilder();
            while ((line=vertexReader.readLine())!=null){
                vertexBuilder.append(line).append("\n");
            }
            while ((line=fragmentReader.readLine())!=null){
                fragmentBuilder.append(line).append("\n");
            }
            vertexCode = vertexBuilder.toString();
            fragmentCode = fragmentBuilder.toString();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        int vertex,fragment;
        // compile shaders
        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex,vertexCode);
        glCompileShader(vertex);
        checkCompileErrors(vertex,"VERTEX");
        // fragment shader
        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment,fragmentCode);
        glCompileShader(fragment);
        checkCompileErrors(fragment,"FRAGMENT");
        // shader program
        ID = glCreateProgram();
        glAttachShader(ID,vertex);
        glAttachShader(ID,fragment);
        glLinkProgram(ID);
        checkCompileErrors(ID,"PROGRAM");
        // delete shaders
        glDeleteShader(vertex);
        glDeleteShader(fragment);

    }
    // 使用/激活程序
    public void use(){
        glUseProgram(ID);
    }

    private void checkCompileErrors(int shader,String type){
        int success;
        String infoLog;
        if(type != "PROGRAM"){
            success = glGetShaderi(shader,GL_COMPILE_STATUS);
            if(success == 0){
                infoLog = glGetShaderInfoLog(shader);
                System.err.println("ERROR::SHADER_COMPILATION_ERROR of type: " + type + "\n" + infoLog);
            }
        }else{
            success = glGetProgrami(shader,GL_LINK_STATUS);
            if(success == 0){
                infoLog = glGetProgramInfoLog(shader);
                System.err.println("ERROR::PROGRAM_LINKING_ERROR of type: " + type + "\n" + infoLog);
            }
        }
    }
    // uniform工具函数
    public void setBool(String name,int value){
        glUniform1i(glGetUniformLocation(ID, name), value);
    }
    public void setInt(String name,int value){
        glUniform1i(glGetUniformLocation(ID, name), value);
    }

    public void setFloat(String name,float value){
        glUniform1f(glGetUniformLocation(ID, name), value);
    }
}

```

### Main类

```java
package org.example;

import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.opengl.GL;

import static org.lwjgl.glfw.GLFW.glfwGetTime;
import static org.lwjgl.opengl.GL11.glViewport;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.*;
import static org.lwjgl.opengl.GL30.glBindVertexArray;

public class Main {
    private static final int SCR_WIDTH = 800;
    private static final int SCR_HEIGHT = 600;

    static final String vertexShaderSource = "vertex.glsl";
    static final String fragmentShaderSource = "fragment.glsl";

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


        MyShader shaderProgram = new MyShader(vertexShaderSource,fragmentShaderSource);

        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------

        float[] vertices = {
                0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  // bottom right
                -0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  // bottom left
                0.0f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f   // top
        };


        int VBO = glGenBuffers();
        int VAO =  glGenVertexArrays();
        // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER,VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW);


        glVertexAttribPointer(0, 3, GL_FLOAT, false, 6 * Float.BYTES, 0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, false, 6 * Float.BYTES, 3*Float.BYTES);
        glEnableVertexAttribArray(1);

        // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
//        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO; keep the EBO bound.
        // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
        // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
//        glBindVertexArray(0);


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
            shaderProgram.use();
            glBindVertexArray(VAO);
            double timeValue = glfwGetTime();
            glDrawArrays(GL_TRIANGLES, 0, 3);
            // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
            // -------------------------------------------------------------------------------
            GLFW.glfwSwapBuffers(window);
            GLFW.glfwPollEvents();
        }
        // optional: de-allocate all resources once they've outlived their purpose:
        // ------------------------------------------------------------------------
        glDeleteVertexArrays(VAO);
        glDeleteBuffers(VBO);
        glDeleteProgram(shaderProgram.ID);

        // glfw: terminate, clearing all previously allocated GLFW resources.
        // ------------------------------------------------------------------
        GLFW.glfwTerminate();
    }
}
```

# 练习

1. 修改顶点着色器让三角形上下颠倒：[参考解答](https://learnopengl.com/code_viewer.php?code=getting-started/shaders-exercise1)
2. 使用uniform定义一个水平偏移量，在顶点着色器中使用这个偏移量把三角形移动到屏幕右侧：[参考解答](https://learnopengl.com/code_viewer.php?code=getting-started/shaders-exercise2)
3. 使用`out`关键字把顶点位置输出到片段着色器，并将片段的颜色设置为与顶点位置相等（来看看连顶点位置值都在三角形中被插值的结果）。做完这些后，尝试回答下面的问题：为什么在三角形的左下角是黑的?：[参考解答](https://learnopengl.com/code_viewer.php?code=getting-started/shaders-exercise3)





```glsl
#version 330 core
layout (location = 0) in vec3 aPos;   // 位置变量的属性位置值为 0
layout (location = 1) in vec3 aColor; // 颜色变量的属性位置值为 1

out vec3 ourColor; // 向片段着色器输出一个颜色

void main()
{
    gl_Position = vec4(aPos.x,-aPos.y,aPos.z, 1.0);
    ourColor = aColor; // 将ourColor设置为我们从顶点数据那里得到的输入颜色
}
```

```java
java === 
float offset = 0.5f;
            shaderProgram.setFloat("xOffset",offset);


glsl ===
    #version 330 core
layout (location = 0) in vec3 aPos;   // 位置变量的属性位置值为 0
layout (location = 1) in vec3 aColor; // 颜色变量的属性位置值为 1

out vec3 ourColor; // 向片段着色器输出一个颜色

uniform float xOffset;
void main()
{
    gl_Position = vec4(aPos.x + xOffset,-aPos.y,aPos.z, 1.0);
    ourColor = aColor; // 将ourColor设置为我们从顶点数据那里得到的输入颜色
}
```



```java
#version 330 core
layout (location = 0) in vec3 aPos;   // 位置变量的属性位置值为 0
layout (location = 1) in vec3 aColor; // 颜色变量的属性位置值为 1

out vec3 ourColor; // 向片段着色器输出一个颜色

void main()
{
    gl_Position = vec4(aPos.x,aPos.y,aPos.z, 1.0);
    ourColor = aPos; // 将ourColor设置为我们从顶点数据那里得到的输入颜色
}
```

黑是因为左下是0，0，0，1
