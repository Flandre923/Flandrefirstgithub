---
title: opengl06变换
date: 2023-12-10 16:55:41
tags:
- opengl
- java
cover: https://view.moezx.cc/images/2017/12/16/31.jpg
---





# 变换



强调了使用矩阵进行变换的重要性，并暗示了它们作为处理物体移动的更有效方法。它指出了矩阵作为数学工具的重要性.

关于向量和矩阵的理解是图形编程中的基础，它们用于描述和控制物体在三维空间中的位置、旋转和缩放。即使对初学者来说，这些概念可能有些复杂，但它们是非常有用的工具。

# 向量

1. **向量基本定义：**
   - 向量具有方向和大小，可以理解为指示方向和长度的箭头。
   - 向量可在任意维度上存在，但通常用于二至四维空间。二维向量对应平面方向，三维向量可表示三维空间方向。
2. **向量的表示和理解：**
   - 用箭头 `(x, y)` 在二维图中展示向量。在二维图像中表示向量更直观。
   - 数学上通常使用带上箭头的字母来表示向量（例如：𝑣¯）。
3. **位置向量：**
   - 为了更形象地表示方向，将向量起点设为原点（0, 0）并指向某一点，形成位置向量（也可从其他点起始指向另一点）。
4. **向量运算：**
   - 向量支持多种运算，类似于数字运算，包括加法、减法、数乘等。

## 向量与标量运算

## 向量取反

## 向量加减

## 长度

## 向量相乘

### 点乘

### 叉乘

# 矩阵

## 矩阵的加减

## 矩阵的数乘

## 矩阵相乘

# 矩阵与向量相乘

## 单位矩阵

## 缩放


这段内容总结了对向量进行缩放的概念，介绍了缩放的基本原理和使用缩放矩阵进行变换的方法。

1. **向量的缩放操作：**
   - 缩放一个向量就是改变它的长度，而保持其方向不变。
   - 在2D或3D操作中，可以分别对每个轴（x、y或z）定义一个缩放因子，以改变向量在该轴上的长度。
2. **非均匀缩放和均匀缩放：**
   - 非均匀缩放是每个轴的缩放因子不同，导致在每个方向上的缩放不同。
   - 均匀缩放是每个轴的缩放因子相同，使得在所有方向上的缩放都一致。
3. **缩放矩阵：**
   - 通过构建缩放矩阵，可以对向量进行缩放操作。矩阵对角线上的元素分别与向量对应元素相乘，实现向量的缩放。
   - 缩放矩阵可以将任意向量 (x, y, z) 缩放为 (S1 * x, S2 * y, S3 * z)，注意最后的分量仍然为1（在3D空间中对w分量的缩放通常是没有意义的）。

![image-20231210170543614](https://s2.loli.net/2023/12/10/CpFIXOvRUMjxg1L.png)

## 位移

1. **位移操作：**
   - 位移是在原始向量基础上加上另一个向量，使得得到一个新的向量在不同位置上。
   - 这是向量加法的一种运用，用来移动原始向量到另一个位置。
2. **位移矩阵：**
   - 在4×4矩阵中，位移矩阵的特殊位置是第四列最上面的3个值。
   - 位移矩阵可以用来对向量进行位移操作，将位移向量 (Tx, Ty, Tz) 加到原始向量上，即 (x + Tx, y + Ty, z + Tz)。
3. **齐次坐标（Homogeneous Coordinates）：**
   - 向量的第四个分量w，也称为齐次坐标，是一种特殊的坐标。
   - 通过齐次坐标，可以将齐次向量转换为三维向量，方法是将x、y和z坐标分别除以w坐标。
   - 齐次坐标的好处是允许在3D空间进行向量的位移操作。
4. **方向向量和位移矩阵的作用：**
   - 当一个向量的齐次坐标w为0时，它是一个方向向量，因为它不会进行位移操作。
   - 位移矩阵是变换工具箱中的重要一环，用于在三个方向（x、y、z）上移动物体。

## 旋转

1. **旋转概念：**
   - 旋转是用角度来表示的，可以是角度制或弧度制。
   - 通过三角学，可以将一个向量绕特定轴旋转特定角度。
2. **旋转表示方法：**
   - 在2D空间中，旋转可以用角度来描述，例如右旋转72度。
   - 在3D空间中，旋转需要定义一个角度和旋转轴，比如绕z轴旋转。
3. **旋转矩阵：**
   - 旋转矩阵是用来在3D空间中进行旋转操作的工具。
   - 不同轴的旋转矩阵定义了绕x、y、z轴旋转的方式，其中θ表示旋转角度。
4. **万向节死锁和四元数：**
   - 多次对旋转矩阵进行复合可能导致万向节死锁问题，这是一种旋转约束的问题。
   - 解决万向节死锁问题的方法之一是使用四元数，它更安全且计算效率更高，但对数学要求较高。

![image-20231210172000361](https://s2.loli.net/2023/12/10/9MSDeJw18dybn3W.png)

![image-20231210172014246](https://s2.loli.net/2023/12/10/xwvoASUjkfDLqtQ.png)

## 矩阵的组合

1. **矩阵组合：**
   - 通过矩阵相乘，可以将多个变换操作组合成一个矩阵，实现对顶点的复合变换。
   - 顺序很重要，因为矩阵乘法不遵守交换律，操作的顺序影响最终的变换效果。
2. **示例：**
   - 如果我们要对一个顶点进行缩放2倍，然后再位移(1, 2, 3)个单位，可以将这两个变换操作合并到一个矩阵中。
   - 结果的变换矩阵是先位移矩阵再缩放矩阵的乘积。
   - 在应用这个变换矩阵到一个顶点时，将该顶点乘以这个变换矩阵会得到最终的变换结果。

# 实践

现在我们已经解释了变换背后的所有理论，是时候将这些知识利用起来了。OpenGL没有自带任何的矩阵和向量知识，所以我们必须定义自己的数学类和函数。在教程中我们更希望抽象所有的数学细节，使用已经做好了的数学库。幸运的是，有个易于使用，专门为OpenGL量身定做的数学库，那就是GLM。

## GLM

使用这个：

```java
// https://mvnrepository.com/artifact/org.joml/joml
    implementation 'org.joml:joml:1.10.5'
```



1. **GLM库简介**:
   - GLM是OpenGL Mathematics的缩写，是一个只有头文件的库，无需链接和编译，可以从官网下载。
   - 下载后，将头文件的根目录复制到项目的includes文件夹中，即可使用该库。
2. **GLM版本差异**:
   - 从0.9.9版本开始，GLM默认将矩阵类型初始化为零矩阵而不是单位矩阵。
   - 如果使用0.9.9或更高版本，需要将所有矩阵初始化改为 `glm::mat4(1.0f)`。
3. **GLM功能**:
   - GLM的大多数功能都在以下三个头文件中：`<glm/glm.hpp>`, `<glm/gtc/matrix_transform.hpp>`, `<glm/gtc/type_ptr.hpp>`。
4. **向量和矩阵变换**:
   - 通过GLM进行向量和矩阵变换的例子：如何将向量 (1, 0, 0) 进行位移 (1, 1, 0) 个单位。
   - 使用 `glm::vec4` 和 `glm::mat4` 来定义向量和矩阵，使用 `glm::translate` 进行位移操作。
5. **旋转和缩放**:
   - 展示如何对一个对象进行逆时针旋转90度和缩小为原来的一半。
   - 使用 `glm::rotate` 和 `glm::scale` 来实现旋转和缩放操作。
6. **矩阵传递给着色器**:
   - 修改顶点着色器，使其接收 `mat4` 类型的 uniform 变量。
   - 使用 `glUniformMatrix4fv` 函数将变换矩阵数据传递给着色器。
7. **动态更新变换**:
   - 在游戏循环中更新变换矩阵，以实现对象随时间旋转和移动的效果。
   - 使用 `glfwGetTime()` 获取不同时间点的角度，动态更新变换矩阵。
8. **总结和下一步**:
   - 矩阵在图形领域中是重要的工具，能够组合多个变换为一个矩阵并重复使用。
   - 在着色器中使用矩阵可以节省处理时间，避免重新定义顶点数据。

java代码1 

```java
package org.example;

import org.joml.Matrix4f;
import org.lwjgl.BufferUtils;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.opengl.GL;

import java.awt.*;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import static org.lwjgl.opengl.GL11.glViewport;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.*;
import static org.lwjgl.opengl.GL30.glBindVertexArray;

public class Main {
    private static final int SCR_WIDTH = 800;
    private static final int SCR_HEIGHT = 600;

    static final String vertexShaderSource = "vertex.glsl";
    static final String fragmentShaderSource = "fragment.glsl";

    public static void main(String[] args) throws IOException {
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
                // positions          // colors           // texture coords
                0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
                0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
                -0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
                -0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left
        };
        int[] indices = {
                0, 1, 3, // first triangle
                1, 2, 3  // second triangle
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

        glVertexAttribPointer(0, 3, GL_FLOAT, false, 8 * Float.BYTES, 0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, false, 8 * Float.BYTES, 3*Float.BYTES);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 3, GL_FLOAT, false, 8 * Float.BYTES, 6*Float.BYTES);
        glEnableVertexAttribArray(2);

        // load and create a texture
        // -------------------------
        int texture = glGenTextures();
        glBindTexture(GL_TEXTURE_2D, texture);// all upcoming GL_TEXTURE_2D operations now have effect on this texture object
        // set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);// set texture wrapping to GL_REPEAT (default wrapping method)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
        // set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        // load image, create texture and generate mipmaps
        ImageReader.ImageData image1 = ImageReader.ReadImage("src/main/resources/container.jpg");

        if (image1!=null){
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image1.width, image1.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image1.data);
            glGenerateMipmap(GL_TEXTURE_2D);
        }else{
            System.out.println("Failed to load texture" );
        }

        int texture2 = glGenTextures();
        glBindTexture(GL_TEXTURE_2D, texture2);// all upcoming GL_TEXTURE_2D operations now have effect on this texture object
        // set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);// set texture wrapping to GL_REPEAT (default wrapping method)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
        // set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        // load image, create texture and generate mipmaps
        ImageReader.ImageData image2 = ImageReader.ReadImage("src/main/resources/awesomeface.png");

        if (image1!=null){
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image2.width, image2.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image2.data);
            glGenerateMipmap(GL_TEXTURE_2D);
        }else{
            System.out.println("Failed to load texture" );
        }


        shaderProgram.use(); // 不要忘记在设置uniform变量之前激活着色器程序！
        glUniform1i(glGetUniformLocation(shaderProgram.ID, "texture1"), 0); // 手动设置
        shaderProgram.setInt("texture2", 1); // 或者使用着色器类设置

        float mixValue = 0.2f;

        Matrix4f trans = new Matrix4f();
        trans =  trans.rotate(3.14f/2f,0f,0f,1f);
        trans = trans.scale(0.5f,0.5f,0.5f);

        FloatBuffer buffer = BufferUtils.createFloatBuffer(16); // Assuming a 4x4 matrix
        trans.get(buffer); // Assuming 'trans' is a FloatBuffer containing the matrix data



        while(!GLFW.glfwWindowShouldClose(window)){
            // input
            // -----
            if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_ESCAPE) == GLFW.GLFW_PRESS) {
                GLFW.glfwSetWindowShouldClose(window, true);
            }

            if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_UP) == GLFW.GLFW_PRESS)
            {
                mixValue += 0.001f; // change this value accordingly (might be too slow or too fast based on system hardware)
                if(mixValue >= 1.0f)
                    mixValue = 1.0f;
            }
            if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_DOWN) == GLFW.GLFW_PRESS)
            {
                mixValue -= 0.001f; // change this value accordingly (might be too slow or too fast based on system hardware)
                if (mixValue <= 0.0f)
                    mixValue = 0.0f;
            }


            // render
            // -----
            glClearColor(0.2f,0.2f,0.2f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            // draw our first triangle
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, texture2);

            shaderProgram.use();
            shaderProgram.setFloat("mixValue", mixValue);

            int transformLoc = glGetUniformLocation(shaderProgram.ID, "transform");
            glUniformMatrix4fv(transformLoc, false, buffer);

            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
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
        glDeleteProgram(shaderProgram.ID);

        // glfw: terminate, clearing all previously allocated GLFW resources.
        // ------------------------------------------------------------------
        GLFW.glfwTerminate();
    }
}
```

glsl

```glsl
#version 330 core
out vec4 FragColor;
in vec3 ourColor;
in vec2 TexCoord;

uniform sampler2D texture1;
uniform sampler2D texture2;

uniform float mixValue;

void main()
{
    FragColor = mix(texture(texture1, TexCoord), texture(texture2, vec2(1.0-TexCoord.x,TexCoord.y)), mixValue);
}
```





Java渲染代码

```java
package org.example;

import org.joml.Matrix4f;
import org.lwjgl.BufferUtils;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.opengl.GL;

import java.awt.*;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import static org.lwjgl.opengl.GL11.glViewport;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.*;
import static org.lwjgl.opengl.GL30.glBindVertexArray;

public class Main {
    private static final int SCR_WIDTH = 800;
    private static final int SCR_HEIGHT = 600;

    static final String vertexShaderSource = "vertex.glsl";
    static final String fragmentShaderSource = "fragment.glsl";

    public static void main(String[] args) throws IOException {
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
                // positions          // colors           // texture coords
                0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
                0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
                -0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
                -0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left
        };
        int[] indices = {
                0, 1, 3, // first triangle
                1, 2, 3  // second triangle
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

        glVertexAttribPointer(0, 3, GL_FLOAT, false, 8 * Float.BYTES, 0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, false, 8 * Float.BYTES, 3*Float.BYTES);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 3, GL_FLOAT, false, 8 * Float.BYTES, 6*Float.BYTES);
        glEnableVertexAttribArray(2);

        // load and create a texture
        // -------------------------
        int texture = glGenTextures();
        glBindTexture(GL_TEXTURE_2D, texture);// all upcoming GL_TEXTURE_2D operations now have effect on this texture object
        // set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);// set texture wrapping to GL_REPEAT (default wrapping method)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
        // set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        // load image, create texture and generate mipmaps
        ImageReader.ImageData image1 = ImageReader.ReadImage("src/main/resources/container.jpg");

        if (image1!=null){
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image1.width, image1.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image1.data);
            glGenerateMipmap(GL_TEXTURE_2D);
        }else{
            System.out.println("Failed to load texture" );
        }

        int texture2 = glGenTextures();
        glBindTexture(GL_TEXTURE_2D, texture2);// all upcoming GL_TEXTURE_2D operations now have effect on this texture object
        // set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);// set texture wrapping to GL_REPEAT (default wrapping method)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
        // set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        // load image, create texture and generate mipmaps
        ImageReader.ImageData image2 = ImageReader.ReadImage("src/main/resources/awesomeface.png");

        if (image1!=null){
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image2.width, image2.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image2.data);
            glGenerateMipmap(GL_TEXTURE_2D);
        }else{
            System.out.println("Failed to load texture" );
        }


        shaderProgram.use(); // 不要忘记在设置uniform变量之前激活着色器程序！
        glUniform1i(glGetUniformLocation(shaderProgram.ID, "texture1"), 0); // 手动设置
        shaderProgram.setInt("texture2", 1); // 或者使用着色器类设置

        float mixValue = 0.2f;



        while(!GLFW.glfwWindowShouldClose(window)){
            // input
            // -----
            if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_ESCAPE) == GLFW.GLFW_PRESS) {
                GLFW.glfwSetWindowShouldClose(window, true);
            }

            if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_UP) == GLFW.GLFW_PRESS)
            {
                mixValue += 0.001f; // change this value accordingly (might be too slow or too fast based on system hardware)
                if(mixValue >= 1.0f)
                    mixValue = 1.0f;
            }
            if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_DOWN) == GLFW.GLFW_PRESS)
            {
                mixValue -= 0.001f; // change this value accordingly (might be too slow or too fast based on system hardware)
                if (mixValue <= 0.0f)
                    mixValue = 0.0f;
            }


            // render
            // -----
            glClearColor(0.2f,0.2f,0.2f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            // draw our first triangle
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, texture2);

            Matrix4f trans = new Matrix4f();
            trans = trans.scale(0.5f,0.5f,0f);
            trans =  trans.rotate((float) GLFW.glfwGetTime(),0f,0f,1f);

            FloatBuffer buffer = BufferUtils.createFloatBuffer(16); // Assuming a 4x4 matrix

            trans.get(buffer); // Assuming 'trans' is a FloatBuffer containing the matrix data
            shaderProgram.use();
            shaderProgram.setFloat("mixValue", mixValue);

            int transformLoc = glGetUniformLocation(shaderProgram.ID, "transform");
            glUniformMatrix4fv(transformLoc, false, buffer);

            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
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
        glDeleteProgram(shaderProgram.ID);

        // glfw: terminate, clearing all previously allocated GLFW resources.
        // ------------------------------------------------------------------
        GLFW.glfwTerminate();
    }
}
```







练习2

```java
package org.example;

import org.joml.Matrix4f;
import org.lwjgl.BufferUtils;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.opengl.GL;

import java.awt.*;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import static org.lwjgl.opengl.GL11.glViewport;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.*;
import static org.lwjgl.opengl.GL30.glBindVertexArray;

public class Main {
    private static final int SCR_WIDTH = 800;
    private static final int SCR_HEIGHT = 600;

    static final String vertexShaderSource = "vertex.glsl";
    static final String fragmentShaderSource = "fragment.glsl";

    public static void main(String[] args) throws IOException {
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
                // positions          // colors           // texture coords
                0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
                0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
                -0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
                -0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left
        };
        int[] indices = {
                0, 1, 3, // first triangle
                1, 2, 3  // second triangle
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

        glVertexAttribPointer(0, 3, GL_FLOAT, false, 8 * Float.BYTES, 0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, false, 8 * Float.BYTES, 3*Float.BYTES);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 3, GL_FLOAT, false, 8 * Float.BYTES, 6*Float.BYTES);
        glEnableVertexAttribArray(2);

        // load and create a texture
        // -------------------------
        int texture = glGenTextures();
        glBindTexture(GL_TEXTURE_2D, texture);// all upcoming GL_TEXTURE_2D operations now have effect on this texture object
        // set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);// set texture wrapping to GL_REPEAT (default wrapping method)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
        // set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        // load image, create texture and generate mipmaps
        ImageReader.ImageData image1 = ImageReader.ReadImage("src/main/resources/container.jpg");

        if (image1!=null){
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image1.width, image1.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image1.data);
            glGenerateMipmap(GL_TEXTURE_2D);
        }else{
            System.out.println("Failed to load texture" );
        }

        int texture2 = glGenTextures();
        glBindTexture(GL_TEXTURE_2D, texture2);// all upcoming GL_TEXTURE_2D operations now have effect on this texture object
        // set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);// set texture wrapping to GL_REPEAT (default wrapping method)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
        // set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        // load image, create texture and generate mipmaps
        ImageReader.ImageData image2 = ImageReader.ReadImage("src/main/resources/awesomeface.png");

        if (image1!=null){
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image2.width, image2.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image2.data);
            glGenerateMipmap(GL_TEXTURE_2D);
        }else{
            System.out.println("Failed to load texture" );
        }


        shaderProgram.use(); // 不要忘记在设置uniform变量之前激活着色器程序！
        glUniform1i(glGetUniformLocation(shaderProgram.ID, "texture1"), 0); // 手动设置
        shaderProgram.setInt("texture2", 1); // 或者使用着色器类设置

        float mixValue = 0.2f;



        while(!GLFW.glfwWindowShouldClose(window)){
            // input
            // -----
            if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_ESCAPE) == GLFW.GLFW_PRESS) {
                GLFW.glfwSetWindowShouldClose(window, true);
            }

            if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_UP) == GLFW.GLFW_PRESS)
            {
                mixValue += 0.001f; // change this value accordingly (might be too slow or too fast based on system hardware)
                if(mixValue >= 1.0f)
                    mixValue = 1.0f;
            }
            if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_DOWN) == GLFW.GLFW_PRESS)
            {
                mixValue -= 0.001f; // change this value accordingly (might be too slow or too fast based on system hardware)
                if (mixValue <= 0.0f)
                    mixValue = 0.0f;
            }


            // render
            // -----
            glClearColor(0.2f,0.2f,0.2f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            // draw our first triangle
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, texture2);

            Matrix4f trans = new Matrix4f();
            trans = trans.scale(0.5f,0.5f,0f);
            trans =  trans.rotate((float) GLFW.glfwGetTime(),0f,0f,1f);

            FloatBuffer buffer = BufferUtils.createFloatBuffer(16); // Assuming a 4x4 matrix

            trans.get(buffer); // Assuming 'trans' is a FloatBuffer containing the matrix data
            shaderProgram.use();
            shaderProgram.setFloat("mixValue", mixValue);

            int transformLoc = glGetUniformLocation(shaderProgram.ID, "transform");
            glUniformMatrix4fv(transformLoc, false, buffer);

            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            Matrix4f trans2 = new Matrix4f();
            trans2 = trans2.translate(1,0,0).scale(0.5f,0.5f,0f);
            trans2.get(buffer);
            glUniformMatrix4fv(transformLoc, false, buffer);

            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
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
        glDeleteProgram(shaderProgram.ID);

        // glfw: terminate, clearing all previously allocated GLFW resources.
        // ------------------------------------------------------------------
        GLFW.glfwTerminate();
    }
}
```

