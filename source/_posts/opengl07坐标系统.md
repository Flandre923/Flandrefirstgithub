---
title: opengl07坐标系统
date: 2023-12-10 20:56:56
tags:
- opengl
- java
cover: https://view.moezx.cc/images/2017/12/16/ef4798d4c975040f.jpg
---

# 坐标系统

1. **局部空间(Local Space) / 物体空间(Object Space)**:
   - 物体自身坐标系，物体的初始位置定义在这个空间中。
2. **世界空间(World Space)**:
   - 物体在全局环境中的位置和方向。
3. **观察空间(View Space) / 视觉空间(Eye Space)**:
   - 从摄像机或观察者的角度观察物体的空间，相当于从一个固定的视角观察物体。
4. **裁剪空间(Clip Space)**:
   - 坐标范围已被裁剪至可视范围内，在这里执行透视投影。
5. **屏幕空间(Screen Space)**:
   - 最终的2D屏幕坐标，在此处顶点被映射为屏幕上的像素。

## 概述

在坐标系转换中，关键是利用多个变换矩阵来将坐标从一个坐标系转换到另一个。其中，最关键的三个矩阵是模型(Model)、观察(View)、投影(Projection)矩阵。坐标的起点是局部空间（Local Space），在转换过程中它会变为局部坐标（Local Coordinate），接着变为世界坐标（World Coordinate），观察坐标（View Coordinate），裁剪坐标（Clip Coordinate），最后以屏幕坐标（Screen Coordinate）结束。

1. **局部坐标**：物体相对于局部原点的坐标，也是物体初始的坐标。
2. **世界坐标**：将局部坐标转换为全局环境中的位置和方向。
3. **观察坐标**：从观察者或摄像机的视角观察物体的坐标。
4. **裁剪坐标**：裁剪至可视范围内的坐标，范围为-1.0到1.0，判断哪些顶点将呈现在屏幕上。
5. **屏幕坐标**：最终映射到屏幕上的2D像素坐标。

转换过程是：

- 将局部坐标转换为世界坐标，使物体处于更大的空间范围内，相对于全局原点摆放。
- 将世界坐标转换为观察坐标，以观察者的角度观察物体。
- 将观察坐标转换为裁剪坐标，在此范围内裁剪坐标，并判断哪些顶点将呈现在屏幕上。
- 将裁剪坐标转换为屏幕坐标，使用视口变换将其转换到由glViewport函数定义的坐标范围内，并最终将其转化为片段。

![image-20231210211512149](https://s2.loli.net/2023/12/10/5wkbJ2RIoyWjY4U.png)

## 局部空间

局部空间指的是物体所处的初始坐标空间，即对象最初所在的位置。想象在建模软件（例如Blender）中创建了一个立方体。即使立方体的原点可能位于(0, 0, 0)，但在程序中，它可能会处于完全不同的位置。甚至你创建的所有模型都以(0, 0, 0)作为初始位置，但它们最终会出现在世界的不同位置。因此，模型的所有顶点都处于局部空间中：它们相对于物体都是局部的。

我们使用的那个箱子的顶点被设定在范围为-0.5到0.5的坐标范围内，其中(0, 0)是它的原点。这些坐标都是局部坐标。

## 世界空间



## 观察空间

## 裁剪空间

在顶点着色器的末尾，OpenGL希望所有坐标都落在特定范围内，超出这一范围的点会被裁剪掉。裁剪空间的名字就来自于此。

为了让坐标适应OpenGL期望的范围，我们定义自己的坐标集，并将其转换回标准化设备坐标系。

将观察空间中的顶点坐标转换到裁剪空间时，需要使用投影矩阵。投影矩阵定义了坐标的范围，比如在每个维度上的-1000到1000。它将这个范围内的坐标变换为标准化设备坐标范围(-1.0, 1.0)内。超出范围的坐标会被裁剪掉。

观察箱是由投影矩阵创建的平截头体。只有位于平截头体范围内的坐标最终会呈现在屏幕上。投影矩阵的作用在于将3D坐标投影到容易映射到2D标准化设备坐标的空间。

透视除法是将4D裁剪空间坐标转换为3D标准化设备坐标的过程，即将位置向量的x，y，z分量分别除以向量的齐次w分量。这个步骤会在顶点着色器的最后自动执行。

一旦顶点被变换到裁剪空间，坐标会映射到屏幕空间（使用glViewport指定的设置），然后转换为片段。

投影矩阵可以采用两种形式：正交投影矩阵和透视投影矩阵，每种定义了不同的平截头体，影响最终的显示效果。

### 正射投影

正交投影矩阵定义了一个类似立方体的平截头箱，裁剪空间之外的顶点会被裁剪掉。这种投影矩阵需要确定可见平截头体的宽、高和长度。在此平截头体内的所有坐标不会被裁剪。其平截头体看起来像容器。

平截头体由宽度、高度、近平面和远平面定义。任何在近平面之前或远平面之后的坐标都会被裁剪。正射投影将内部坐标直接映射为标准化设备坐标，因为向量的w分量没有改变；如果w分量为1.0，透视除法不会影响坐标。

使用GLM的`glm::ortho`函数创建正射投影矩阵：

```
cppCopy code
glm::ortho(0.0f, 800.0f, 0.0f, 600.0f, 0.1f, 100.0f);
```

前两个参数定义平截头体的左右坐标，接着两个参数定义底部和顶部。这四个参数确定近平面和远平面的大小，而最后两个参数确定了这两个平面的距离。该投影矩阵将指定范围内的坐标变换为标准化设备坐标。

尽管正射投影矩阵直接映射到2D平面，但实际上直接使用它会得到不真实的结果，因为没有考虑透视。这就需要透视投影矩阵来解决。

### 透视投影

透视效果源于现实生活中物体距离观察者的远近导致物体看起来大小不同。透视投影矩阵模拟这种效果，修改顶点坐标的w值以使远离观察者的坐标变小。OpenGL要求所有可见坐标都在-1.0到1.0之间，透视除法则将裁剪空间坐标映射到标准化设备坐标。

使用GLM的`glm::perspective`可以创建透视投影矩阵。该函数定义可视空间的大平截头体，裁剪空间外的物体将被裁剪。透视平截头体可以看作是不均匀的箱子，内部的坐标映射到裁剪空间的点。

参数设置视野大小、宽高比、平截头体的近和远平面。通常，fov值设置为45.0f获得真实感，较大的值会呈现末日风格效果。近平面通常设为0.1f，远平面设为100.0f。

正射投影直接映射到裁剪空间而不考虑透视，远处物体与近处大小相同，通常用于二维渲染或需要精确绘制的场景。某些三维建模软件如Blender有时也用正射投影，因为它准确地描绘了各个维度下的物体。在Blender中，透视投影使得远处物体看起来更小，而正射投影保持均匀大小。

这两种投影方式的对比可以在Blender中清晰看到，透视投影下，远处物体较小，而在正射投影中每个物体的大小保持一致。

投影矩阵

```c++
glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)width/(float)height, 0.1f, 100.0f);

```

它的第一个参数定义了fov的值，它表示的是视野(Field of View)，并且设置了观察空间的大小。如果想要一个真实的观察效果，它的值通常设置为45.0f，但想要一个末日风格的结果你可以将其设置一个更大的值。第二个参数设置了宽高比，由视口的宽除以高所得。第三和第四个参数设置了平截头体的**近**和**远**平面。我们通常设置近距离为0.1f，而远距离设为100.0f。所有在近平面和远平面内且处于平截头体内的顶点都会被渲染。

# 进入3D

在进行3D绘图前，我们创建模型矩阵实现位移、缩放和旋转操作，将顶点变换到世界空间。观察矩阵则模拟摄像机移动，让整个场景移动到相反的方向。OpenGL使用右手坐标系，其中x轴在右侧，y轴向上，z轴朝后。为了理解右手坐标系，可以尝试使用右臂的手指指向不同方向。

模型矩阵通过旋转使平面向地板倾斜，观察矩阵则沿z轴负方向移动场景，模拟后退感。投影矩阵用于透视投影，在顶点着色器中将顶点坐标乘以这些矩阵完成变换。Uniform变量将这些矩阵传递给着色器，允许顶点坐标进行变换。

最终物体：

- 会稍微向后倾斜至地板方向。
- 与观察者有一定距离。
- 具有透视效果，离观察者越远，物体越小。

这种设置可以让一个平面看起来像一个静止的3D对象放在虚构的地板上



java代码

```java

            FloatBuffer modelBuffer = BufferUtils.createFloatBuffer(16); // Assuming a 4x4 matrix
            FloatBuffer viewBuffer = BufferUtils.createFloatBuffer(16); // Assuming a 4x4 matrix
            FloatBuffer ProjectionBuffer = BufferUtils.createFloatBuffer(16); // Assuming a 4x4 matrix
            Matrix4f model = new Matrix4f().rotate(-55f * 2f * 3.14f/360f ,1f,0f,0f);
            Matrix4f view  = new Matrix4f().translate(0f,0f,-3f);
            Matrix4f projection  = new Matrix4f().perspective(45f* 2f * 3.14f/360f,SCR_WIDTH/SCR_HEIGHT,0.1f,100f);
            model.get(modelBuffer);
            view.get(viewBuffer);
            projection.get(ProjectionBuffer);

            shaderProgram.use();
            shaderProgram.setFloat("mixValue", mixValue);

            glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "model"), false, modelBuffer);
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "view"), false, viewBuffer);
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "projection"), false, ProjectionBuffer);
```



## 更加 3D

我们扩展了2D平面到一个3D立方体，用36个顶点绘制，每个面有两个三角形组成。为了增加趣味性，我们让立方体随时间旋转，并使用`glDrawArrays`绘制这个立方体。

然而，可能观察到一些奇怪的效果，如某些面看起来未被正确遮挡。这是因为OpenGL以三角形为单位绘制立方体，有些三角形可能被错误地渲染在其他三角形上方，导致未预期的结果。

幸运的是，OpenGL使用Z缓冲存储深度信息，这允许进行深度测试。通过配置OpenGL进行深度测试，可以让它在绘制时根据深度信息决定何时覆盖像素。



### Z缓冲

OpenGL利用Z缓冲（也称为深度缓冲）存储深度信息。这个缓冲类似颜色缓冲，GLFW会自动生成它。深度值存储在每个片段中作为片段的z值。在渲染过程中，OpenGL会比较当前片段的深度值与Z缓冲中的值，如果当前片段在Z轴上在其他片段之后，它将会被绘制，否则将被丢弃。这个过程被称为深度测试，OpenGL自动执行它。

为了确保深度测试被执行，需要显式告诉OpenGL启用深度测试。默认情况下是关闭的，可以使用`glEnable`函数启用深度测试：

```
cCopy code
glEnable(GL_DEPTH_TEST);
```

同时，我们也希望在每次渲染迭代前清除深度缓冲，以避免前一帧的深度信息干扰当前渲染。类似清除颜色缓冲，使用`glClear`函数并指定`GL_DEPTH_BUFFER_BIT`位来清除深度缓冲：

```
cCopy code
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
```

这样的设置下，可以看到一个开启深度测试的效果，各个面都被正确绘制纹理，并且立方体在旋转！如果你的程序有问题，可以下载源代码进行比对。

```java
package org.example;

import org.joml.Matrix4f;
import org.lwjgl.BufferUtils;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.opengl.GL;

import java.io.IOException;
import java.nio.FloatBuffer;

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
                -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
                0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
                0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
                0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
                -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
                -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

                -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
                0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
                0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
                0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
                -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
                -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

                -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
                -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
                -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
                -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
                -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
                -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

                0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
                0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
                0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
                0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
                0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
                0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

                -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
                0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
                0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
                0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
                -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
                -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

                -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
                0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
                0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
                0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
                -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
                -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
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


        glVertexAttribPointer(0, 3, GL_FLOAT, false, 5 * Float.BYTES, 0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, false, 5 * Float.BYTES, 3*Float.BYTES);
        glEnableVertexAttribArray(1);

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

        //开日Z缓冲
        glEnable(GL_DEPTH_TEST);

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
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


            // draw our first triangle
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, texture2);


            FloatBuffer modelBuffer = BufferUtils.createFloatBuffer(16); // Assuming a 4x4 matrix
            FloatBuffer viewBuffer = BufferUtils.createFloatBuffer(16); // Assuming a 4x4 matrix
            FloatBuffer ProjectionBuffer = BufferUtils.createFloatBuffer(16); // Assuming a 4x4 matrix
            Matrix4f model = new Matrix4f().rotate((float) GLFW.glfwGetTime() * 50f * 2f * 3.14f/360f ,0.5f,1f,0f);
            Matrix4f view  = new Matrix4f().translate(0f,0f,-3f);
            Matrix4f projection  = new Matrix4f().perspective(45f* 2f * 3.14f/360f,SCR_WIDTH/SCR_HEIGHT,0.1f,100f);
            model.get(modelBuffer);
            view.get(viewBuffer);
            projection.get(ProjectionBuffer);

            shaderProgram.use();
            shaderProgram.setFloat("mixValue", mixValue);

            glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "model"), false, modelBuffer);
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "view"), false, viewBuffer);
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "projection"), false, ProjectionBuffer);

            glBindVertexArray(VAO);
            glDrawArrays(GL_TRIANGLES, 0, 36);

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

顶点着色器

```glsl
#version 330 core
layout (location = 0) in vec3 aPos;   // 位置变量的属性位置值为 0
//layout (location = 1) in vec3 aColor; // 颜色变量的属性位置值为 1
layout (location = 1) in vec2 aTexCoord;

//out vec3 ourColor; // 向片段着色器输出一个颜色
out vec2 TexCoord;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
//    ourColor = aColor; // 将ourColor设置为我们从顶点数据那里得到的输入颜色
    TexCoord = aTexCoord;
}
```

片段着色器

```glsl
#version 330 core
out vec4 FragColor;
//in vec3 ourColor;
in vec2 TexCoord;

uniform sampler2D texture1;
uniform sampler2D texture2;

uniform float mixValue;

void main()
{
    FragColor = mix(texture(texture1, TexCoord), texture(texture2, vec2(1.0-TexCoord.x,TexCoord.y)), mixValue);
}
```



### 更多的立方体！

我们想在屏幕上显示10个立方体，它们的外观相同，区别在于位置和旋转角度。我们已经定义了立方体的图形布局，所以渲染更多物体时无需更改缓冲数组和属性数组，只需改变每个对象的模型矩阵即可将立方体变换到世界坐标系中。

首先，为每个立方体定义一个位移向量，指定它在世界空间的位置。我们将使用一个`glm::vec3`数组定义10个立方体的位置：

```
cppCopy codeglm::vec3 cubePositions[] = {
    glm::vec3( 0.0f,  0.0f,  0.0f), 
    glm::vec3( 2.0f,  5.0f, -15.0f), 
    // 更多位置...
};
```

在游戏循环中，我们调用`glDrawArrays` 10次，每次在渲染之前传入一个不同的模型矩阵到顶点着色器中。使用一个小循环，在每次渲染时更新模型矩阵并渲染10个物体。对每个立方体增加了一些旋转：

```
cppCopy codeglBindVertexArray(VAO);
for(unsigned int i = 0; i < 10; i++)
{
    glm::mat4 model;
    model = glm::translate(model, cubePositions[i]);
    float angle = 20.0f * i; 
    model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
    ourShader.setMat4("model", model);

    glDrawArrays(GL_TRIANGLES, 0	, 36);
}
```

以上代码将会在每次新的立方体绘制时更新模型矩阵，重复10次。这样就能看到10个立方体，每个都在以奇特的角度旋转。

这看起来不错！立方体们找到了他们的伙伴。如果你在实现这部分时遇到问题，可以参考源代码进行对比。



## 练习

练习3：

```jjava
                float angle =  i%3==0?20.0f * (i+1)  * (float)GLFW.glfwGetTime():20.0f * (i+1);

```



 

