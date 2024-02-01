---
title: opengl08摄像机
date: 2023-12-11 22:04:28
tags:
- java
- opengl
cover: https://view.moezx.cc/images/2017/12/16/QP.jpg
---

# 摄像机

观察矩阵在OpenGL中移动场景，模拟出摄像机效果

讨论如何配置OpenGL摄像机，实现FPS风格的自由移动，并创建一个自定义摄像机类。

## 摄像机/观察空间

- 观察矩阵将世界坐标转换为相对于摄像机位置与方向的观察坐标。
- 定义摄像机需要位置、观察方向、右侧向量和上方向向量。
- 获取摄像机位置、方向向量、右向量和上向量的方法。
- 使用这些向量创建LookAt矩阵，将世界坐标转换到观察空间。
- 示例中展示了如何使用GLM库中的函数来创建观察矩阵。
- 最后通过一段代码展示了如何让摄像机围绕场景旋转，并使用LookAt矩阵实现视角的变化。、

```java
package org.example;

import org.joml.Matrix4f;
import org.joml.Vector3f;
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
        Vector3f[] cubePositions = {
                new Vector3f( 0.0f,  0.0f,  0.0f),
                new Vector3f( 2.0f,  5.0f, -15.0f),
                new Vector3f(-1.5f, -2.2f, -2.5f),
                new Vector3f(-3.8f, -2.0f, -12.3f),
                new Vector3f( 2.4f, -0.4f, -3.5f),
                new Vector3f(-1.7f,  3.0f, -7.5f),
                new Vector3f( 1.3f, -2.0f, -2.5f),
                new Vector3f( 1.5f,  2.0f, -2.5f),
                new Vector3f( 1.5f,  0.2f, -1.5f),
                new Vector3f(-1.3f,  1.0f, -1.5f)
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


        //开日Z缓冲
        glEnable(GL_DEPTH_TEST);

        while(!GLFW.glfwWindowShouldClose(window)){
            // input
            // -----
            processInput(window);


            // render
            // -----
            glClearColor(0.2f,0.2f,0.2f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


            // draw our first triangle
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, texture2);

            float radius = 10.0f;
            float camX = (float) (Math.sin(GLFW.glfwGetTime()) * radius);
            float camZ = (float) Math.cos(GLFW.glfwGetTime()) * radius;
            Matrix4f view  = new Matrix4f().lookAt(new Vector3f(camX,0,camZ),new Vector3f(0,0,0),new Vector3f(0,1,0));
            Matrix4f projection  = new Matrix4f().perspective(45f* 2f * 3.14f/360f,SCR_WIDTH/SCR_HEIGHT,0.1f,100f);
            glBindVertexArray(VAO);
            for(int i = 0; i < 10; i++)
            {
                float angle = 20.0f * i;
                Matrix4f model = new Matrix4f().translate(cubePositions[i]).rotate(angle * 2f * 3.14f/360f ,0.5f,1f,0f);
                shaderProgram.setMat4("model", model);
                shaderProgram.setMat4("view", view);
                shaderProgram.setMat4("projection", projection);

                shaderProgram.use();
                glDrawArrays(GL_TRIANGLES, 0	, 36);

            }



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

    public static void processInput(long window){
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_ESCAPE) == GLFW.GLFW_PRESS) {
            GLFW.glfwSetWindowShouldClose(window, true);
        }
    }
}
```



# 自由移动

- 定义摄像机变量包括位置（cameraPos）、前方向（cameraFront）和上方向（cameraUp）。
- 使用`glm::lookAt`函数设置视角，将摄像机定位在`cameraPos`，并使其朝向目标位置(`cameraPos + cameraFront`)，并设定上方向为`cameraUp`。
- 通过按键命令更新`cameraPos`向量，实现摄像机在按下W、A、S、D键时的移动操作。
- 按下W键会使摄像机沿着正前方向移动，S键则相反，A键和D键则实现左右移动。
- 根据按键的不同，对`cameraPos`进行加减操作，使用了方向向量和叉乘操作来创建摄像机移动的效果。
- 在移动过程中，对右向量进行了标准化处理，确保移动速度匀速，而不因摄像机朝向不同而改变速度。

```java
package org.example;

import org.joml.Matrix4f;
import org.joml.Vector3f;
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

    private static Vector3f cameraPos   = new Vector3f(0.0f, 0.0f,  3.0f);
    private static Vector3f cameraFront = new Vector3f(0.0f, 0.0f, -3.0f);
    private static Vector3f cameraUp    = new Vector3f(0.0f, 1.0f,  0.0f);



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
        Vector3f[] cubePositions = {
                new Vector3f( 0.0f,  0.0f,  0.0f),
                new Vector3f( 2.0f,  5.0f, -15.0f),
                new Vector3f(-1.5f, -2.2f, -2.5f),
                new Vector3f(-3.8f, -2.0f, -12.3f),
                new Vector3f( 2.4f, -0.4f, -3.5f),
                new Vector3f(-1.7f,  3.0f, -7.5f),
                new Vector3f( 1.3f, -2.0f, -2.5f),
                new Vector3f( 1.5f,  2.0f, -2.5f),
                new Vector3f( 1.5f,  0.2f, -1.5f),
                new Vector3f(-1.3f,  1.0f, -1.5f)
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


        //开日Z缓冲
        glEnable(GL_DEPTH_TEST);

        while(!GLFW.glfwWindowShouldClose(window)){
            // input
            // -----
            processInput(window);


            // render
            // -----
            glClearColor(0.2f,0.2f,0.2f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


            // draw our first triangle
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, texture2);


            Matrix4f view  = new Matrix4f().lookAt(cameraPos,new Vector3f(cameraPos).add(cameraFront),cameraUp);
            Matrix4f projection  = new Matrix4f().perspective(45f* 2f * 3.14f/360f,SCR_WIDTH/SCR_HEIGHT,0.1f,100f);
            glBindVertexArray(VAO);
            for(int i = 0; i < 10; i++)
            {
                float angle = 20.0f * i;
                Matrix4f model = new Matrix4f().translate(cubePositions[i]).rotate(angle * 2f * 3.14f/360f ,0.5f,1f,0f);
                shaderProgram.setMat4("model", model);
                shaderProgram.setMat4("view", view);
                shaderProgram.setMat4("projection", projection);

                shaderProgram.use();
                glDrawArrays(GL_TRIANGLES, 0	, 36);

            }



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

    public static void processInput(long window){
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_ESCAPE) == GLFW.GLFW_PRESS) {
            GLFW.glfwSetWindowShouldClose(window, true);
        }
        float cameraSpeed = 0.05f; // adjust accordingly
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_W) == GLFW.GLFW_PRESS)
            cameraPos.add(new Vector3f(cameraFront).mul(cameraSpeed));
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_S) == GLFW.GLFW_PRESS)
            cameraPos.sub(new Vector3f(cameraFront).mul(cameraSpeed));
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_A) == GLFW.GLFW_PRESS)
            cameraPos.sub(new Vector3f(cameraFront).cross(cameraUp).normalize().mul(cameraSpeed));
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_D) == GLFW.GLFW_PRESS)
            cameraPos.add(new Vector3f(cameraFront).cross(cameraUp).normalize().mul(cameraSpeed));
    }
}
```



## 移动速度

- 解释了由于处理器性能不同可能导致程序在不同硬件上移动速度不同的问题。
- 引入了时间差（deltaTime）的概念，用以追踪当前帧与上一帧之间的时间差值。
- 通过计算每一帧的时间差，将其考虑进速度计算中（例如，`cameraSpeed = 2.5f * deltaTime`）。
- 指出了使用时间差来调整速度能够确保在不同硬件上保持相同的移动速度，从而保证用户体验的一致性。
- 最终，指出这一技术使得摄像机系统更加流畅，无论在任何系统上都能获得相同的移动速度。

```java
package org.example;

import org.joml.Matrix4f;
import org.joml.Vector3f;
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

    private static Vector3f cameraPos   = new Vector3f(0.0f, 0.0f,  3.0f);
    private static Vector3f cameraFront = new Vector3f(0.0f, 0.0f, -3.0f);
    private static Vector3f cameraUp    = new Vector3f(0.0f, 1.0f,  0.0f);

    private static float deltaTime = 0.0f;
    private static float lastFrame = 0.0f;



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
        Vector3f[] cubePositions = {
                new Vector3f( 0.0f,  0.0f,  0.0f),
                new Vector3f( 2.0f,  5.0f, -15.0f),
                new Vector3f(-1.5f, -2.2f, -2.5f),
                new Vector3f(-3.8f, -2.0f, -12.3f),
                new Vector3f( 2.4f, -0.4f, -3.5f),
                new Vector3f(-1.7f,  3.0f, -7.5f),
                new Vector3f( 1.3f, -2.0f, -2.5f),
                new Vector3f( 1.5f,  2.0f, -2.5f),
                new Vector3f( 1.5f,  0.2f, -1.5f),
                new Vector3f(-1.3f,  1.0f, -1.5f)
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


        //开日Z缓冲
        glEnable(GL_DEPTH_TEST);

        while(!GLFW.glfwWindowShouldClose(window)){
            // input
            // -----
            processInput(window);

            float currentFrame = (float) GLFW.glfwGetTime();
            deltaTime = currentFrame - lastFrame;
            lastFrame = currentFrame;


            // render
            // -----
            glClearColor(0.2f,0.2f,0.2f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


            // draw our first triangle
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, texture2);


            Matrix4f view  = new Matrix4f().lookAt(cameraPos,new Vector3f(cameraPos).add(cameraFront),cameraUp);
            Matrix4f projection  = new Matrix4f().perspective(45f* 2f * 3.14f/360f,SCR_WIDTH/SCR_HEIGHT,0.1f,100f);
            glBindVertexArray(VAO);
            for(int i = 0; i < 10; i++)
            {
                float angle = 20.0f * i;
                Matrix4f model = new Matrix4f().translate(cubePositions[i]).rotate(angle * 2f * 3.14f/360f ,0.5f,1f,0f);
                shaderProgram.setMat4("model", model);
                shaderProgram.setMat4("view", view);
                shaderProgram.setMat4("projection", projection);

                shaderProgram.use();
                glDrawArrays(GL_TRIANGLES, 0	, 36);

            }



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

    public static void processInput(long window){
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_ESCAPE) == GLFW.GLFW_PRESS) {
            GLFW.glfwSetWindowShouldClose(window, true);
        }
        float cameraSpeed = 2.5f * deltaTime;// adjust accordingly
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_W) == GLFW.GLFW_PRESS)
            cameraPos.add(new Vector3f(cameraFront).mul(cameraSpeed));
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_S) == GLFW.GLFW_PRESS)
            cameraPos.sub(new Vector3f(cameraFront).mul(cameraSpeed));
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_A) == GLFW.GLFW_PRESS)
            cameraPos.sub(new Vector3f(cameraFront).cross(cameraUp).normalize().mul(cameraSpeed));
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_D) == GLFW.GLFW_PRESS)
            cameraPos.add(new Vector3f(cameraFront).cross(cameraUp).normalize().mul(cameraSpeed));
    }
}
```



# 视角移动

## 欧拉角

- 俯仰角描述垂直上下方向的角度变化，对应于观察者往上或往下看的程度，其方向向量的y分量为sin(俯仰角)。
- 俯仰角也会影响方向向量的x和z分量，它们分别为cos(俯仰角)。
- 偏航角描述水平左右方向的角度变化，对应观察者往左或往右看的程度，其方向向量的x分量为cos(偏航角)，z分量为sin(偏航角)。
- 最终，结合俯仰角和偏航角，得到了描述摄像机自由旋转视角的三维方向向量。

```java
package org.example;

import org.joml.Matrix4f;
import org.joml.Vector3f;
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

    private static Vector3f cameraPos   = new Vector3f(0.0f, 0.0f,  3.0f);
    private static Vector3f cameraFront = new Vector3f(0.0f, 0.0f, -3.0f);
    private static Vector3f cameraUp    = new Vector3f(0.0f, 1.0f,  0.0f);

    private static float deltaTime = 0.0f;
    private static float lastFrame = 0.0f;

    private static float lastX = 400;
    private static float lastY = 300;
    private static boolean firstMouse = true;

    private static float yaw = 0;
    private static float pitch  = 0;




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
        GLFW.glfwSetInputMode(window, GLFW.GLFW_CURSOR, GLFW.GLFW_CURSOR_DISABLED);
        GLFW.glfwSetFramebufferSizeCallback(window,(window1, width, height) -> glViewport(0,0,width,height));
        GLFW.glfwSetCursorPosCallback(window, Main::mouseCallback);
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
        Vector3f[] cubePositions = {
                new Vector3f( 0.0f,  0.0f,  0.0f),
                new Vector3f( 2.0f,  5.0f, -15.0f),
                new Vector3f(-1.5f, -2.2f, -2.5f),
                new Vector3f(-3.8f, -2.0f, -12.3f),
                new Vector3f( 2.4f, -0.4f, -3.5f),
                new Vector3f(-1.7f,  3.0f, -7.5f),
                new Vector3f( 1.3f, -2.0f, -2.5f),
                new Vector3f( 1.5f,  2.0f, -2.5f),
                new Vector3f( 1.5f,  0.2f, -1.5f),
                new Vector3f(-1.3f,  1.0f, -1.5f)
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


        //开日Z缓冲
        glEnable(GL_DEPTH_TEST);

        while(!GLFW.glfwWindowShouldClose(window)){
            // input
            // -----
            processInput(window);

            float currentFrame = (float) GLFW.glfwGetTime();
            deltaTime = currentFrame - lastFrame;
            lastFrame = currentFrame;


            // render
            // -----
            glClearColor(0.2f,0.2f,0.2f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


            // draw our first triangle
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, texture2);


            Matrix4f view  = new Matrix4f().lookAt(cameraPos,new Vector3f(cameraPos).add(cameraFront),cameraUp);
            Matrix4f projection  = new Matrix4f().perspective(45f* 2f * 3.14f/360f,SCR_WIDTH/SCR_HEIGHT,0.1f,100f);
            glBindVertexArray(VAO);
            for(int i = 0; i < 10; i++)
            {
                float angle = 20.0f * i;
                Matrix4f model = new Matrix4f().translate(cubePositions[i]).rotate(angle * 2f * 3.14f/360f ,0.5f,1f,0f);
                shaderProgram.setMat4("model", model);
                shaderProgram.setMat4("view", view);
                shaderProgram.setMat4("projection", projection);

                shaderProgram.use();
                glDrawArrays(GL_TRIANGLES, 0	, 36);

            }



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

    public static void processInput(long window){
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_ESCAPE) == GLFW.GLFW_PRESS) {
            GLFW.glfwSetWindowShouldClose(window, true);
        }
        float cameraSpeed = 2.5f * deltaTime;// adjust accordingly
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_W) == GLFW.GLFW_PRESS)
            cameraPos.add(new Vector3f(cameraFront).mul(cameraSpeed));
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_S) == GLFW.GLFW_PRESS)
            cameraPos.sub(new Vector3f(cameraFront).mul(cameraSpeed));
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_A) == GLFW.GLFW_PRESS)
            cameraPos.sub(new Vector3f(cameraFront).cross(cameraUp).normalize().mul(cameraSpeed));
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_D) == GLFW.GLFW_PRESS)
            cameraPos.add(new Vector3f(cameraFront).cross(cameraUp).normalize().mul(cameraSpeed));
    }


    public static void mouseCallback(long window,double xpos,double ypos){
        if(firstMouse) // 这个bool变量初始时是设定为true的
        {
            lastX = (float) xpos;
            lastY = (float) ypos;
            firstMouse = false;
        }

        float xoffset = (float) (xpos - lastX);
        float yoffset = (float) (lastY - ypos); // 注意这里是相反的，因为y坐标是从底部往顶部依次增大的
        lastX = (float) xpos;
        lastY = (float) ypos;

        float sensitivity = 0.05f;
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw   += xoffset;
        pitch += yoffset;

        if(pitch > 89.0f)
            pitch =  89.0f;
        if(pitch < -89.0f)
            pitch = -89.0f;

        Vector3f front = new Vector3f();
        front.x = (float) (Math.cos(Math.toRadians(yaw)) * Math.cos(Math.toRadians(pitch)));
        front.y = (float) Math.sin(Math.toRadians(pitch));
        front.z = (float) (Math.sin(Math.toRadians(yaw)) * Math.cos(Math.toRadians(pitch)));
        cameraFront =front.normalize();

    }
}
```



## 鼠标输入

1. 设置GLFW捕捉光标，隐藏光标并使其停留在窗口内，适用于FPS摄像机系统。
2. 注册鼠标移动事件的回调函数，并在每一帧中计算鼠标位置之间的偏移量。
3. 将偏移量乘以灵敏度，并将其分别加到全局的俯仰角（pitch）和偏航角（yaw）中。
4. 设置俯仰角的限制，使其不会超过指定范围，而偏航角则无限制。
5. 根据俯仰角和偏航角的变化计算新的方向向量，即摄像机的前方向量（cameraFront）。
6. 处理第一次获取鼠标输入时产生的问题，确保在第一次获取输入时不会产生摄像机突然跳动的情况，通过更新初始鼠标位置来解决这个问题。



## 缩放

1. 创建鼠标滚轮的回调函数 `scroll_callback`，用于处理竖直滚动对视野变化的影响。通过检测滚动量 `yoffset` 的变化来调整视野大小，并将其限制在特定范围内（1.0f 到 45.0f）。
2. 在每一帧中，根据更新后的 `fov` 变量，使用 `glm::perspective` 函数重新设置透视投影矩阵，将 `fov` 作为视野参数。
3. 注册鼠标滚轮的回调函数，使用 `glfwSetScrollCallback` 函数将 `scroll_callback` 与窗口的滚轮事件关联起来。



```java
package org.example;

import org.joml.Matrix4f;
import org.joml.Vector3f;
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

    private static Vector3f cameraPos   = new Vector3f(0.0f, 0.0f,  3.0f);
    private static Vector3f cameraFront = new Vector3f(0.0f, 0.0f, -3.0f);
    private static Vector3f cameraUp    = new Vector3f(0.0f, 1.0f,  0.0f);

    private static float deltaTime = 0.0f;
    private static float lastFrame = 0.0f;

    private static float lastX = 400;
    private static float lastY = 300;
    private static boolean firstMouse = true;

    private static float yaw = 0;
    private static float pitch  = 0;

    private static float fov = 45;



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
        GLFW.glfwSetInputMode(window, GLFW.GLFW_CURSOR, GLFW.GLFW_CURSOR_DISABLED);
        GLFW.glfwSetFramebufferSizeCallback(window,(window1, width, height) -> glViewport(0,0,width,height));
        GLFW.glfwSetCursorPosCallback(window, Main::mouseCallback);
        GLFW.glfwSetScrollCallback(window, Main::scrollCallback);
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
        Vector3f[] cubePositions = {
                new Vector3f( 0.0f,  0.0f,  0.0f),
                new Vector3f( 2.0f,  5.0f, -15.0f),
                new Vector3f(-1.5f, -2.2f, -2.5f),
                new Vector3f(-3.8f, -2.0f, -12.3f),
                new Vector3f( 2.4f, -0.4f, -3.5f),
                new Vector3f(-1.7f,  3.0f, -7.5f),
                new Vector3f( 1.3f, -2.0f, -2.5f),
                new Vector3f( 1.5f,  2.0f, -2.5f),
                new Vector3f( 1.5f,  0.2f, -1.5f),
                new Vector3f(-1.3f,  1.0f, -1.5f)
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


        //开日Z缓冲
        glEnable(GL_DEPTH_TEST);

        while(!GLFW.glfwWindowShouldClose(window)){
            // input
            // -----
            processInput(window);

            float currentFrame = (float) GLFW.glfwGetTime();
            deltaTime = currentFrame - lastFrame;
            lastFrame = currentFrame;


            // render
            // -----
            glClearColor(0.2f,0.2f,0.2f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


            // draw our first triangle
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, texture2);


            Matrix4f view  = new Matrix4f().lookAt(cameraPos,new Vector3f(cameraPos).add(cameraFront),cameraUp);
            Matrix4f projection  = new Matrix4f().perspective((float) Math.toRadians(fov),SCR_WIDTH/SCR_HEIGHT,0.1f,100f);
            glBindVertexArray(VAO);
            for(int i = 0; i < 10; i++)
            {
                float angle = 20.0f * i;
                Matrix4f model = new Matrix4f().translate(cubePositions[i]).rotate(angle * 2f * 3.14f/360f ,0.5f,1f,0f);
                shaderProgram.setMat4("model", model);
                shaderProgram.setMat4("view", view);
                shaderProgram.setMat4("projection", projection);

                shaderProgram.use();
                glDrawArrays(GL_TRIANGLES, 0	, 36);

            }



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

    public static void processInput(long window){
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_ESCAPE) == GLFW.GLFW_PRESS) {
            GLFW.glfwSetWindowShouldClose(window, true);
        }
        float cameraSpeed = 2.5f * deltaTime;// adjust accordingly
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_W) == GLFW.GLFW_PRESS)
            cameraPos.add(new Vector3f(cameraFront).mul(cameraSpeed));
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_S) == GLFW.GLFW_PRESS)
            cameraPos.sub(new Vector3f(cameraFront).mul(cameraSpeed));
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_A) == GLFW.GLFW_PRESS)
            cameraPos.sub(new Vector3f(cameraFront).cross(cameraUp).normalize().mul(cameraSpeed));
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_D) == GLFW.GLFW_PRESS)
            cameraPos.add(new Vector3f(cameraFront).cross(cameraUp).normalize().mul(cameraSpeed));
    }


    public static void mouseCallback(long window,double xpos,double ypos){
        if(firstMouse) // 这个bool变量初始时是设定为true的
        {
            lastX = (float) xpos;
            lastY = (float) ypos;
            firstMouse = false;
        }

        float xoffset = (float) (xpos - lastX);
        float yoffset = (float) (lastY - ypos); // 注意这里是相反的，因为y坐标是从底部往顶部依次增大的
        lastX = (float) xpos;
        lastY = (float) ypos;

        float sensitivity = 0.05f;
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw   += xoffset;
        pitch += yoffset;

        if(pitch > 89.0f)
            pitch =  89.0f;
        if(pitch < -89.0f)
            pitch = -89.0f;

        Vector3f front = new Vector3f();
        front.x = (float) (Math.cos(Math.toRadians(yaw)) * Math.cos(Math.toRadians(pitch)));
        front.y = (float) Math.sin(Math.toRadians(pitch));
        front.z = (float) (Math.sin(Math.toRadians(yaw)) * Math.cos(Math.toRadians(pitch)));
        cameraFront =front.normalize();

    }

    public static void scrollCallback(long window, double xoffset, double yoffset)
    {
        if(fov >= 1.0f && fov <= 45.0f)
            fov -= yoffset;
        if(fov <= 1.0f)
            fov = 1.0f;
        if(fov >= 45.0f)
            fov = 45.0f;
    }

}
```



# 摄像机类

```java
package org.example;

import org.joml.Matrix4f;
import org.joml.Vector3f;

enum CameraMovement {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
}


public class Camera {
    private Vector3f position;
    private Vector3f front;
    private Vector3f up;
    private Vector3f right;
    private Vector3f worldUp;
    private float yaw;
    private float pitch;
    private float movementSpeed;
    private float mouseSensitivity;
    public float zoom;

    public Camera(Vector3f position, Vector3f up, float yaw, float pitch) {
        this.position = position;
        this.worldUp = up;
        this.yaw = yaw;
        this.pitch = pitch;
        this.front = new Vector3f(0.0f, 0.0f, -1.0f);
        this.movementSpeed = 2.5f;
        this.mouseSensitivity = 0.1f;
        this.zoom = 45.0f;
        updateCameraVectors();
    }

    public Matrix4f getViewMatrix() {
        Vector3f center = new Vector3f();
        position.add(front, center);
        return new Matrix4f().lookAt(position, center, up);
    }

    public void processKeyboard(CameraMovement direction, float deltaTime) {
        float velocity = movementSpeed * deltaTime;
        if (direction == CameraMovement.FORWARD)
            position.add(new Vector3f(front).mul(velocity));
        if (direction == CameraMovement.BACKWARD)
            position.sub(new Vector3f(front).mul(velocity));
        if (direction == CameraMovement.LEFT)
            position.sub(new Vector3f(right).mul(velocity));
        if (direction == CameraMovement.RIGHT)
            position.add(new Vector3f(right).mul(velocity));
    }

    public void processMouseMovement(float xoffset, float yoffset, boolean constrainPitch) {
        xoffset *= mouseSensitivity;
        yoffset *= mouseSensitivity;

        yaw += xoffset;
        pitch += yoffset;

        if (constrainPitch) {
            if (pitch > 89.0f)
                pitch = 89.0f;
            if (pitch < -89.0f)
                pitch = -89.0f;
        }

        updateCameraVectors();
    }

    public void processMouseScroll(float yoffset) {
        zoom -= yoffset;
        if (zoom < 1.0f)
            zoom = 1.0f;
        if (zoom > 45.0f)
            zoom = 45.0f;
    }



    private void updateCameraVectors() {
        Vector3f front = new Vector3f();
        front.x = (float) Math.cos(Math.toRadians(yaw)) * (float) Math.cos(Math.toRadians(pitch));
        front.y = (float) Math.sin(Math.toRadians(pitch));
        front.z = (float) Math.sin(Math.toRadians(yaw)) * (float) Math.cos(Math.toRadians(pitch));
        this.front = front.normalize();

        this.right = this.front.cross(worldUp, new Vector3f()).normalize();
        this.up = this.right.cross(this.front, new Vector3f()).normalize();
    }

}

```





```java
package org.example;

import org.joml.Matrix4f;
import org.joml.Vector3f;
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

    private static Vector3f cameraPos   = new Vector3f(0.0f, 0.0f,  3.0f);
    private static Vector3f cameraFront = new Vector3f(0.0f, 0.0f, -3.0f);
    private static Vector3f cameraUp    = new Vector3f(0.0f, 1.0f,  0.0f);

    private static float deltaTime = 0.0f;
    private static float lastFrame = 0.0f;

    private static float lastX = 400;
    private static float lastY = 300;
    private static boolean firstMouse = true;

    private static float yaw = 0;
    private static float pitch  = 0;

    private static float fov = 45;

    private static Camera camera = new Camera(new Vector3f(0,0,3),new Vector3f(0,1,0),0,0);



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
        GLFW.glfwSetInputMode(window, GLFW.GLFW_CURSOR, GLFW.GLFW_CURSOR_DISABLED);
        GLFW.glfwSetFramebufferSizeCallback(window,(window1, width, height) -> glViewport(0,0,width,height));
        GLFW.glfwSetCursorPosCallback(window, Main::mouseCallback);
        GLFW.glfwSetScrollCallback(window, Main::scrollCallback);
//        creates the necessary function pointers for OpenGL's functions, making them accessible and usable within your Java code.
        GL.createCapabilities();
        //开日Z缓冲
        glEnable(GL_DEPTH_TEST);

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
        Vector3f[] cubePositions = {
                new Vector3f( 0.0f,  0.0f,  0.0f),
                new Vector3f( 2.0f,  5.0f, -15.0f),
                new Vector3f(-1.5f, -2.2f, -2.5f),
                new Vector3f(-3.8f, -2.0f, -12.3f),
                new Vector3f( 2.4f, -0.4f, -3.5f),
                new Vector3f(-1.7f,  3.0f, -7.5f),
                new Vector3f( 1.3f, -2.0f, -2.5f),
                new Vector3f( 1.5f,  2.0f, -2.5f),
                new Vector3f( 1.5f,  0.2f, -1.5f),
                new Vector3f(-1.3f,  1.0f, -1.5f)
        };
        int VBO = glGenBuffers();
        int VAO =  glGenVertexArrays();

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



        while(!GLFW.glfwWindowShouldClose(window)){
            // input
            // -----
            processInput(window);

            float currentFrame = (float) GLFW.glfwGetTime();
            deltaTime = currentFrame - lastFrame;
            lastFrame = currentFrame;

            // render
            // -----
            glClearColor(0.2f,0.2f,0.2f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, texture2);

            shaderProgram.use();

            Matrix4f projection  = new Matrix4f().perspective((float) Math.toRadians(camera.zoom),SCR_WIDTH/SCR_HEIGHT,0.1f,100f);
            Matrix4f view  =  camera.getViewMatrix();
            shaderProgram.setMat4("view", view);
            shaderProgram.setMat4("projection", projection);

            glBindVertexArray(VAO);
            for(int i = 0; i < 10; i++)
            {
                float angle = 20.0f * i;
                Matrix4f model = new Matrix4f().translate(cubePositions[i]).rotate(angle * 2f * 3.14f/360f ,0.5f,1f,0f);
                shaderProgram.setMat4("model", model);


                glDrawArrays(GL_TRIANGLES, 0	, 36);

            }



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

    public static void processInput(long window){
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_ESCAPE) == GLFW.GLFW_PRESS) {
            GLFW.glfwSetWindowShouldClose(window, true);
        }
        float cameraSpeed = 2.5f * deltaTime;// adjust accordingly
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_W) == GLFW.GLFW_PRESS)
            camera.processKeyboard(CameraMovement.FORWARD, deltaTime);
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_S) == GLFW.GLFW_PRESS)
            camera.processKeyboard(CameraMovement.BACKWARD, deltaTime);
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_A) == GLFW.GLFW_PRESS)
            camera.processKeyboard(CameraMovement.LEFT, deltaTime);
        if (GLFW.glfwGetKey(window, GLFW.GLFW_KEY_D) == GLFW.GLFW_PRESS)
            camera.processKeyboard(CameraMovement.RIGHT, deltaTime);
    }

    public static void mouseCallback(long window,double xpos,double ypos){
        if(firstMouse) // 这个bool变量初始时是设定为true的
        {
            lastX = (float) xpos;
            lastY = (float) ypos;
            firstMouse = false;
        }

        float xoffset = (float) (xpos - lastX);
        float yoffset = (float) (lastY - ypos); // 注意这里是相反的，因为y坐标是从底部往顶部依次增大的
        lastX = (float) xpos;
        lastY = (float) ypos;

        camera.processMouseMovement(xoffset,yoffset,true);

    }

    public static void scrollCallback(long window, double xoffset, double yoffset)
    {
        camera.processMouseScroll((float) yoffset);
    }

}
```

