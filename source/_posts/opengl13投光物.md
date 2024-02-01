---
title: opengl13投光物
date: 2023-12-16 23:11:13
tags:
- java
- opengl
cover: https://view.moezx.cc/images/2019/01/30/photo.png
---



# 平行光

```glsl
#version 330 core

struct Material{
    sampler2D diffuse;
    sampler2D specular;
    float shininess;
};

struct Light{
//    vec3 position;
    vec3 direction;

    vec3 amnbien;
    vec3 diffuse;
    vec3 specular;
};

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

out vec4 FragColor;
uniform vec3 viewPos;
uniform Material material;
uniform Light light;

void main()
{
    // 环境光
    vec3 ambient = light.amnbien  * vec3(texture(material.diffuse,TexCoords));

    // 漫反射
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(-light.direction);
    float diff = max(dot(norm,lightDir),0.0);
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse,TexCoords));

    //镜面
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir,norm);
    float spec = pow(max(dot(viewDir,reflectDir),0.0),material.shininess);
    vec3 specular = light.specular * spec * vec3(texture(material.specular,TexCoords));

    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result,1.0);

}
```





片段着色器

```glsl
#version 330 core

struct Material{
    sampler2D diffuse;
    sampler2D specular;
    float shininess;
};

struct Light{
//    vec3 position;
    vec3 direction;

    vec3 amnbien;
    vec3 diffuse;
    vec3 specular;
};

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

out vec4 FragColor;
uniform vec3 viewPos;
uniform Material material;
uniform Light light;

void main()
{
    // 环境光
    vec3 ambient = light.amnbien  * vec3(texture(material.diffuse,TexCoords));

    // 漫反射
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(-light.direction);
    float diff = max(dot(norm,lightDir),0.0);
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse,TexCoords));

    //镜面
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir,norm);
    float spec = pow(max(dot(viewDir,reflectDir),0.0),material.shininess);
    vec3 specular = light.specular * spec * vec3(texture(material.specular,TexCoords));

    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result,1.0);

}
```



顶点着色器

```glsl
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoords = aTexCoords;
}
```





# 点光源





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

    private static float deltaTime = 0.0f;
    private static float lastFrame = 0.0f;
    private static float lastX = SCR_WIDTH/2;
    private static float lastY = SCR_HEIGHT/2;
    private static boolean firstMouse = true;
    private static Camera camera = new Camera(new Vector3f(0,0,3),new Vector3f(0,1,0),0,0);

    private static Vector3f lightPos = new Vector3f(1.2f,1.0f,2.0f);

    private static Vector3f[] cubePosition = {
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
        MyShader lightShader = new MyShader("lightcube.vert","lightcube.frag");

        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------

        float[] vertices = {
                -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,
                0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 0.0f,
                0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
                0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
                -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f,
                -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,

                -0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   0.0f, 0.0f,
                0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   1.0f, 0.0f,
                0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   1.0f, 1.0f,
                0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   1.0f, 1.0f,
                -0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   0.0f, 1.0f,
                -0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   0.0f, 0.0f,

                -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
                -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
                -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
                -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
                -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
                -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,

                0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
                0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
                0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
                0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
                0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
                0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,

                -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,
                0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 1.0f,
                0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
                0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
                -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 0.0f,
                -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,

                -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f,
                0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 1.0f,
                0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
                0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
                -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f,
                -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f
        };
        int VBO = glGenBuffers();
        int VAO =  glGenVertexArrays();

        // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER,VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, false, 8 * Float.BYTES, 0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, false, 8 * Float.BYTES, 3 * Float.BYTES);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 2, GL_FLOAT, false, 8 * Float.BYTES, 6 * Float.BYTES);
        glEnableVertexAttribArray(2);


        int lightCubeVAO  = glGenVertexArrays();
        glBindVertexArray(lightCubeVAO);
        // we only need to bind to the VBO (to link it with glVertexAttribPointer), no need to fill it; the VBO's data already contains all we need (it's already bound, but we do it again for educational purposes)
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        glVertexAttribPointer(0, 3, GL_FLOAT, false, 8 *Float.BYTES, 0);
        glEnableVertexAttribArray(0);

        int diffuseMap = loadTexture("src/main/resources/container2.png");
        int specularMap = loadTexture("src/main/resources/container2_specular.png");


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

            float angle = (float) (20 * GLFW.glfwGetTime());
            float radius = 2;
            Vector3f lightPos = new Vector3f(
                    (float) (Math.sin(Math.toRadians(angle)) * radius),
                    1.0f,
                    (float) (Math.cos(Math.toRadians(angle)) * radius)
            );


            shaderProgram.use();
            shaderProgram.setInt("material.diffuse",   0);
            shaderProgram.setInt("material.specular", 1);
            shaderProgram.setFloat("material.shininess", 32.0f);
            shaderProgram.setVec3("light.ambient",  0.2f, 0.2f, 0.2f);
            shaderProgram.setVec3("light.diffuse",  0.5f, 0.5f, 0.5f); // 将光照调暗了一些以搭配场景
            shaderProgram.setVec3("light.specular", 1.0f, 1.0f, 1.0f);
            shaderProgram.setVec3("light.position", lightPos);
            shaderProgram.setFloat("light.constant", 1f);
            shaderProgram.setFloat("light.linear", 0.09f);
            shaderProgram.setFloat("light.quadratic", 0.032f);


            shaderProgram.setVec3("lightColor",  1f,1f,1f);
            shaderProgram.setVec3("viewPos", camera.position);


            Matrix4f projection  = new Matrix4f().perspective((float) Math.toRadians(camera.zoom),SCR_WIDTH/SCR_HEIGHT,0.1f,100f);
            Matrix4f view  =  camera.getViewMatrix();
            shaderProgram.setMat4("projection", projection);
            shaderProgram.setMat4("view",view);
//            shaderProgram.setMat4("model",new Matrix4f().translate(0f,1f,0f));

            // bind diffuse map
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, diffuseMap);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, specularMap);

            glBindVertexArray(VAO);
            for(int i =0;i<10;i++){
                Matrix4f model = new Matrix4f();
                model.translate(cubePosition[i]);
                float angle2 = 20 * i;
                model.rotate((float) Math.toRadians(angle2),new Vector3f(1.0f,0.3f,0.5f));
                shaderProgram.setMat4("model",model);
                glDrawArrays(GL_TRIANGLES, 0	, 36);
            }


            lightShader.use();
            lightShader.setMat4("projection",projection);
            lightShader.setMat4("view",view);
            lightShader.setMat4("model",new Matrix4f().translate(lightPos).scale(0.2f));

            glBindVertexArray(lightCubeVAO);
            glDrawArrays(GL_TRIANGLES, 0	, 36);

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


    public static int loadTexture(String path) throws IOException {
        int textureID;
        textureID = glGenTextures();

        ImageReader.ImageData imageData = ImageReader.ReadImage(path);
        if(imageData != null){
            glBindTexture(GL_TEXTURE_2D,textureID);
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,imageData.width,imageData.height,0,GL_RGBA,GL_UNSIGNED_BYTE,imageData.data);
            glGenerateMipmap(GL_TEXTURE_2D);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }else{
            System.out.println("fail load path"+path);
        }
        return textureID;
    }

}
```





片段着色器

```glsl
#version 330 core

struct Material{
    sampler2D diffuse;
    sampler2D specular;
    float shininess;
};

struct Light{
    vec3 position;
//    vec3 direction;

    vec3 amnbien;
    vec3 diffuse;
    vec3 specular;

    float constant;
    float linear;
    float quadratic;
};

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

out vec4 FragColor;
uniform vec3 viewPos;
uniform Material material;
uniform Light light;

void main()
{

    float distance = length(light.position - FragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    // 环境光
    vec3 ambient = light.amnbien  * vec3(texture(material.diffuse,TexCoords));

    // 漫反射
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(light.position - FragPos);
    float diff = max(dot(norm,lightDir),0.0);
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse,TexCoords));

    //镜面
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir,norm);
    float spec = pow(max(dot(viewDir,reflectDir),0.0),material.shininess);
    vec3 specular = light.specular * spec * vec3(texture(material.specular,TexCoords));

    ambient *= attenuation;
    diff *= attenuation;
    specular *= attenuation;

    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result,1.0);

}
```





# 聚光



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

    private static float deltaTime = 0.0f;
    private static float lastFrame = 0.0f;
    private static float lastX = SCR_WIDTH/2;
    private static float lastY = SCR_HEIGHT/2;
    private static boolean firstMouse = true;
    private static Camera camera = new Camera(new Vector3f(0,0,3),new Vector3f(0,1,0),0,0);

    private static Vector3f lightPos = new Vector3f(1.2f,1.0f,2.0f);

    private static Vector3f[] cubePosition = {
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
        MyShader lightShader = new MyShader("lightcube.vert","lightcube.frag");

        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------

        float[] vertices = {
                -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,
                0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 0.0f,
                0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
                0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
                -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f,
                -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,

                -0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   0.0f, 0.0f,
                0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   1.0f, 0.0f,
                0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   1.0f, 1.0f,
                0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   1.0f, 1.0f,
                -0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   0.0f, 1.0f,
                -0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   0.0f, 0.0f,

                -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
                -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
                -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
                -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
                -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
                -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,

                0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
                0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
                0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
                0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
                0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
                0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,

                -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,
                0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 1.0f,
                0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
                0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
                -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 0.0f,
                -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,

                -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f,
                0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 1.0f,
                0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
                0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
                -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f,
                -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f
        };
        int VBO = glGenBuffers();
        int VAO =  glGenVertexArrays();

        // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER,VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, false, 8 * Float.BYTES, 0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, false, 8 * Float.BYTES, 3 * Float.BYTES);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 2, GL_FLOAT, false, 8 * Float.BYTES, 6 * Float.BYTES);
        glEnableVertexAttribArray(2);


        int lightCubeVAO  = glGenVertexArrays();
        glBindVertexArray(lightCubeVAO);
        // we only need to bind to the VBO (to link it with glVertexAttribPointer), no need to fill it; the VBO's data already contains all we need (it's already bound, but we do it again for educational purposes)
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        glVertexAttribPointer(0, 3, GL_FLOAT, false, 8 *Float.BYTES, 0);
        glEnableVertexAttribArray(0);

        int diffuseMap = loadTexture("src/main/resources/container2.png");
        int specularMap = loadTexture("src/main/resources/container2_specular.png");


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

            float angle = (float) (20 * GLFW.glfwGetTime());
            float radius = 2;
            Vector3f lightPos = new Vector3f(
                    (float) (Math.sin(Math.toRadians(angle)) * radius),
                    1.0f,
                    (float) (Math.cos(Math.toRadians(angle)) * radius)
            );


            shaderProgram.use();
            shaderProgram.setInt("material.diffuse",   0);
            shaderProgram.setInt("material.specular", 1);
            shaderProgram.setFloat("material.shininess", 32.0f);
            shaderProgram.setVec3("light.ambient",  0.2f, 0.2f, 0.2f);
            shaderProgram.setVec3("light.diffuse",  0.5f, 0.5f, 0.5f); // 将光照调暗了一些以搭配场景
            shaderProgram.setVec3("light.specular", 1.0f, 1.0f, 1.0f);
            shaderProgram.setVec3("light.position", camera.position);
            shaderProgram.setVec3("light.direction", camera.front);
            shaderProgram.setFloat("light.cutoff", (float) Math.cos(Math.toRadians(12.5f)));


            shaderProgram.setVec3("lightColor",  1f,1f,1f);
            shaderProgram.setVec3("viewPos", camera.position);


            Matrix4f projection  = new Matrix4f().perspective((float) Math.toRadians(camera.zoom),SCR_WIDTH/SCR_HEIGHT,0.1f,100f);
            Matrix4f view  =  camera.getViewMatrix();
            shaderProgram.setMat4("projection", projection);
            shaderProgram.setMat4("view",view);
//            shaderProgram.setMat4("model",new Matrix4f().translate(0f,1f,0f));

            // bind diffuse map
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, diffuseMap);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, specularMap);

            glBindVertexArray(VAO);
            for(int i =0;i<10;i++){
                Matrix4f model = new Matrix4f();
                model.translate(cubePosition[i]);
                float angle2 = 20 * i;
                model.rotate((float) Math.toRadians(angle2),new Vector3f(1.0f,0.3f,0.5f));
                shaderProgram.setMat4("model",model);
                glDrawArrays(GL_TRIANGLES, 0	, 36);
            }


            lightShader.use();
            lightShader.setMat4("projection",projection);
            lightShader.setMat4("view",view);
            lightShader.setMat4("model",new Matrix4f().translate(lightPos).scale(0.2f));

            glBindVertexArray(lightCubeVAO);
            glDrawArrays(GL_TRIANGLES, 0	, 36);

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


    public static int loadTexture(String path) throws IOException {
        int textureID;
        textureID = glGenTextures();

        ImageReader.ImageData imageData = ImageReader.ReadImage(path);
        if(imageData != null){
            glBindTexture(GL_TEXTURE_2D,textureID);
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,imageData.width,imageData.height,0,GL_RGBA,GL_UNSIGNED_BYTE,imageData.data);
            glGenerateMipmap(GL_TEXTURE_2D);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }else{
            System.out.println("fail load path"+path);
        }
        return textureID;
    }

}
```





```glsl
#version 330 core

struct Material{
    sampler2D diffuse;
    sampler2D specular;
    float shininess;
};

struct Light{
    vec3 position;
    vec3 direction;
    float cutoff;

    vec3 amnbien;
    vec3 diffuse;
    vec3 specular;
};

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

out vec4 FragColor;
uniform vec3 viewPos;
uniform Material material;
uniform Light light;

void main()
{

    // 环境光
    vec3 ambient = light.amnbien  * vec3(texture(material.diffuse,TexCoords));

    // 漫反射
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(light.position - FragPos);
    float diff = max(dot(norm,lightDir),0.0);
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse,TexCoords));

    //镜面
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir,norm);
    float spec = pow(max(dot(viewDir,reflectDir),0.0),material.shininess);
    vec3 specular = light.specular * spec * vec3(texture(material.specular,TexCoords));

    //
    float theta = dot(lightDir,normalize(-light.direction));
    if(theta > light.cutoff){
        vec3 result = ambient + diffuse + specular;
        FragColor = vec4(result,1.0);
    }else
        FragColor = vec4(light.amnbien * vec3(texture(material.diffuse, TexCoords)), 1.0);

}
```

## 模糊边缘

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

    private static float deltaTime = 0.0f;
    private static float lastFrame = 0.0f;
    private static float lastX = SCR_WIDTH/2;
    private static float lastY = SCR_HEIGHT/2;
    private static boolean firstMouse = true;
    private static Camera camera = new Camera(new Vector3f(0,0,3),new Vector3f(0,1,0),0,0);

    private static Vector3f lightPos = new Vector3f(1.2f,1.0f,2.0f);

    private static Vector3f[] cubePosition = {
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
        MyShader lightShader = new MyShader("lightcube.vert","lightcube.frag");

        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------

        float[] vertices = {
                -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,
                0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 0.0f,
                0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
                0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
                -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f,
                -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,

                -0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   0.0f, 0.0f,
                0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   1.0f, 0.0f,
                0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   1.0f, 1.0f,
                0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   1.0f, 1.0f,
                -0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   0.0f, 1.0f,
                -0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,   0.0f, 0.0f,

                -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
                -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
                -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
                -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
                -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
                -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,

                0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
                0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
                0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
                0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
                0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
                0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,

                -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,
                0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 1.0f,
                0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
                0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
                -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 0.0f,
                -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,

                -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f,
                0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 1.0f,
                0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
                0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
                -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f,
                -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f
        };
        int VBO = glGenBuffers();
        int VAO =  glGenVertexArrays();

        // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER,VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, false, 8 * Float.BYTES, 0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, false, 8 * Float.BYTES, 3 * Float.BYTES);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 2, GL_FLOAT, false, 8 * Float.BYTES, 6 * Float.BYTES);
        glEnableVertexAttribArray(2);


        int lightCubeVAO  = glGenVertexArrays();
        glBindVertexArray(lightCubeVAO);
        // we only need to bind to the VBO (to link it with glVertexAttribPointer), no need to fill it; the VBO's data already contains all we need (it's already bound, but we do it again for educational purposes)
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        glVertexAttribPointer(0, 3, GL_FLOAT, false, 8 *Float.BYTES, 0);
        glEnableVertexAttribArray(0);

        int diffuseMap = loadTexture("src/main/resources/container2.png");
        int specularMap = loadTexture("src/main/resources/container2_specular.png");


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

            float angle = (float) (20 * GLFW.glfwGetTime());
            float radius = 2;
            Vector3f lightPos = new Vector3f(
                    (float) (Math.sin(Math.toRadians(angle)) * radius),
                    1.0f,
                    (float) (Math.cos(Math.toRadians(angle)) * radius)
            );


            shaderProgram.use();
            shaderProgram.setInt("material.diffuse",   0);
            shaderProgram.setInt("material.specular", 1);
            shaderProgram.setFloat("material.shininess", 32.0f);
            shaderProgram.setVec3("light.ambient",  0.2f, 0.2f, 0.2f);
            shaderProgram.setVec3("light.diffuse",  0.5f, 0.5f, 0.5f); // 将光照调暗了一些以搭配场景
            shaderProgram.setVec3("light.specular", 1.0f, 1.0f, 1.0f);
            shaderProgram.setVec3("light.position", camera.position);
            shaderProgram.setVec3("light.direction", camera.front);
            shaderProgram.setFloat("light.cutoff", (float) Math.cos(Math.toRadians(12.5f)));
            shaderProgram.setFloat("light.outerCutOff", (float) Math.cos(Math.toRadians(17.5f)));


            shaderProgram.setVec3("lightColor",  1f,1f,1f);
            shaderProgram.setVec3("viewPos", camera.position);


            Matrix4f projection  = new Matrix4f().perspective((float) Math.toRadians(camera.zoom),SCR_WIDTH/SCR_HEIGHT,0.1f,100f);
            Matrix4f view  =  camera.getViewMatrix();
            shaderProgram.setMat4("projection", projection);
            shaderProgram.setMat4("view",view);
//            shaderProgram.setMat4("model",new Matrix4f().translate(0f,1f,0f));

            // bind diffuse map
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, diffuseMap);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, specularMap);

            glBindVertexArray(VAO);
            for(int i =0;i<10;i++){
                Matrix4f model = new Matrix4f();
                model.translate(cubePosition[i]);
                float angle2 = 20 * i;
                model.rotate((float) Math.toRadians(angle2),new Vector3f(1.0f,0.3f,0.5f));
                shaderProgram.setMat4("model",model);
                glDrawArrays(GL_TRIANGLES, 0	, 36);
            }


            lightShader.use();
            lightShader.setMat4("projection",projection);
            lightShader.setMat4("view",view);
            lightShader.setMat4("model",new Matrix4f().translate(lightPos).scale(0.2f));

            glBindVertexArray(lightCubeVAO);
            glDrawArrays(GL_TRIANGLES, 0	, 36);

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


    public static int loadTexture(String path) throws IOException {
        int textureID;
        textureID = glGenTextures();

        ImageReader.ImageData imageData = ImageReader.ReadImage(path);
        if(imageData != null){
            glBindTexture(GL_TEXTURE_2D,textureID);
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,imageData.width,imageData.height,0,GL_RGBA,GL_UNSIGNED_BYTE,imageData.data);
            glGenerateMipmap(GL_TEXTURE_2D);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }else{
            System.out.println("fail load path"+path);
        }
        return textureID;
    }

}
```







```glsl
#version 330 core

struct Material{
    sampler2D diffuse;
    sampler2D specular;
    float shininess;
};

struct Light{
    vec3 position;
    vec3 direction;
    float cutoff;
    float outerCutOff;

    vec3 amnbien;
    vec3 diffuse;
    vec3 specular;
};

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

out vec4 FragColor;
uniform vec3 viewPos;
uniform Material material;
uniform Light light;

void main()
{

    // 环境光
    vec3 ambient = light.amnbien  * vec3(texture(material.diffuse,TexCoords));

    // 漫反射
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(light.position - FragPos);
    float diff = max(dot(norm,lightDir),0.0);
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse,TexCoords));

    //镜面
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir,norm);
    float spec = pow(max(dot(viewDir,reflectDir),0.0),material.shininess);
    vec3 specular = light.specular * spec * vec3(texture(material.specular,TexCoords));

    //
    float theta = dot(lightDir,normalize(-light.direction));
    float epsilon = light.cutoff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff)/epsilon,0.0,1.0);

    diffuse *= intensity;
    specular *= intensity;

    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result,1.0);

}
```

