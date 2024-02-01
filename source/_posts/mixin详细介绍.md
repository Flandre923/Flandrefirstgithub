---
title: mixin进一步简介
date: 2023-12-04 17:56:12
tags:
- 我的世界
- java
- fabric
- mixin
cover: https://view.moezx.cc/images/2022/03/26/6d1f8382cfa121721d7efd29f8e1e9b9.jpg
---



# introduction

1. Mixins是Fabric生态系统中的一个强大而重要的工具。它们的主要用途是修改基础游戏中的现有代码，可以通过注入自定义逻辑、删除机制或修改数值来实现。
2. Mixins必须使用Java编写，即使你使用Kotlin或其他语言。
3. 如果想要了解Mixin的功能、用法和机制的详细步骤，可以查看Mixin官方Wiki。Mixin Javadoc中还有其他文档可供参考。
4. Fabric Wiki还提供了几篇实际示例和解释的文章，可以提供更多实用的示例和解释。

## 文献

[Introduction to Mixins [Fabric Wiki\] (fabricmc.net)](https://fabricmc.net/wiki/tutorial:mixin_introduction)

[Mixin Official Wiki](https://github.com/SpongePowered/Mixin/wiki). 

[Mixin Javadoc](https://jenkins.liteloader.com/view/Other/job/Mixin/javadoc/index.html).

## Mixins

一些mixin可能会导致冲突，但负责任地使用它们是为mod添加独特行为的关键。

Mixins有各种各样的口味，大致按偏好顺序排列：

- Adding Interfaces 添加接口
- Callback Injectors 回调注入
- Redirect Injectors 重定向注入
- Overwrites 覆盖

这不是一个完整的列表，而是一个快速的概述。这里省略了一些mixin类型。

### Adding Interfaces

这可能是使用mixin最安全的方法之一。新的接口实现可以添加到任何Minecraft类中。然后你可以通过强制转换类来访问接口。这不会改变类的任何内容，只是增加了新的东西，因此不太可能发生冲突。

### Callback Injectors

回调注入器允许您将回调钩子添加到现有方法，以及该方法中的特定方法调用。它们还允许您拦截和更改方法的返回值，并提前返回。回调注入可以堆栈，因此不太可能导致mods之间的冲突。

### Redirect Injectors

重定向使您可以使用自己的代码包装目标方法中的方法调用或变量访问。使用这些非常谨慎，一个目标调用或访问只能在所有MOD之间重定向一次。如果两个mod重定向相同的值，这将导致冲突。首先考虑回调注入。

### Overwrite 

使用覆盖时要非常小心。它们完全替换方法，删除所有现有代码，并与方法上的任何其他类型的mixin冲突。它们极有可能不仅与其他MOD冲突，而且与Minecraft本身的更改也发生冲突。您很可能不需要覆盖来执行您想要执行的操作。

# Registering Mixins

- 在资源文件夹中的fabric.mod.json文件中注册Mixins。
- 在fabric.mod.json文件中的mixins数组中添加元素来告诉Fabric应该在哪里查找Mixins。
- 使用fabric.mod.json文件来定义Fabric应该在哪里查找mixins.json文件。
- mixins.json文件中定义了要注册的Mixin的详细信息。
- 通过添加元素到fabric.mod.json文件的mixins数组中，告诉Fabric在哪里注册Mixin。
- ![image-20231204225257597](https://s2.loli.net/2023/12/04/1JrQlEjhxe7KBGd.png)

- 在mixins数组中提供字符串"<modid>.mixins.json"告诉Fabric加载在文件<modid>.mixins.json中定义的mixins。
- 需要定义要加载的mixins以及这些mixins的位置。
- 在注册的<modid>.mixins.json文件中完成上述定义。
- ![image-20231204225501253](https://s2.loli.net/2023/12/04/n5aSWLP4x17vCYl.png)

- 在开始使用mixins时，你应该关注的四个主要字段是package字段以及mixins、client和server数组。
- package字段定义了在哪个文件夹（包）中查找Mixins。
- mixins数组定义了应该在客户端和服务器上加载的类。
- client数组定义了应该在客户端上加载的类。
- server数组定义了应该在服务器上加载的类。
- 根据这个逻辑：net.fabricmc.example.mixin.TitleScreenMixin是将在客户端上加载的mixin类。

# Mixin Injects

## introduciton

- Mixin Injects允许您在现有方法的指定位置放置自定义代码。
- 注入的标准形式如下所示：

```java
@Inject(method = "方法名或者签名", at = @At("注入点参考"))
private void injectMethod(METHOD ARGS, CallbackInfo info) {
 
}
```

- 注入参考点是指在方法体内注入代码的位置。可以使用几个选项来指定注入点，例如HEAD、RETURN、INVOKE和TAIL。

- | Name   | Description        |
  | :----- | :----------------- |
  | HEAD   | 方法顶部           |
  | RETURN | 每一个放回语句前   |
  | INVOKE | 在方法调用时       |
  | TAIL   | 在最终返回声明之前 |

- 注入点可以引用语句或成员，目标值可以在@At中设置。目标值使用JVM字节码描述符来指定。

- | Descriptor    | Primitive | Description                                         |
  | :------------ | :-------- | :-------------------------------------------------- |
  | B             | byte      | 有符号字节                                          |
  | C             | char      | 基本多语言平面中的Unicode字符代码点，使用UTF-16编码 |
  | D             | double    | 双精度浮点数                                        |
  | F             | float     | 单精度浮点数                                        |
  | I             | int       | 有符号整数                                          |
  | J             | long      | 有符号长整数                                        |
  | L*ClassName*; | reference | ClassName*的实例                                    |
  | S             | short     | 有符号短整型                                        |
  | Z             | boolean   | `true` or `false`                                   |
  | [             | reference | 一个数组维度                                        |

- 方法描述符由方法名称、包含参数类型的括号和返回类型组成。如果返回类型是void，需要使用V（Void Descriptor Type）作为类型。

```java
 Object m(int i, double[] d, Thread t)

m(I[DLjava/lang/Thread;)Ljava/lang/Object
```

- 方法“void foo(String bar)”的 方法描述符是“foo(Ljava/lang/String;)V”。
- Generics' types are left out, as generics don't exist on runtime. So `Pair<Integer, ? extends Task<? super VillagerEntity>‍>` would become `Lcom/mojang/datafixers/util/Pair`.
- @Inject 注解的方法始终具有 void 返回类型。方法名称和访问修饰符都不重要，使用描述注入作用的名称最佳。目标方法的参数首先放置在方法头中，后跟一个 CallbackInfo 对象。如果目标方法具有返回类型 (T)，则使用 CallbackInfoReturnable<T> 而非 CallbackInfo。

```java
@Inject
public void initialize(CallbackInfo info) {
    // 执行初始化逻辑
}

@Inject
public void processRequest(String request, CallbackInfoReturnable<Response> info) {
    // 处理请求逻辑
    // ...
    return response;
}

```



- 在方法中提前返回或取消操作时，可以使用CallbackInfo#cancel或CallbackInfoReturnable<T>#setReturnValue(T)。在@Inject注解中，cancellable必须设置为true才能使用这些功能。

```java
@Inject(method = "...", at = @At("..."), cancellable = true)

```



- 要注入构造函数，请使用 <init>()V 作为方法目标，其中 () 包含构造函数参数描述符。 在注入构造函数时，@At 必须设置为 TAIL 或 RETURN。 官方不支持其他形式的注入。 请注意，一些类的名称为 init 的方法与 <init> 不同。 不要混淆！
- 要注入静态构造函数，请使用 <clinit> 作为方法名。
- 例子(这是一个普通的init方法不是构造方法)

```java
@Mixin(TitleScreen.class)
public class ExampleMixin {
	@Inject(at = @At("HEAD"), method = "init()V")
	private void init(CallbackInfo info) {
		System.out.println("This line is printed by an example mod mixin!");
	}
}
```

# Mixin Accessors & Invokers

Mixin Accessors & Invokers 是一种用于访问或修改非公共或 final 字段和方法的机制

## accessors

@Accessors注解用于访问字段。它允许开发者读取和写入原本无法访问的字段的值。

### Getting a value from the field

假设你要访问MinecraftClient类的itemUseCooldown字段

```java
@Mixin(MinecraftClient.class)
public interface MinecraftClientAccessor {
    @Accessor
    int getItemUseCooldown();
}
```

使用方法：

```java
int itemUseCooldown = ((MinecraftClientAccessor) MinecraftClient.getInstance()).getItemUseCooldown();

```



### Setting a value to the field

```java
@Mixin(MinecraftClient.class)
public interface MinecraftClientAccessor {
    @Accessor("itemUseCooldown")
    public void setItemUseCooldown(int itemUseCooldown);
}
```

使用方法

```java
((MinecraftClientAccessor) MinecraftClient.getInstance()).setItemUseCooldown(100);

```

## Accessor for static fields

假设我们希望访问VanillaLayeredBiomeSource类的BIOMES字段

### Getting a value from the field

```java
@Mixin(VanillaLayeredBiomeSource.class)
public interface VanillaLayeredBiomeSourceAccessor {
  @Accessor("BIOMES")
  public static List<RegistryKey<Biome>> getBiomes() {
    throw new AssertionError();
  }
}
```

使用

```java
List<RegistryKey<Biome>> biomes = VanillaLayeredBiomeSourceAccessor.getBiomes();

```

### Setting a value to the field

```java
@Mixin(VanillaLayeredBiomeSource.class)
public interface VanillaLayeredBiomeSourceAccessor {
  @Accessor("BIOMES")
  public static void setBiomes(List<RegistryKey<Biome>> biomes) {
    throw new AssertionError();
  }
}
```

使用方法：

```java
VanillaLayeredBiomeSourceAccessor.setBiomes(biomes);
```

## Invoker

@Invoker允许你方法。假设我们想要访问EndermanEntity类的teleportTo方法。

```java
@Mixin(EndermanEntity.class)
public interface EndermanEntityInvoker {
  @Invoker("teleportTo")
  public boolean invokeTeleportTo(double x, double y, double z);
}
```

使用例子

```java
EndermanEntity enderman = ...;
((EndermanEntityInvoker) enderman).invokeTeleportTo(0.0D, 70.0D, 0.0D);
```

## Invoker for static methods

假设我们想要调用BrewingRecipeRegistry类的方法registerPotionType

```java
@Mixin(BrewingRecipeRegistry.class)
public interface BrewingRecipeRegistryInvoker {
  @Invoker("registerPotionType")
  public static void invokeRegisterPotionType(Item item) {
    throw new AssertionError();
  }
}
```

使用方法

```java
BrewingRecipeRegistryInvoker.invokeRegisterPotionType(item);

```

# Mixin redirectors

## introduction

重定向器（Redirectors）是可以替代方法调用、字段访问、对象创建和 instanceof 检查的方法。重定向器通过 @Redirect 注解声明，并且通常的格式如下：

```java
@Redirect(method = "${signatureOfMethodInWhichToRedirect}",
          at = @At(value = "${injectionPointReference}", target = "${signature}"))
public ReturnType redirectSomeMethod(Arg0Type arg0, Arg1Type arg1) {
    MyClass.doMyComputations();
 
    return computeSomethingElse();
}
```

请参考特定的重定向教程，了解有关注入点引用的信息。

- [redirecting methods](https://fabricmc.net/wiki/tutorial:mixin_redirectors_methods)
- [redirecting field access](https://fabricmc.net/wiki/tutorial:mixin_redirectors_fields)
- [redirecting object creation](https://fabricmc.net/wiki/tutorial:mixin_redirectors_constructors)
- [redirecting instanceof checks](https://fabricmc.net/wiki/tutorial:mixin_redirectors_instanceof)

> 由于重定向可能会导致模组之间冲突这里就不详细讲了

# Mixin Tips (WIP)

这是一些可能有用的不同技巧的集合。建议阅读以前的文章以了解技巧。

### Why make a class abstract? （为什么要创建一个抽象类？）

**1. Prevent instantiation  （1.防止实例化）**

公平地说，你永远不应该实例化一个mixin类，主要是因为如果这个类没有在mixin环境中使用，它对java来说就没有意义了，还有其他方法可以访问mixin中声明的方法。

声明一个mixin类的抽象并不影响它的功能，并且它可以防止意外的实例化。

```java
MixinClass foo = new MixinClass(); // can't do that with an abstract class

```

**2. Make more elegant shadow methods**(**制作更优雅的shadow方法**)

如果你想从你的目标类中访问一个不可访问的方法或字段到你的mixin类中，你需要使用 `@Shadow` 来使那个方法/字段可见。

你可以通过使用虚拟方法体(dummy body)在普通类中完美地做到这一点:

```java
@Shadow
protected void hiddenMethod() {/*dummy body*/}
```

但使用抽象方法（因此是抽象类）要优雅得多：

```java
@Shadow
protected abstract void hiddenMethod(); // no dummy body necessary
```

注意：这不适用于私有方法，因为你不能有私有抽象方法，因此你需要使用普通方法。

**3. Access the `this` instance more easily** 轻松地访问 `this` 实例

在mixin中，如果你想访问“this”实例，你必须在mixin类中进行强制转换：

```java
((TargetClass)(Object)this).field/method();

```

但是，只有当你的mixin类扩展/实现了你的目标类所做的一切时，这才有效，这可能是一个痛苦，因为如果其中一个类/接口有一个你需要实现的方法，你就会遇到一些麻烦。

幸运的是，这一切都可以通过使用抽象类来避免，在这种情况下，您不必实现方法，所有问题都可以避免。

### How to mixin inner classes 如何混合内部类

**1. Normal inaccessible inner classes 正规不可达内部类**

由于您不能从外部直接访问（并提及）这些类，因此需要使用mixin注释的“targets”字段来指定名称。

你可以使用外部类的完整名称，然后是 `$` ，最后是内部类的名称，如下所示：

Class: 

```java
package some.random.package;
 
public class Outer {
     private class Inner {
         public void someRandomMethod() {}
     }
}
```

Mixin with injection:

```java
@Mixin(targets = "some.random.package.Outer$Inner")
public class MyMixin {
    @Inject(method = "someRandomMethod()V", at = @At("HEAD")
    private void injected(CallbackInfo ci) {
        // your code here
    }
}
```

唯一需要注意的是，如果你想混入内部类构造函数，第一个参数必须是外部类的类型（这是编译器隐式添加的，以允许内部类访问外部类的实例方法）：

```java
@Inject(method = "<init>(Lsome/random/package/Outer;)V", at = @At("HEAD")
private void injected(CallbackInfo ci) {
    // your code here
}
```

**2. Static inaccessible inner classes
静态不可访问的内部类**

这些和上面一样，唯一的区别是构造函数没有外部类的第一个参数（因为在静态内部类中，只有私有静态方法可以从内部类访问，因此不需要该参数）。

**3. Anonymous inner classes 3.匿名内部类**

这些与静态不可访问的内部类相同，唯一的区别是，由于它们没有名称，因此它们是按外观顺序声明的，例如：匿名内部类如果在我们前面的示例中声明，首先将命名为Outer\$1，第二个将命名为Outer\$2，第三个命名为Outer\$3，依此类推（声明顺序是在源代码级别）。

# Mixin Examples

这是一些常用的混入（Mixin）示例。本页面旨在作为一个速查表。如果您还没有阅读过混入介绍，请参阅混入介绍。

## Mixing into a private inner class

使用 targets 参数和 $ 符号来获取内部类。

```java
@Mixin(targets = "net.minecraft.client.render.block.BlockModelRenderer$AmbientOcclusionCalculator")
public class AmbientOcclusionCalculatorMixin {
    // do your stuff here
}
```



## Access the this instance of the class your mixin is targeting访问混入目标类的实例

注意：在可能的情况下应避免双重转型。如果方法或字段来自目标类，请使用 `@Shadow`。如果方法或字段来自目标类的父类，请将混入类直接继承目标类的直接父类。

```java
@Mixin(TargetClass.class)
public class MyMixin extends EveryThingThatTargetClassExtends implements EverythingThatTargetClassImplements {
  @Inject(method = "foo()V", at = @At("HEAD"))
  private void injected(CallbackInfo ci) {
    ((TargetClass)(Object)this).methodOfTheTargetClass();
  }
}
```

## Injecting into the head of a static block 注入到一个静态块的头部

mixin:

```java
@Inject(method = "<clinit>", at = @At("HEAD"))
private void injected(CallbackInfo ci) {
    doSomething3();
}
```

result:

```diff
static {
+   injected(new CallbackInfo(“<clinit>”, false));
    doSomething1();
    doSomething2();
}
```

## Injecting into the head of a method 注入到一个方法的头部

Mixin:

```java
@Inject(method = "foo()V", at = @At("HEAD"))
private void injected(CallbackInfo ci) {
  doSomething4();
}
```

Result:

```diff
  public void foo() {
+   injected(new CallbackInfo("foo", false));
    doSomething1();
    doSomething2();
    doSomething3();
  }
```

## Injecting into the tail of a method注入到一个方法末尾

Mixin:

```java
@Inject(method = "foo()V", at = @At("TAIL"))
private void injected(CallbackInfo ci) {
  doSomething4();
}
```

Result:

```diff
  public void foo() {
    doSomething1();
    if (doSomething2()) {
      return;
    }
    doSomething3();
+   injected(new CallbackInfo("foo", false));
  }
```

## Injecting into the returns of a method  注入到方法返回前

Mixin:

```java
@Inject(method = "foo()V", at = @At("RETURN"))
private void injected(CallbackInfo ci) {
  doSomething4();
}
```

Result:

```diff
  public void foo() {
    doSomething1();
    if (doSomething2()) {
+     injected(new CallbackInfo("foo", false));
      return;
    }
    doSomething3();
+   injected(new CallbackInfo("foo", false));
  }
```

## Injecting into the point before a method call  注入到一个方法调用前

Mixin:

```java
@Inject(method = "foo()V", at = @At(value = "INVOKE", target = "La/b/c/Something;doSomething()V"))
private void injected(CallbackInfo ci) {
  doSomething3();
}
```

Result:

```diff
  public void foo() {
    doSomething1();
    Something something = new Something();
+   injected(new CallbackInfo("foo", false));
    something.doSomething();
    doSomething2();
  }
```

## Injecting into the point after a method call

Mixin:

```java
@Inject(method = "foo()V", at = @At(value = "INVOKE", target = "La/b/c/Something;doSomething()V", shift = At.Shift.AFTER))
private void injected(CallbackInfo ci) {
  doSomething3();
}
```

Result:

```diff
  public void foo() {
    doSomething1();
    Something something = new Something();
    something.doSomething();
+   injected(new CallbackInfo("foo", false));
    doSomething2();
  }
```

## Injecting into the point before a method call with shift amount 在方法调用之前以位移量注入的内容。

Mixin:

```java
@Inject(method = "foo()V", at = @At(value = "INVOKE", target = "La/b/c/Something;doSomething()V", shift = At.Shift.BY, by = 2))
private void injected(CallbackInfo ci) {
  doSomething3();
}
```

Result:

```diff
  public void foo() {
    doSomething1();
    Something something = new Something();
    something.doSomething();
    doSomething2();
+   injected(new CallbackInfo("foo", false));
    }
```

## Injecting with a slice在切片范围内注入

```java
@Inject(
  method = "foo()V",
  at = @At(
    value = "INVOKE",
    target = "La/b/c/Something;doSomething()V"
  ),
  slice = @Slice(
    from = @At(value = "INVOKE", target = "La/b/c/Something;doSomething2()V"),
    to = @At(value = "INVOKE", target = "La/b/c/Something;doSomething3()V")
  )
)
private void injected(CallbackInfo ci) {
  doSomething5();
}
```

Result:

```diff
  public class Something {
    public void foo() {
      this.doSomething1();
+     // It will not inject into here because this is outside of the slice section
      this.doSomething();
      this.doSomething2();
+     injected(new CallbackInfo("foo", false));
      this.doSomething();
      this.doSomething3();
+     // It will not inject into here because this is outside of the slice section
      this.doSomething();
      this.doSomething4();
    }
  }
```

## Injecting and cancelling 注入和取消

Mixin:

```java
@Inject(method = "foo()V", at = @At("HEAD"), cancellable = true)
private void injected(CallbackInfo ci) {
  ci.cancel();
}
```

Result:

```diff
  public void foo() {
+   CallbackInfo ci = new CallbackInfo("foo", true);
+   injected(ci);
+   if (ci.isCancelled()) return;
    doSomething1();
    doSomething2();
    doSomething3();
  }
```

## Injecting and cancelling with a return value 注入取消并且返回数值

Mixin:

```java
@Inject(method = "foo()I;", at = @At("HEAD"), cancellable = true)
private void injected(CallbackInfoReturnable<Integer> cir) {
  cir.setReturnValue(3);
}
```

Result:

```diff
  public int foo() {
+   CallbackInfoReturnable<Integer> cir = new CallbackInfoReturnable<Integer>("foo", true);
+   injected(cir);
+   if (cir.isCancelled()) return cir.getReturnValue();
    doSomething1();
    doSomething2();
    doSomething3();
    return 10;
  }
```

## Capturing local values 捕获本地数值

Mixin:

```java
@Inject(method = "foo()V", at = @At(value = "TAIL"), locals = LocalCapture.CAPTURE_FAILHARD)
private void injected(CallbackInfo ci, TypeArg1 arg1) {
  //CAPTURE_FAILHARD: If the calculated locals are different from the expected values, throws an error.
  arg1.doSomething4();
}
```

Result:

```diff
  public void foo() {
    TypeArg1 arg1 = getArg1();
    arg1.doSomething1();
    arg1.doSomething2();
    TypeArg2 arg2 = getArg2();
    arg2.doSomething3();
+   injected(new CallbackInfo("foo", false), arg1);
  }
```

## Modifying a return value 修改返回值

Mixin:

```java
@Inject(method = "foo()I;", at = @At("RETURN"), cancellable = true)
private void injected(CallbackInfoReturnable<Integer> cir) {
  cir.setReturnValue(cir.getReturnValue() * 3);
}
```

Result:

```diff
  public int foo() {
    doSomething1();
    doSomething2();
-   return doSomething3() + 7;
+   int i = doSomething3() + 7;
+   CallbackInfoReturnable<Integer> cir = new CallbackInfoReturnable<Integer>("foo", true, i);
+   injected(cir);
+   if (cir.isCancelled()) return cir.getReturnValue();
+   return i;
  }
```

## Redirecting a method call 重定向一个方法调用

Mixin:

```java
@Redirect(method = "foo()V", at = @At(value = "INVOKE", target = "La/b/c/Something;doSomething(I)I"))
private int injected(Something something, int x) {
  return x + 3;
}
```

Result:

```diff
  public void foo() {
    doSomething1();
    Something something = new Something();
-   int i = something.doSomething(10);
+   int i = injected(something, 10);
    doSomething2();
  }
```

## Redirecting a get field  重定向获得一个字段

Mixin:

```java
@Redirect(method = "foo()V", at = @At(value = "FIELD", target = "La/b/c/Something;aaa:I", opcode = Opcodes.GETFIELD))
private int injected(Something something) {
  return 12345;
}
```

Result:

```diff
  public class Something {
    public int aaa;
    public void foo() {
      doSomething1();
-     if (this.aaa > doSomething2()) {
+     if (injected(this) > doSomething2()) {
        doSomething3();
      }
      doSomething4();
    }
  }
```

## Redirecting a put field 重定向 放置一个字段

Mixin:

```java
@Redirect(method = "foo()V", at = @At(value = "FIELD", target = "La/b/c/Something;aaa:I", opcode = Opcodes.PUTFIELD))
private void injected(Something something, int x) {
  something.aaa = x + doSomething5();
}
```

Result:

```diff
  public class Something {
    public int aaa;
    public void foo() {
      doSomething1();
-     this.aaa = doSomething2() + doSomething3();
+     inject(this, doSomething2() + doSomething3());
      doSomething4();
    }
  }
```

## Modifying an argument 修改一个参数

Mixin:

```java
@ModifyArg(method = "foo()V", at = @At(value = "INVOKE", target = "La/b/c/Something;doSomething(ZIII)V"), index = 2)
private int injected(int x) {
  return x * 3;
}
```

Result:

```diff
  public void foo() {
    doSomething1();
    Something something = new Something();
-   something.doSomething(true, 1, 4, 5);
+   something.doSomething(true, 1, injected(4), 5);
    doSomething2();
  }
```

## Modifying multiple arguments 修改多个参数

Mixin:

```java
@ModifyArgs(method = "foo()V", at = @At(value = "INVOKE", target = "La/b/c/Something;doSomething(IDZ)V"))
private void injected(Args args) {
    int a0 = args.get(0);
    double a1 = args.get(1);
    boolean a2 = args.get(2);
    args.set(0, a0 + 3);
    args.set(1, a1 * 2.0D);
    args.set(2, !a2);
}
```

Result:

```diff
  public void foo() {
    doSomething1();
    Something something = new Something();
-   something.doSomething(3, 2.5D, true);
+   // Actually, synthetic subclass of Args is generated at runtime,
+   // but we omit the details to make it easier to understand the concept.
+   Args args = new Args(new Object[] { 3, 2.5D, true });
+   injected(args);
+   something.doSomething(args.get(0), args.get(1), args.get(2));
    doSomething2();
  }
```

## Modifying a parameter 修改parameter

Mixin:

```java
@ModifyVariable(method = "foo(ZIII)V", at = @At("HEAD"), ordinal = 1)
private int injected(int y) {
  return y * 3;
}
```

Result:

```diff
  public void foo(boolean b, int x, int y, int z) {
+   y = injected(y);
    doSomething1();
    doSomething2();
    doSomething3();
  }
```

## Modifying a local variable on an assignment 修改一个局域变量

Mixin:

```java
@ModifyVariable(method = "foo()V", at = @At("STORE"), ordinal = 1)
private double injected(double x) {
  return x * 1.5D;
}
```

Result:

```diff
  public void foo() {
    int i0 = doSomething1();
    double d0 = doSomething2();
-   double d1 = doSomething3() + 0.8D;
+   double d1 = injected(doSomething3() + 0.8D);
    double d2 = doSomething4();
  }
```

## Modifying a constant 修改一个常量

Mixin:

```java
@ModifyConstant(method = "foo()V", constant = @Constant(intValue = 4))
private int injected(int value) {
  return ++value;
}
```

Result:

```diff
  public void foo() {
-   for (int i = 0; i < 4; i++) {
+   for (int i = 0; i < injected(4); i++) {
      doSomething(i);
    }
  }
```

# Exporting Mixin Classes 导出Mixin类

在调试mixin时，能够看到已完成的类以及插入的更改和注入是非常有用的。Mixin提供了一个标志，允许这样做：

```
-Dmixin.debug.export=true
```

这应该放在您的VM选项中。加载类后，它们将出现在 `\run\.mixin.out` 中

![img](https://fabricmc.net/wiki/lib/exe/fetch.php?tok=68f5a2&media=https%3A%2F%2Fi.imgur.com%2Fd7oKQkg.png)

### Only Exporting Required Classes 仅导出必需的类

转储每一个混合类可能是有用的，但通常是不必要的，确实会减慢minecraft的速度。Mixin提供了一个方便的注释，用于将调试功能应用于单个mixin：

```java
@Debug(export = true) // Enables exporting for the targets of this mixin
@Mixin(...)
public class MyMixin {
    // Mixin code here
}
```

注意：某些类可能在游戏运行（或世界加载）之前不会出现。

# Interface Injection 接口注入

这是Loom 0.11引入的一项新技术，用于将方法添加到特定的现有类中。更具体地说，您可以创建一个接口，然后将此接口注入到类中。因此，目标类将获取接口的所有方法，就好像它总是拥有它们一样。接口注入是一个编译时的特性，这意味着Mixin也应该被用来将接口实现到目标类中。

这对库特别有用，有了它，你可以向现有的类添加新的方法并使用它们，而不需要每次都转换或重新实现接口。

让我们用一个例子更好地解释：

这个例子的范围是将下面的方法添加到 `net.minecraft.fluid.FlowableFluid` 中，以获得桶清空时的声音。这通常是不可能的，因为 `net.minecraft.fluid.FlowableFluid` 没有类似的方法。

```java
Optional<SoundEvent> getBucketEmptySound()
```

要将方法添加到类中，首先需要创建一个接口：

```java
package net.fabricmc.example;
 
public interface BucketEmptySoundGetter {
	// The methods in an injected interface MUST be default,
	// otherwise code referencing them won't compile!
	default Optional<SoundEvent> getBucketEmptySound() {
		return Optional.empty();
	}
}
```

现在你需要用一个mixin实现这个接口，把这个接口实现到 `net.minecraft.fluid.FlowableFluid` 中：

```java
@Mixin(FlowableFluid.class)
public class MixinFlowableFluid implements BucketEmptySoundGetter {
	@Override
	public Optional<SoundEvent> getBucketEmptySound() {
	    //This is how to get the default sound, copied from BucketItem class.
	    return Optional.of(((FlowableFluid) (Object) this).isIn(FluidTags.LAVA) ? SoundEvents.ITEM_BUCKET_EMPTY_LAVA : SoundEvents.ITEM_BUCKET_EMPTY);
	}
}
```

最后，你需要将接口注入到 `net.minecraft.fluid.FlowableFluid` 中。可以将以下代码片段添加到fabric.mod.json文件中，以向 `net.minecraft.fluid.FlowableFluid` 类添加一个或多个接口。请注意，这里所有的类名都必须使用“内部名称”，即使用斜线而不是点（ `path/to/my/Class` ）。

```json
{
	"custom": {
		"loom:injected_interfaces": {
			"net/minecraft/class_3609": ["net/fabricmc/example/BucketEmptySoundGetter"]
		}
	}
}
```

现在你可以使用新的方法：

```java
Optional<SoundEvent> sound = mytestfluid.getBucketEmptySound();
```

您也可以在扩展FlowableFluid的类中重写此方法以实现自定义行为。

# 更进一步？

[Home · SpongePowered/Mixin Wiki (github.com)](https://github.com/SpongePowered/Mixin/wiki)

# 参考文献

[Introduction to Mixins [Fabric Wiki\] (fabricmc.net)](https://fabricmc.net/wiki/tutorial:mixin_introduction)

[Registering Mixins [Fabric Wiki\] (fabricmc.net)](https://fabricmc.net/wiki/tutorial:mixin_registration)

[Mixin Injects [Fabric Wiki\] (fabricmc.net)](https://fabricmc.net/wiki/tutorial:mixin_injects)

[Mixin Accessors & Invokers [Fabric Wiki\] (fabricmc.net)](https://fabricmc.net/wiki/tutorial:mixin_accessors)

[Introduction to Modding with Fabric [Fabric Wiki\] --- 使用Fabric进行改装的简介[Fabric Wiki] (fabricmc.net)](https://fabricmc.net/wiki/tutorial:introduction)

[Mixin Tips (WIP) [Fabric Wiki\] --- Mixin Tips（Mixin Tips）[织物Wiki] (fabricmc.net)](https://fabricmc.net/wiki/tutorial:mixin_tips)
