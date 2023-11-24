---
title: 01_pytorch_tensors
date: 2023-11-10 14:32:11
tags:
- 深度学习
- pytorch
- 记录
cover: https://w.wallhaven.cc/full/qz/wallhaven-qzpkrr.jpg
---

# Tensors

## Tensors是什么

张量是一种特殊的数据结构，与数组和矩阵非常相似。

## Tensors什么作用

张量对模型的输入和输出以及模型的参数进行编码

## Tensors 和 [NumPy](https://numpy.org/)的ndarrays对比

不同：

1. 张量可以在 GPU 或其他硬件加速器上运行
2. 张量还针对自动微分进行了优化

## Tensors 和Numpy的ndarrays联系

张量和 NumPy 数组通常可以共享相同的底层内存，从而无需复制数据。

## Initializing a Tensor

1.通过数据创建，类型可以自动推导

```python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

2.通过numpy创建

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

3.从另一个Tensor创建,新Tensor保留参数Tensor的属性（**形状、数据类型**），除非显式覆盖。

```python
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")
# 显示覆盖类型
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
```

```
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.8823, 0.9150],
        [0.3829, 0.9593]])
```



## **With random or constant values:**

`shape` 是张量维度的元组。在下面的函数中，它确定输出张量的维数。

```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

```
Random Tensor:
 tensor([[0.3904, 0.6009, 0.2566],
        [0.7936, 0.9408, 0.1332]])

Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

## Attributes of a Tensor 

```python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

```
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

## Operations on Tensors

### 有哪些运算

 100 多种[张量运算](https://pytorch.org/docs/stable/torch.html)，包括算术、线性代数、矩阵操作（转置、索引、切片）、采样等。

### 和一般的计算库相比有哪些优点

这些操作中的每一个都可以在 GPU 上运行

> 如果使用Colab如何使用GPU：
>
> 请通过转至运行时 > 更改运行时类型 > GPU 来分配 GPU

### 为什么我创建后在cpu上？

默认情况下，张量是在 CPU 上创建的。我们需要使用 `.to` 方法显式地将张量移动到 GPU

> 请记住，跨设备复制大张量在时间和内存方面可能会很昂贵！

```python
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
```

### 对tensor的切片索引修改

```python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
```

```
First row: tensor([1., 1., 1., 1.])
First column: tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

### tensor的拼接，维度扩大

连接张量 您可以使用 `torch.cat` 沿给定维度连接一系列张量。

另请参见 torch.stack，这是另一个与 `torch.cat` 略有不同的张量连接运算符。

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

```
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```

### **Arithmetic operations**

```python
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T # tensor 矩阵乘 tesnor的转置
y2 = tensor.matmul(tensor.T) # 同上

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3) # 矩阵乘，结果输出给y3


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor # 计算元素的乘积，下同
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```

### 如何将单一数值的tensor和python的数值类型进行转化

```python
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```

```
12.0 <class 'float'>
```

### tensors的**In-place operations**

就地运算 将结果存储到操作数中的操作称为就地运算。它们由 `_` 后缀表示。例如： `x.copy_(y)` 、 `x.t_()` 会更改 `x` 。

```python
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
```

```
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```

> 确保你的数据没用了，再采取就地运算(In-place operations)

## 和numpy的联系

1. CPU上时，NumPy 数组上的张量可以共享其底层内存位置。

### Tensor to NumPy array

```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

```
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
```

当t数据变化时候，对应的n也发生变化，简单理解就是n是对t的引用（指针）

```python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

```
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```

### NumPy array to Tensor 

```
n = np.ones(5)
t = torch.from_numpy(n)
```

NumPy 数组中的变化反映在张量中。

```
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

```
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
```

