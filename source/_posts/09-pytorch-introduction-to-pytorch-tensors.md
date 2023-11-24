---
title: 09-pytorch-introduction to pytorch tensors
date: 2023-11-13 15:04:29
tags:
- 深度学习
- pytorch
cover: https://view.moezx.cc/images/2020/07/04/a7dc77598363f55791bf5a1f241bbb8b.png
---

# INTRODUCTION TO PYTORCH TENSORS 

张量是 PyTorch 中的核心数据抽象。此交互式笔记本深入介绍了 `torch.Tensor` 类。

首先，让我们导入 PyTorch 模块。我们还将添加 Python 的数学模块来简化一些示例

```python
import torch
import math
```

## Creating Tensors 

创建张量的最简单方法是使用 `torch.empty()` 调用：

```python
x = torch.empty(3, 4)
print(type(x))
print(x)
```

```
<class 'torch.Tensor'>
tensor([[7.4055e-29, 0.0000e+00, 7.4101e-04, 0.0000e+00],
        [2.8026e-45, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]])
```

让我们解释我们刚刚所做的事情：

- 我们使用 `torch` 模块附带的众多工厂方法之一创建了一个张量。

- 张量本身是二维的，有 3 行和 4 列。
- 返回对象的类型是 `torch.Tensor` ，它是 `torch.FloatTensor` 的别名；默认情况下，PyTorch 张量由 32 位浮点数填充。 （下面详细介绍数据类型。）
- 打印张量时，您可能会看到一些看起来随机的值。 `torch.empty()` 调用为张量分配内存，但不使用任何值对其进行初始化 - 因此您看到的是分配时内存中的内容。

关于张量及其维数和术语的简要说明：

- 有时您会看到称为向量的一维张量。
- 同样，二维张量通常称为矩阵。
- 任何超过二维的东西通常都被称为张量。

通常，您需要使用某个值来初始化张量。常见情况是全零、全一或随机值， `torch` 模块为所有这些提供工厂方法：

```python
zeros = torch.zeros(2, 3)
print(zeros)

ones = torch.ones(2, 3)
print(ones)

torch.manual_seed(1729)
random = torch.rand(2, 3)
print(random)
```

```
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[0.3126, 0.3791, 0.3087],
        [0.0736, 0.4216, 0.0691]])
```

工厂方法都按照您的预期执行操作 - 我们有一个充满 0 的张量，另一个充满 1 的张量，另一个充满 0 到 1 之间的随机值的张量。

### Random Tensors and Seeding 

说到随机张量，您是否注意到紧随其前面的 `torch.manual_seed()` 调用？使用随机值初始化张量（例如模型的学习权重）很常见，但有时（尤其是在研究环境中）您需要确保结果的可重复性。手动设置随机数生成器的种子是实现此目的的方法。让我们更仔细地看看：

```python
torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)
```

```
tensor([[0.3126, 0.3791, 0.3087],
        [0.0736, 0.4216, 0.0691]])
tensor([[0.2332, 0.4047, 0.2162],
        [0.9927, 0.4128, 0.5938]])
tensor([[0.3126, 0.3791, 0.3087],
        [0.0736, 0.4216, 0.0691]])
tensor([[0.2332, 0.4047, 0.2162],
        [0.9927, 0.4128, 0.5938]])
```

您应该在上面看到的是 `random1` 和 `random3` 具有相同的值， `random2` 和 `random4` 也是如此。手动设置 RNG 的种子会重置它，因此在大多数设置中，取决于随机数的相同计算应该提供相同的结果。

有关更多信息，请参阅有关重现性的 PyTorch 文档。 [PyTorch documentation on reproducibility](https://pytorch.org/docs/stable/notes/randomness.html).

### Tensor Shapes 

通常，当您对两个或多个张量执行操作时，它们需要具有相同的形状 - 即具有相同的维度数以及每个维度中相同的单元数。为此，我们有 `torch.*_like()` 方法：

```python
x = torch.empty(2, 2, 3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)
```

```python
torch.Size([2, 2, 3])
tensor([[[9.4454e-02, 0.0000e+00, 1.4013e-45],
         [1.4013e-45, 2.8026e-45, 4.5902e-41]],

        [[0.0000e+00, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 4.5901e-41]]])
torch.Size([2, 2, 3])
tensor([[[ 2.7817e+07,  0.0000e+00,  1.9560e-03],
         [ 0.0000e+00,  1.8199e-37,  0.0000e+00]],

        [[ 0.0000e+00,  0.0000e+00,  4.7316e-03],
         [ 0.0000e+00, -8.1359e-04,  4.5901e-41]]])
torch.Size([2, 2, 3])
tensor([[[0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.]]])
torch.Size([2, 2, 3])
tensor([[[1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.]]])
torch.Size([2, 2, 3])
tensor([[[0.6128, 0.1519, 0.0453],
         [0.5035, 0.9978, 0.3884]],

        [[0.6929, 0.1703, 0.1384],
         [0.4759, 0.7481, 0.0361]]])
```

上面代码单元中的第一个新内容是在张量上使用 `.shape` 属性。此属性包含张量每个维度的范围的列表 - 在我们的示例中， `x` 是形状为 2 x 2 x 3 的三维张量。

下面，我们调用 `.empty_like()` 、 `.zeros_like()` 、 `.ones_like()` 和 `.rand_like()` 方法。使用 `.shape` 属性，我们可以验证这些方法中的每一个都返回具有相同维度和范围的张量。

创建将覆盖的张量的最后一种方法是直接从 PyTorch 集合指定其数据：

```python
some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print(some_constants)

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print(more_integers)
```

```
tensor([[3.1416, 2.7183],
        [1.6180, 0.0073]])
tensor([ 2,  3,  5,  7, 11, 13, 17, 19])
tensor([[2, 4, 6],
        [3, 6, 9]])
```

如果 Python 元组或列表中已有数据，则使用 `torch.tensor()` 是创建张量的最直接方法。如上所示，嵌套集合将产生多维张量。

> `torch.tensor()` 创建数据的副本。

### Tensor Data Types 

```python
a = torch.ones((2, 3), dtype=torch.int16)
print(a)

b = torch.rand((2, 3), dtype=torch.float64) * 20.
print(b)

c = b.to(torch.int32)
print(c)
```

```
tensor([[1, 1, 1],
        [1, 1, 1]], dtype=torch.int16)
tensor([[ 0.9956,  1.4148,  5.8364],
        [11.2406, 11.2083, 11.6692]], dtype=torch.float64)
tensor([[ 0,  1,  5],
        [11, 11, 11]], dtype=torch.int32)
```

设置张量基础数据类型的最简单方法是在创建时使用可选参数。在上面单元格的第一行中，我们为张量 `a` 设置 `dtype=torch.int16` 。当我们打印 `a` 时，我们可以看到它充满了 `1` 而不是 `1.` - Python 的微妙提示是这是一个整数类型而不是浮点数。

关于打印 `a` 需要注意的另一件事是，与我们将 `dtype` 保留为默认值（32 位浮点）不同，打印张量还指定其 `dtype` .

您可能还发现，我们从将张量的形状指定为一系列整数参数，到将这些参数分组到一个元组中。这并不是绝对必要的 - PyTorch 会将一系列初始的、未标记的整数参数作为张量形状 - 但在添加可选参数时，它可以使您的意图更具可读性。

设置数据类型的另一种方法是使用 `.to()` 方法。在上面的单元格中，我们以通常的方式创建一个随机浮点张量 `b` 。接下来，我们通过使用 `.to()` 方法将 `b` 转换为 32 位整数来创建 `c` 。请注意， `c` 包含与 `b` 相同的所有值，但被截断为整数。

可用的数据类型包括：

- `torch.bool`
- `torch.int8`
- `torch.uint8`
- `torch.int16`
- `torch.int32`
- `torch.int64`
- `torch.half`
- `torch.float`
- `torch.double`
- `torch.bfloat`

## Math & Logic with PyTorch Tensors 

现在您已经了解了创建张量的一些方法……您可以用它们做什么？

让我们首先看看基本算术，以及张量如何与简单标量交互：

```python
ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)
```

```
tensor([[1., 1.],
        [1., 1.]])
tensor([[2., 2.],
        [2., 2.]])
tensor([[3., 3.],
        [3., 3.]])
tensor([[4., 4.],
        [4., 4.]])
tensor([[1.4142, 1.4142],
        [1.4142, 1.4142]])
```

如上所示，张量和标量之间的算术运算（例如加法、减法、乘法、除法和求幂）分布在张量的每个元素上。由于此类操作的输出将是一个张量，因此您可以使用通常的运算符优先级规则将它们链接在一起，如我们创建 `threes` 的行中所示。

两个张量之间的类似操作也像您直观地期望的那样：

```python
powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

fives = ones + fours
print(fives)

dozens = threes * fours
print(dozens)
```

```
tensor([[ 2.,  4.],
        [ 8., 16.]])
tensor([[5., 5.],
        [5., 5.]])
tensor([[12., 12.],
        [12., 12.]])
```

这里需要注意的是，前面的代码单元中的所有张量都具有相同的形状。当我们尝试对形状不同的张量执行二元运算时会发生什么？

> 以下单元格抛出运行时错误。这是故意的。

```python
a = torch.rand(2, 3)
b = torch.rand(3, 2)

print(a * b)
```

在一般情况下，您不能以这种方式对不同形状的张量进行操作，即使在像上面的单元格这样的情况下，其中张量具有相同数量的元素。

### In Brief: Tensor Broadcasting 

> 如果您熟悉 NumPy ndarray 中的广播语义，您会发现此处适用相同的规则。

相同形状规则的例外是张量广播。这是一个例子：

```python
rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)
```

```
tensor([[0.6146, 0.5999, 0.5013, 0.9397],
        [0.8656, 0.5207, 0.6865, 0.3614]])
tensor([[1.2291, 1.1998, 1.0026, 1.8793],
        [1.7312, 1.0413, 1.3730, 0.7228]])
```

这里有什么技巧呢？我们如何将 2x4 张量乘以 1x4 张量？

广播是一种在形状相似的张量之间执行操作的方法。在上面的示例中，一行四列张量乘以两行四列张量的两行。

这是深度学习中的一个重要操作。常见的示例是将学习权重张量乘以一批输入张量，分别将运算应用于批次中的每个实例，并返回相同形状的张量 - 就像我们的 (2, 4) * (1, 4)上面的示例返回形状为 (2, 4) 的张量。

广播规则如下：

- 每个张量必须至少有一个维度 - 没有空张量。
- 比较两个张量的维度的大小，从最后到第一
  - 每个维度必须相等，或者
  - 其中一个维度必须为 1，或者
  - 该维度在张量之一中不存在

当然，正如您之前所见，形状相同的张量通常是“可广播的”。

以下是遵守上述规则并允许广播的一些情况示例：

```python
a =     torch.ones(4, 3, 2)

b = a * torch.rand(   3, 2) # 3rd & 2nd dims identical to a, dim 1 absent
print(b)

c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
print(c)

d = a * torch.rand(   1, 2) # 3rd dim identical to a, 2nd dim = 1
print(d)
```

```python
tensor([[[0.6493, 0.2633],
         [0.4762, 0.0548],
         [0.2024, 0.5731]],

        [[0.6493, 0.2633],
         [0.4762, 0.0548],
         [0.2024, 0.5731]],

        [[0.6493, 0.2633],
         [0.4762, 0.0548],
         [0.2024, 0.5731]],

        [[0.6493, 0.2633],
         [0.4762, 0.0548],
         [0.2024, 0.5731]]])
tensor([[[0.7191, 0.7191],
         [0.4067, 0.4067],
         [0.7301, 0.7301]],

        [[0.7191, 0.7191],
         [0.4067, 0.4067],
         [0.7301, 0.7301]],

        [[0.7191, 0.7191],
         [0.4067, 0.4067],
         [0.7301, 0.7301]],

        [[0.7191, 0.7191],
         [0.4067, 0.4067],
         [0.7301, 0.7301]]])
tensor([[[0.6276, 0.7357],
         [0.6276, 0.7357],
         [0.6276, 0.7357]],

        [[0.6276, 0.7357],
         [0.6276, 0.7357],
         [0.6276, 0.7357]],

        [[0.6276, 0.7357],
         [0.6276, 0.7357],
         [0.6276, 0.7357]],

        [[0.6276, 0.7357],
         [0.6276, 0.7357],
         [0.6276, 0.7357]]])
```

仔细观察上面每个张量的值：

- 创建 `b` 的乘法运算在 `a` 的每个“层”上广播。
- 对于 `c` ，该操作在 `a` 的每一层和每一行上广播 - 每个 3 元素列都是相同的。
- 对于 `d` ，我们将其切换 - 现在跨层和列的每一行都是相同的。

有关广播的更多信息，请参阅有关该主题的 PyTorch 文档。

以下是一些尝试广播失败的示例：

> 以下单元格抛出运行时错误。这是故意的。

```python
a =     torch.ones(4, 3, 2)

b = a * torch.rand(4, 3)    # dimensions must match last-to-first

c = a * torch.rand(   2, 3) # both 3rd & 2nd dims different

d = a * torch.rand((0, ))   # can't broadcast with an empty tensor
```

### More Math with Tensors 

PyTorch 张量有超过三百种可以对其执行的操作。

以下是一些主要操作类别的一个小样本：

```python
# common functions
a = torch.rand(2, 4) * 2 - 1
print('Common functions:')
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5))

# trigonometric functions and their inverses
angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
sines = torch.sin(angles)
inverses = torch.asin(sines)
print('\nSine and arcsine:')
print(angles)
print(sines)
print(inverses)

# bitwise operations
print('\nBitwise XOR:')
b = torch.tensor([1, 5, 11])
c = torch.tensor([2, 7, 10])
print(torch.bitwise_xor(b, c))

# comparisons:
print('\nBroadcasted, element-wise equality comparison:')
d = torch.tensor([[1., 2.], [3., 4.]])
e = torch.ones(1, 2)  # many comparison ops support broadcasting!
print(torch.eq(d, e)) # returns a tensor of type bool

# reductions:
print('\nReduction ops:')
print(torch.max(d))        # returns a single-element tensor
print(torch.max(d).item()) # extracts the value from the returned tensor
print(torch.mean(d))       # average
print(torch.std(d))        # standard deviation
print(torch.prod(d))       # product of all numbers
print(torch.unique(torch.tensor([1, 2, 1, 2, 1, 2]))) # filter unique elements

# vector and linear algebra operations
v1 = torch.tensor([1., 0., 0.])         # x unit vector
v2 = torch.tensor([0., 1., 0.])         # y unit vector
m1 = torch.rand(2, 2)                   # random matrix
m2 = torch.tensor([[3., 0.], [0., 3.]]) # three times identity matrix

print('\nVectors & Matrices:')
print(torch.cross(v2, v1)) # negative of z unit vector (v1 x v2 == -v2 x v1)
print(m1)
m3 = torch.matmul(m1, m2)
print(m3)                  # 3 times m1
print(torch.svd(m3))       # singular value decomposition
```

```
Common functions:
tensor([[0.9238, 0.5724, 0.0791, 0.2629],
        [0.1986, 0.4439, 0.6434, 0.4776]])
tensor([[-0., -0., 1., -0.],
        [-0., 1., 1., -0.]])
tensor([[-1., -1.,  0., -1.],
        [-1.,  0.,  0., -1.]])
tensor([[-0.5000, -0.5000,  0.0791, -0.2629],
        [-0.1986,  0.4439,  0.5000, -0.4776]])

Sine and arcsine:
tensor([0.0000, 0.7854, 1.5708, 2.3562])
tensor([0.0000, 0.7071, 1.0000, 0.7071])
tensor([0.0000, 0.7854, 1.5708, 0.7854])

Bitwise XOR:
tensor([3, 2, 1])

Broadcasted, element-wise equality comparison:
tensor([[ True, False],
        [False, False]])

Reduction ops:
tensor(4.)
4.0
tensor(2.5000)
tensor(1.2910)
tensor(24.)
tensor([1, 2])

Vectors & Matrices:
tensor([ 0.,  0., -1.])
tensor([[0.7375, 0.8328],
        [0.8444, 0.2941]])
tensor([[2.2125, 2.4985],
        [2.5332, 0.8822]])
torch.return_types.svd(
U=tensor([[-0.7889, -0.6145],
        [-0.6145,  0.7889]]),
S=tensor([4.1498, 1.0548]),
V=tensor([[-0.7957,  0.6056],
        [-0.6056, -0.7957]]))
```

这是操作的一个小样本。有关更多详细信息和数学函数的完整清单，请查看文档。

### Altering Tensors in Place 

大多数张量上的二元运算都会返回第三个新张量。当我们说 `c = a * b` （其中 `a` 和 `b` 是张量）时，新张量 `c` 将占据与之前的张量不同的内存区域。其他张量。

不过，有时您可能希望就地更改张量 - 例如，如果您正在进行逐元素计算，您可以丢弃中间值。为此，大多数数学函数都有一个带有附加下划线（ `_` ）的版本，它将改变张量。

```python
a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('a:')
print(a)
print(torch.sin(a))   # this operation creates a new tensor in memory
print(a)              # a has not changed

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('\nb:')
print(b)
print(torch.sin_(b))  # note the underscore
print(b)              # b has changed
```

```
a:
tensor([0.0000, 0.7854, 1.5708, 2.3562])
tensor([0.0000, 0.7071, 1.0000, 0.7071])
tensor([0.0000, 0.7854, 1.5708, 2.3562])

b:
tensor([0.0000, 0.7854, 1.5708, 2.3562])
tensor([0.0000, 0.7071, 1.0000, 0.7071])
tensor([0.0000, 0.7071, 1.0000, 0.7071])
```

对于算术运算，有一些行为类似的函数：

```python
a = torch.ones(2, 2)
b = torch.rand(2, 2)

print('Before:')
print(a)
print(b)
print('\nAfter adding:')
print(a.add_(b))
print(a)
print(b)
print('\nAfter multiplying')
print(b.mul_(b))
print(b)
```

请注意，这些就地算术函数是 `torch.Tensor` 对象上的方法，而不是像许多其他函数（例如 `torch.sin()` ）一样附加到 `torch` 模块。正如您从 `a.add_(b)` 中看到的，调用张量是就地更改的张量。

还有另一种选择可以将计算结果放入现有的分配张量中。到目前为止我们已经看到的许多方法和函数 - 包括创建方法！ - 有一个 `out` 参数，可让您指定一个张量来接收输出。如果 `out` 张量的形状正确且 `dtype` ，则无需新的内存分配即可发生这种情况：

```python
a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.zeros(2, 2)
old_id = id(c)

print(c)
d = torch.matmul(a, b, out=c)
print(c)                # contents of c have changed

assert c is d           # test c & d are same object, not just containing equal values
assert id(c) == old_id  # make sure that our new c is the same object as the old one

torch.rand(2, 2, out=c) # works for creation too!
print(c)                # c has changed again
assert id(c) == old_id  # still the same object!
```

```
tensor([[0., 0.],
        [0., 0.]])
tensor([[0.3653, 0.8699],
        [0.2364, 0.3604]])
tensor([[0.0776, 0.4004],
        [0.9877, 0.0352]])
```

## Copying Tensors

与 Python 中的任何对象一样，将张量分配给变量会使该变量成为张量的标签，并且不会复制它。例如：

```python
a = torch.ones(2, 2)
b = a

a[0][1] = 561  # we change a...
print(b)       # ...and b is also altered
```

```
tensor([[  1., 561.],
        [  1.,   1.]])
```

但是，如果您想要处理数据的单独副本怎么办？ `clone()` 方法适合您：

```python
a = torch.ones(2, 2)
b = a.clone()

assert b is not a      # different objects in memory...
print(torch.eq(a, b))  # ...but still with the same contents!

a[0][1] = 561          # a changes...
print(b)               # ...but b is still all ones
```

```
tensor([[True, True],
        [True, True]])
tensor([[1., 1.],
        [1., 1.]])
```

使用“clone()”时需要注意一件重要的事情。如果您的源张量启用了 autograd，那么克隆张量也将启用。这将在 autograd 的视频中更深入地介绍，但如果您想要详细信息的简单版本，请继续。

在许多情况下，这就是您想要的。例如，如果您的模型在其 `forward()` 方法中具有多个计算路径，并且原始张量及其克隆都对模型的输出有贡献，那么为了启用模型学习，您需要为两个张量打开 autograd。如果您的源张量启用了自动梯度（如果它是一组学习权重或从涉及权重的计算中派生的，通常会启用自动梯度），那么您将得到您想要的结果。

另一方面，如果您正在进行计算，其中原始张量及其克隆都不需要跟踪梯度，那么只要源张量关闭了 autograd，您就可以开始了。

不过，还有第三种情况：假设您正在模型的 `forward()` 函数中执行计算，其中默认情况下为所有内容打开渐变，但您想在中途提取一些值来生成一些指标。在这种情况下，您不希望源张量的克隆副本跟踪梯度 - 通过关闭 autograd 的历史记录跟踪可以提高性能。为此，您可以在源张量上使用 `.detach()` 方法：

```python
a = torch.rand(2, 2, requires_grad=True) # turn on autograd
print(a)

b = a.clone()
print(b)

c = a.detach().clone()
print(c)

print(a)
```

```
tensor([[0.0905, 0.4485],
        [0.8740, 0.2526]], requires_grad=True)
tensor([[0.0905, 0.4485],
        [0.8740, 0.2526]], grad_fn=<CloneBackward0>)
tensor([[0.0905, 0.4485],
        [0.8740, 0.2526]])
tensor([[0.0905, 0.4485],
        [0.8740, 0.2526]], requires_grad=True)
```

- 我们创建 `a` 并打开 `requires_grad=True` 。我们还没有讨论这个可选参数，但会在 autograd 单元中讨论。
- 当我们打印 `a` 时，它通知我们属性 `requires_grad=True` - 这意味着自动分级和计算历史跟踪已打开。
- 我们克隆 `a` 并将其标记为 `b` 。当我们打印 `b` 时，我们可以看到它正在跟踪其计算历史记录 - 它继承了 `a` 的 autograd 设置，并添加到计算历史记录中。
- 我们克隆了a到c，但是我们调用了a的detach方法后才调用clone
- 打印 `c` ，我们没有看到计算历史，也没有 `requires_grad=True` 。

`detach()` 方法将张量从其计算历史中分离出来。它说，“做接下来的任何事情，就好像自动分级已关闭一样。”它在不更改 `a` 的情况下执行此操作 - 您可以看到，当我们在末尾再次打印 `a` 时，它保留了其 `requires_grad=True` 属性。

## Moving to GPU 

PyTorch的主要优势之一是它在兼容CUDA的Nvidia GPU上具有强大的加速能力。（“CUDA”代表计算统一设备架构，这是英伟达的并行计算平台。）到目前为止，我们所做的一切都在CPU上。我们如何转向更快的硬件？

首先，我们应该使用 `is_available()` 方法检查 GPU 是否可用。

> 如果您没有安装 CUDA 兼容的 GPU 和 CUDA 驱动程序，本节中的可执行单元将不会执行任何与 GPU 相关的代码。

```python
if torch.cuda.is_available():
    print('We have a GPU!')
else:
    print('Sorry, CPU only.')
```

```
We have a GPU!
```

一旦我们确定一个或多个 GPU 可用，我们就需要将数据放在 GPU 可以看到的地方。您的 CPU 对计算机 RAM 中的数据进行计算。您的 GPU 附有专用内存。每当您想要在设备上执行计算时，您必须将该计算所需的所有数据移动到该设备可访问的内存中。 （通俗地说，“将数据移至 GPU 可访问的内存”缩写为“将数据移至 GPU”。）

有多种方法可以将数据传输到目标设备上。您可以在创建时执行此操作：

```python
if torch.cuda.is_available():
    gpu_rand = torch.rand(2, 2, device='cuda')
    print(gpu_rand)
else:
    print('Sorry, CPU only.')
```

```python
tensor([[0.3344, 0.2640],
        [0.2119, 0.0582]], device='cuda:0')
```

默认情况下，新的张量是在CPU上创建的，因此我们必须使用可选的设备参数指定何时在GPU上创建张量。你可以看到，当我们打印新的张量时，PyTorch会通知我们它在哪个设备上（如果它不在CPU上）。

您可以通过 `torch.cuda.device_count()` 查询 GPU 数量。如果您有多个 GPU，您可以通过索引指定它们： `device='cuda:0'` 、 `device='cuda:1'` 等。

作为一种编码实践，用字符串常量指定我们的设备是非常脆弱的。在理想的情况下，无论您是在 CPU 还是 GPU 硬件上，您的代码都会稳定地执行。您可以通过创建一个可以传递给张量而不是字符串的设备句柄来做到这一点：

```python
if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

x = torch.rand(2, 2, device=my_device)
print(x)
```

```python
Device: cuda
tensor([[0.0024, 0.6778],
        [0.2441, 0.6812]], device='cuda:0')
```

如果一台设备上有一个现有张量，则可以使用 `to()` 方法将其移动到另一台设备。以下代码行在 CPU 上创建一个张量，并将其移动到您在上一个单元中获取的设备句柄。

```python
y = torch.rand(2, 2)
y = y.to(my_device)
```

重要的是要知道，为了进行涉及两个或多个张量的计算，所有张量必须位于同一设备上。无论您是否有可用的 GPU 设备，以下代码都会引发运行时错误：

```python
x = torch.rand(2, 2)
y = torch.rand(2, 2, device='gpu')
z = x + y  # exception will be thrown
```

## Manipulating Tensor Shapes 

有时，您需要更改张量的形状。下面，我们将讨论一些常见情况以及如何处理它们。

### Changing the Number of Dimensions 

您可能需要更改维度数的一种情况是将单个输入实例传递给模型。 PyTorch 模型通常需要批量输入。

例如，假设有一个模型适用于 3 x 226 x 226 图像 - 具有 3 个颜色通道的 226 像素正方形。当您加载并转换它时，您将获得形状 `(3, 226, 226)` 的张量。不过，您的模型需要输入形状 `(N, 3, 226, 226)` ，其中 `N` 是批次中的图像数量。那么如何制作一批呢？

```python
a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)

print(a.shape)
print(b.shape)
```

`unsqueeze()` 方法添加范围为 1 的维度。 `unsqueeze(0)` 将其添加为新的第 0 维 - 现在您拥有一批 1 维！

那么如果这不挤压呢？我们所说的挤压是什么意思？我们利用了这样一个事实：范围为 1 的任何维度都不会改变张量中的元素数量。

```python
c = torch.rand(1, 1, 1, 1, 1)
print(c)
```

```
tensor([[[[[0.2347]]]]])
```

继续上面的示例，假设模型的输出是每个输入的 20 元素向量。然后，您会期望输出具有形状 `(N, 20)` ，其中 `N` 是输入批次中的实例数。这意味着对于我们的单输入批次，我们将获得形状 `(1, 20)` 的输出。

如果您想使用该输出进行一些非批量计算（只需要 20 个元素向量）怎么办？

```python
a = torch.rand(1, 20)
print(a.shape)
print(a)

b = a.squeeze(0)
print(b.shape)
print(b)

c = torch.rand(2, 2)
print(c.shape)

d = c.squeeze(0)
print(d.shape)
```

```
torch.Size([1, 20])
tensor([[0.1899, 0.4067, 0.1519, 0.1506, 0.9585, 0.7756, 0.8973, 0.4929, 0.2367,
         0.8194, 0.4509, 0.2690, 0.8381, 0.8207, 0.6818, 0.5057, 0.9335, 0.9769,
         0.2792, 0.3277]])
torch.Size([20])
tensor([0.1899, 0.4067, 0.1519, 0.1506, 0.9585, 0.7756, 0.8973, 0.4929, 0.2367,
        0.8194, 0.4509, 0.2690, 0.8381, 0.8207, 0.6818, 0.5057, 0.9335, 0.9769,
        0.2792, 0.3277])
torch.Size([2, 2])
torch.Size([2, 2])
```

您可以从形状中看到我们的二维张量现在是一维的，如果您仔细观察上面单元格的输出，您会发现打印 `a` 显示了一组“额外”的方括号 `[]` 由于有额外的维度。

您只能 `squeeze()` 范围为 1 的维度。请参阅上面我们尝试在 `c` 中压缩大小为 2 的维度，并返回与我们开始时相同的形状。对 `squeeze()` 和 `unsqueeze()` 的调用只能作用于范围 1 的维度，因为否则会改变张量中的元素数量。

您可能使用 `unsqueeze()` 的另一个地方是简化广播。回想一下上面的例子，我们有以下代码：

```python
a =     torch.ones(4, 3, 2)

c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
print(c)
```



其最终效果是在维度 0 和 2 上广播操作，导致随机 3 x 1 张量按元素乘以 `a` 中的每个 3 元素列。

如果随机向量只是三元素向量怎么办？我们将失去进行广播的能力，因为最终尺寸将不符合广播规则。 `unsqueeze()` 来救援：

```python
a = torch.ones(4, 3, 2)
b = torch.rand(   3)     # trying to multiply a * b will give a runtime error
c = b.unsqueeze(1)       # change to a 2-dimensional tensor, adding new dim at the end
print(c.shape)
print(a * c)             # broadcasting works again!
```

```python
torch.Size([3, 1])
tensor([[[0.1891, 0.1891],
         [0.3952, 0.3952],
         [0.9176, 0.9176]],

        [[0.1891, 0.1891],
         [0.3952, 0.3952],
         [0.9176, 0.9176]],

        [[0.1891, 0.1891],
         [0.3952, 0.3952],
         [0.9176, 0.9176]],

        [[0.1891, 0.1891],
         [0.3952, 0.3952],
         [0.9176, 0.9176]]])
```

`squeeze()` 和 `unsqueeze()` 方法也有就地版本 `squeeze_()` 和 `unsqueeze_()` ：

```python
batch_me = torch.rand(3, 226, 226)
print(batch_me.shape)
batch_me.unsqueeze_(0)
print(batch_me.shape)
```

```
torch.Size([3, 226, 226])
torch.Size([1, 3, 226, 226])
```

有时您会想要更彻底地改变张量的形状，同时仍然保留元素的数量及其内容。发生这种情况的一种情况是在模型的卷积层和模型的线性层之间的接口处 - 这在图像分类模型中很常见。卷积核将产生形状特征 x 宽度 x 高度的输出张量，但下面的线性层需要一维输入。 `reshape()` 将为您执行此操作，前提是您请求的维度产生与输入张量相同数量的元素：

```python
output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape(6 * 20 * 20)
print(input1d.shape)

# can also call it as a method on the torch module:
print(torch.reshape(output3d, (6 * 20 * 20,)).shape)
```

```
torch.Size([6, 20, 20])
torch.Size([2400])
torch.Size([2400])
```

> 上面单元格最后一行中的 `(6 * 20 * 20,)` 参数是因为 PyTorch 在指定张量形状时需要一个元组 - 但是当形状是方法的第一个参数时，它让我们作弊并只使用一个系列整数。在这里，我们必须添加括号和逗号来让方法相信这确实是一个单元素元组。

如果可以， `reshape()` 将返回要更改的张量的视图 - 即查看同一底层内存区域的单独张量对象。这很重要：这意味着对源张量所做的任何更改都将反映在该张量的视图中，除非您 `clone()` 它。

在某些情况下， `reshape()` 必须返回携带数据副本的张量，这超出了本介绍的范围。有关更多信息，请参阅文档。

## NumPy Bridge

在上面关于广播的部分中，提到 PyTorch 的广播语义与 NumPy 兼容 - 但 PyTorch 和 NumPy 之间的亲缘关系比这更深。

如果您现有的 ML 或科学代码的数据存储在 NumPy ndarray 中，您可能希望将相同的数据表示为 PyTorch 张量，无论是利用 PyTorch 的 GPU 加速还是利用其构建 ML 模型的高效抽象。在 ndarrays 和 PyTorch 张量之间切换很容易：

```python
import numpy as np

numpy_array = np.ones((2, 3))
print(numpy_array)

pytorch_tensor = torch.from_numpy(numpy_array)
print(pytorch_tensor)
```

PyTorch 创建一个与 NumPy 数组形状相同并包含相同数据的张量，甚至保留 NumPy 的默认 64 位浮点数据类型。

转换也可以很容易地以另一种方式进行：

```python
pytorch_rand = torch.rand(2, 3)
print(pytorch_rand)

numpy_rand = pytorch_rand.numpy()
print(numpy_rand)
```

```
tensor([[0.8716, 0.2459, 0.3499],
        [0.2853, 0.9091, 0.5695]])
[[0.87163675 0.2458961  0.34993553]
 [0.2853077  0.90905803 0.5695162 ]]
```

重要的是要知道这些转换后的对象使用与其源对象相同的底层内存，这意味着对一个对象的更改会反映在另一个对象中：

```
numpy_array[1, 1] = 23
print(pytorch_tensor)

pytorch_rand[1, 1] = 17
print(numpy_rand)
```



## 总结

- 如何创建张量，随机数种子，张量的类型
- 张量的计算，张量计算的限制
  - 张量的广播算法，为什么需要，限制条件是什么
- 张量的原地运算符
- 如何复制张量，张量的引用和复制，张量复制的自动求导问题。
- 如何使用GPU加速，优雅健壮的GPU代码书写方式，多GPU如何使用，不同GPU之间张量不可运算。
- 如何拓展张量维度，如何减小张量维度，限制是什么？如何通过改变张量让张量传播算法合法。
- reshape作用和使用场景
- pytorch和numpy之间的关系，cpu上数据共通？如何互相转化。
