---
title: 10-pytorch-the fundaments of autograd
date: 2023-11-14 15:31:09
tags:
- pytorch
- 深度学习
cover: https://view.moezx.cc/images/2019/01/30/70076841_p1_master1200.jpg
---

# THE FUNDAMENTALS OF AUTOGRAD 

PyTorch 的 Autograd 功能是 PyTorch 灵活快速地构建机器学习项目的一部分。它允许在复杂的计算中快速、轻松地计算多个偏导数（也称为梯度）。此操作是基于反向传播的神经网络学习的核心。

autograd 的强大之处在于它在运行时动态跟踪您的计算，这意味着如果您的模型有决策分支或循环，其长度直到运行时才知道，计算仍然会被正确跟踪，并且您将得到正确的结果梯度来驱动学习。再加上您的模型是用 Python 构建的，与依赖于更严格结构的模型的静态分析来计算梯度的框架相比，它提供了更大的灵活性。

## What Do We Need Autograd For? 

机器学习模型是一个具有输入和输出的函数。在本次讨论中，我们将输入视为一个维度向量 $$ \vec{x} $$，其中包含元素$$x_i$$。然后我们可以将模型 M 表示为输入的向量值函数：$$ \vec{y} = \vec{M}(\vec{x}) $$。 （我们将 M 的输出值视为向量，因为一般来说，模型可能有任意数量的输出。）

由于我们主要在训练的背景下讨论 autograd，因此我们感兴趣的输出将是模型的损失。损失函数 $$  L(\vec y ) = L( \vec M ( \vec x )) $$ 是模型输出的单值标量函数。该函数表示我们的模型的预测与特定输入的理想输出的差距有多大。注意：在此之后，我们通常会在上下文应该清晰的地方省略矢量符号 - 例如，  $$ y $$而不是$$\vec y$$ 。

在训练模型时，我们希望最小化损失。在完美模型的理想情况下，这意味着调整其学习权重 - 即函数的可调整参数 - 使得所有输入的损失为零。在现实世界中，这意味着一个不断调整学习权重的迭代过程，直到我们看到对于各种输入我们得到了可以容忍的损失。

我们如何决定轻推权重的距离和方向？我们希望最小化损失，这意味着使其相对于输入的一阶导数等于 0： $$\frac{\partial L }{\partial x} = 0$$

但请记住，损失不是直接从输入导出的，而是模型输出的函数（直接是输入的函数）， $$\frac{∂L}{∂x} = \frac{∂L(y))}{∂x}$$ 。根据微积分的链式法则，我们有 $$\frac{∂L(\vec y)}{∂x} = \frac{∂L}{∂y}\frac{∂y}{∂x}=\frac{∂L}{∂y}\frac{∂M(x)}{∂x}$$ 

 $$\frac{∂M(x)}{∂x}$$是事情变得复杂的地方。如果我们再次使用链式法则扩展表达式，模型输出相对于输入的偏导数将涉及模型中每个相乘的学习权重、每个激活函数以及每个其他数学变换的许多局部偏导数。每个此类偏导数的完整表达式是通过计算图的每个可能路径的局部梯度的乘积之和，该计算图以我们试图测量其梯度的变量结束。

特别是，我们对学习权重的梯度感兴趣——它们告诉我们改变每个权重的方向以使损失函数更接近于零。

由于此类局部导数（每个导数对应于模型计算图中的一条单独路径）的数量往往会随着神经网络的深度呈指数级增长，因此计算它们的复杂性也会随之增加。这就是 autograd 发挥作用的地方：它跟踪每次计算的历史记录。 PyTorch 模型中的每个计算张量都带有其输入张量和用于创建它的函数的历史记录。结合 PyTorch 函数旨在作用于张量的事实，每个函数都有一个用于计算自己的导数的内置实现，这大大加快了学习所需的局部导数的计算速度。

## A Simple Example 

这是很多理论 - 但在实践中使用 autograd 是什么样子呢？

让我们从一个简单的例子开始。首先，我们将进行一些导入以绘制结果：

```python
# %matplotlib inline

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
```

接下来，我们将创建一个在间隔 \[0,2*PI\]上充满均匀间隔值的输入张量，并指定 `requires_grad=True` 。 （与大多数创建张量的函数一样， `torch.linspace()` 接受可选的 `requires_grad` 选项。）设置此标志意味着在接下来的每次计算中，autograd将在该计算的输出张量。

接下来，我们将执行计算，并根据输入绘制其输出：

```python
b = torch.sin(a)
plt.plot(a.detach(), b.detach())
```

让我们仔细看看张量 `b` 。当我们打印它时，我们会看到一个指示符，表明它正在跟踪其计算历史记录：

```python
print(b)
```

```
tensor([ 0.0000e+00,  2.5882e-01,  5.0000e-01,  7.0711e-01,  8.6603e-01,
         9.6593e-01,  1.0000e+00,  9.6593e-01,  8.6603e-01,  7.0711e-01,
         5.0000e-01,  2.5882e-01, -8.7423e-08, -2.5882e-01, -5.0000e-01,
        -7.0711e-01, -8.6603e-01, -9.6593e-01, -1.0000e+00, -9.6593e-01,
        -8.6603e-01, -7.0711e-01, -5.0000e-01, -2.5882e-01,  1.7485e-07],
       grad_fn=<SinBackward0>)
```

这个 `grad_fn` 给了我们一个提示，当我们执行反向传播步骤并计算梯度时，我们需要计算所有该张量输入的 $$ sin(x)$$的导数。

让我们执行更多计算：

```python
c = 2 * b
print(c)

d = c + 1
print(d)
```

```
tensor([ 0.0000e+00,  5.1764e-01,  1.0000e+00,  1.4142e+00,  1.7321e+00,
         1.9319e+00,  2.0000e+00,  1.9319e+00,  1.7321e+00,  1.4142e+00,
         1.0000e+00,  5.1764e-01, -1.7485e-07, -5.1764e-01, -1.0000e+00,
        -1.4142e+00, -1.7321e+00, -1.9319e+00, -2.0000e+00, -1.9319e+00,
        -1.7321e+00, -1.4142e+00, -1.0000e+00, -5.1764e-01,  3.4969e-07],
       grad_fn=<MulBackward0>)
tensor([ 1.0000e+00,  1.5176e+00,  2.0000e+00,  2.4142e+00,  2.7321e+00,
         2.9319e+00,  3.0000e+00,  2.9319e+00,  2.7321e+00,  2.4142e+00,
         2.0000e+00,  1.5176e+00,  1.0000e+00,  4.8236e-01, -3.5763e-07,
        -4.1421e-01, -7.3205e-01, -9.3185e-01, -1.0000e+00, -9.3185e-01,
        -7.3205e-01, -4.1421e-01,  4.7684e-07,  4.8236e-01,  1.0000e+00],
       grad_fn=<AddBackward0>)
```

最后，让我们计算一个单元素输出。当您在不带参数的张量上调用 `.backward()` 时，它期望调用张量仅包含单个元素，就像计算损失函数时的情况一样。

```python
out = d.sum()
print(out)
```

tensor(25., grad_fn=<SumBackward0>)

使用张量存储的每个 `grad_fn` 都允许您使用其 `next_functions` 属性将计算一直返回到其输入。我们可以在下面看到，在 `d` 上深入研究这个属性向我们展示了所有先前张量的梯度函数。请注意， `a.grad_fn` 报告为 `None` ，表明这是函数的输入，没有自己的历史记录。

```python
print('d:')
print(d.grad_fn)
print(d.grad_fn.next_functions)
print(d.grad_fn.next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
print('\nc:')
print(c.grad_fn)
print('\nb:')
print(b.grad_fn)
print('\na:')
print(a.grad_fn)
```

```
d:
<AddBackward0 object at 0x7f334cdf7190>
((<MulBackward0 object at 0x7f334cdf5b40>, 0), (None, 0))
((<SinBackward0 object at 0x7f334cdf5b40>, 0), (None, 0))
((<AccumulateGrad object at 0x7f334cdf7190>, 0),)
()

c:
<MulBackward0 object at 0x7f334cdf5b40>

b:
<SinBackward0 object at 0x7f334cdf5b40>

a:
None
```

有了所有这些机制，我们如何推出衍生品呢？您在输出上调用 `backward()` 方法，并检查输入的 `grad` 属性以检查渐变：

```python
out.backward()
print(a.grad)
plt.plot(a.detach(), a.grad.detach())
```

```python
tensor([ 2.0000e+00,  1.9319e+00,  1.7321e+00,  1.4142e+00,  1.0000e+00,
         5.1764e-01, -8.7423e-08, -5.1764e-01, -1.0000e+00, -1.4142e+00,
        -1.7321e+00, -1.9319e+00, -2.0000e+00, -1.9319e+00, -1.7321e+00,
        -1.4142e+00, -1.0000e+00, -5.1764e-01,  2.3850e-08,  5.1764e-01,
         1.0000e+00,  1.4142e+00,  1.7321e+00,  1.9319e+00,  2.0000e+00])

[<matplotlib.lines.Line2D object at 0x7f334cdd0460>]
```

回想一下我们达到这里所采取的计算步骤：

```python
a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
b = torch.sin(a)
c = 2 * b
d = c + 1
out = d.sum()
```

添加一个常数，就像我们计算 `d` 一样，不会改变导数。剩下 $$c=2*b=2*sin(a)$$，它的导数应该是 $$2*cos(a)$$ 。看看上面的图表，这就是我们所看到的。

请注意，只有计算的叶节点才会计算其梯度。例如，如果您尝试 `print(c.grad)` 您会得到 `None` 。在这个简单的示例中，只有输入是叶节点，因此只有它计算了梯度。

## Autograd in Training

我们已经简要了解了 autograd 的工作原理，但是当它用于其预期目的时，它会是什么样子呢？让我们定义一个小模型并检查它在单个训练批次后如何变化。首先，定义一些常量、我们的模型以及输入和输出的一些替代：

```python
BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.layer1 = torch.nn.Linear(1000, 100)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

model = TinyModel()
```

您可能会注意到的一件事是，我们从未为模型的层指定 `requires_grad=True` 。在 `torch.nn.Module` 的子类中，假设我们想要跟踪层权重的梯度以进行学习。

如果我们查看模型的各层，我们可以检查权重的值，并验证尚未计算任何梯度：

```python
print(model.layer2.weight[0][0:10]) # just a small slice
print(model.layer2.weight.grad)
```

```
tensor([ 0.0920,  0.0916,  0.0121,  0.0083, -0.0055,  0.0367,  0.0221, -0.0276,
        -0.0086,  0.0157], grad_fn=<SliceBackward0>)
None
```



让我们看看当我们运行一批训练时，情况会发生什么变化。对于损失函数，我们将仅使用 `prediction` 和 `ideal_output` 之间的欧几里德距离的平方，并且我们将使用基本的随机梯度下降优化器。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

prediction = model(some_input)

loss = (ideal_output - prediction).pow(2).sum()
print(loss)
```

```
tensor(211.2634, grad_fn=<SumBackward0>)
```

现在，让我们调用 `loss.backward()` 看看会发生什么：

```python
loss.backward()
print(model.layer2.weight[0][0:10])
print(model.layer2.weight.grad[0][0:10])
```

```
tensor([ 0.0920,  0.0916,  0.0121,  0.0083, -0.0055,  0.0367,  0.0221, -0.0276,
        -0.0086,  0.0157], grad_fn=<SliceBackward0>)
tensor([12.8997,  2.9572,  2.3021,  1.8887,  5.0710,  7.3192,  3.5169,  2.4319,
         0.1732, -5.3835])
```

我们可以看到每个学习权重的梯度都已计算出来，但权重保持不变，因为我们还没有运行优化器。优化器负责根据计算的梯度更新模型权重。

```python
optimizer.step()
print(model.layer2.weight[0][0:10])
print(model.layer2.weight.grad[0][0:10])
```

```
tensor([ 0.0791,  0.0886,  0.0098,  0.0064, -0.0106,  0.0293,  0.0186, -0.0300,
        -0.0088,  0.0211], grad_fn=<SliceBackward0>)
tensor([12.8997,  2.9572,  2.3021,  1.8887,  5.0710,  7.3192,  3.5169,  2.4319,
         0.1732, -5.3835])
```

您应该看到 `layer2` 的权重已更改。

该过程中一件重要的事情是：调用 `optimizer.step()` 后，您需要调用 `optimizer.zero_grad()` ，否则每次运行 `loss.backward()` 时，学习权重的梯度都会积累：

```python
print(model.layer2.weight.grad[0][0:10])

for i in range(0, 5):
    prediction = model(some_input)
    loss = (ideal_output - prediction).pow(2).sum()
    loss.backward()

print(model.layer2.weight.grad[0][0:10])

optimizer.zero_grad(set_to_none=False)

print(model.layer2.weight.grad[0][0:10])
```

运行上面的单元格后，您应该看到多次运行 `loss.backward()` 后，大多数梯度的幅度都会大得多。在运行下一个训练批次之前未能将梯度归零将导致梯度以这种方式爆炸，从而导致不正确且不可预测的学习结果。

## Turning Autograd Off and On 

在某些情况下，您需要对是否启用自动分级进行细粒度控制。根据具体情况，有多种方法可以做到这一点。

最简单的方法是直接更改张量上的 `requires_grad` 标志：

```python
a = torch.ones(2, 3, requires_grad=True)
print(a)

b1 = 2 * a
print(b1)

a.requires_grad = False
b2 = 2 * a
print(b2)
```

```
tensor([[1., 1., 1.],
        [1., 1., 1.]], requires_grad=True)
tensor([[2., 2., 2.],
        [2., 2., 2.]], grad_fn=<MulBackward0>)
tensor([[2., 2., 2.],
        [2., 2., 2.]])
```

在上面的单元格中，我们看到 `b1` 有一个 `grad_fn` （即跟踪的计算历史），这正是我们所期望的，因为它是从张量 `a` ，已打开 autograd。当我们使用 `a.requires_grad = False` 显式关闭 autograd 时，将不再跟踪计算历史记录，正如我们在计算 `b2` 时看到的那样。

如果您只需要暂时关闭 autograd，更好的方法是使用 `torch.no_grad()` ：

```python
a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3

c1 = a + b
print(c1)

with torch.no_grad():
    c2 = a + b

print(c2)

c3 = a * b
print(c3)
```

```
tensor([[5., 5., 5.],
        [5., 5., 5.]], grad_fn=<AddBackward0>)
tensor([[5., 5., 5.],
        [5., 5., 5.]])
tensor([[6., 6., 6.],
        [6., 6., 6.]], grad_fn=<MulBackward0>)
```

`torch.no_grad()` 也可以用作函数或方法装饰器：

```python
def add_tensors1(x, y):
    return x + y

@torch.no_grad()
def add_tensors2(x, y):
    return x + y


a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3

c1 = add_tensors1(a, b)
print(c1)

c2 = add_tensors2(a, b)
print(c2)
```

```
tensor([[5., 5., 5.],
        [5., 5., 5.]], grad_fn=<AddBackward0>)
tensor([[5., 5., 5.],
        [5., 5., 5.]])
```

有一个相应的上下文管理器 `torch.enable_grad()` ，用于在尚未打开 autograd 时打开它。它也可以用作装饰器。

最后，您可能有一个需要梯度跟踪的张量，但您想要一个不需要梯度跟踪的副本。为此，我们有 `Tensor` 对象的 `detach()` 方法 - 它创建与计算历史分离的张量的副本：

```python
x = torch.rand(5, requires_grad=True)
y = x.detach()

print(x)
print(y)
```

```
tensor([0.0670, 0.3890, 0.7264, 0.3559, 0.6584], requires_grad=True)
tensor([0.0670, 0.3890, 0.7264, 0.3559, 0.6584])
```

当我们想要绘制一些张量的图表时，我们就这样做了。这是因为 `matplotlib` 期望 NumPy 数组作为输入，并且对于 require_grad=True 的张量，不会启用从 PyTorch 张量到 NumPy 数组的隐式转换。制作一份独立的副本可以让我们继续前进。

### Autograd and In-place Operations 

到目前为止，在本笔记本的每个示例中，我们都使用变量来捕获计算的中间值。 Autograd 需要这些中间值来执行梯度计算。因此，在使用 autograd 时必须小心使用就地操作。这样做可能会破坏在 `backward()` 调用中计算导数所需的信息。如果您尝试对需要自动分级的叶变量进行就地操作，PyTorch 甚至会阻止您，如下所示。

> 以下代码单元引发运行时错误。这是预料之中的。

```python
a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
torch.sin_(a)
```

## Autograd Profiler

Autograd 详细跟踪计算的每一步。这样的计算历史记录与计时信息相结合，将成为一个方便的分析器 - 并且 autograd 具有该功能。下面是一个快速示例用法：

```python
device = torch.device('cpu')
run_on_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda')
    run_on_gpu = True

x = torch.randn(2, 3, requires_grad=True)
y = torch.rand(2, 3, requires_grad=True)
z = torch.ones(2, 3, requires_grad=True)

with torch.autograd.profiler.profile(use_cuda=run_on_gpu) as prf:
    for _ in range(1000):
        z = (z / x) * y

print(prf.key_averages().table(sort_by='self_cpu_time_total'))
```

```
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                aten::div        50.96%       5.808ms        50.96%       5.808ms       5.808us      16.107ms        50.44%      16.107ms      16.107us          1000
                aten::mul        48.96%       5.581ms        48.96%       5.581ms       5.581us      15.827ms        49.56%      15.827ms      15.827us          1000
    cudaDeviceSynchronize         0.08%       9.000us         0.08%       9.000us       9.000us       0.000us         0.00%       0.000us       0.000us             1
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 11.398ms
Self CUDA time total: 31.934ms
```

分析器还可以标记各个代码子块，按输入张量形状分解数据，并将数据导出为 Chrome 跟踪工具文件。有关 API 的完整详细信息，请参阅文档。

## Advanced Topic: More Autograd Detail and the High-Level API 

如果您有一个具有 n 维输入和 m 维输出的函数 $$\vec{y}=f(\vec{x})$$，则完整梯度是每个输出相对于每个输入的导数的矩阵，称为雅可比行列式：
$$
J = \begin{pmatrix}
 \frac{∂y1}{∂x_1} & ... & \frac{∂y_1}{∂x_n} \\
 ... & ... & ...\\
 \frac{∂y_m}{∂x_n} & ...  & \frac{∂y_m}{∂x_n}
\end{pmatrix}
$$


如果您有第二个函数$$l=g(\vec{y})$$ ，它接受 m 维输入（即与上面的输出相同的维度），并返回标量输出，您可以表达其相对于 $$\vec{y}$$作为列向量， $$v=(\frac{∂l}{∂y1} ...  \frac{∂l}{∂y_m})$$ - 这实际上只是一个单列雅可比行列式。

更具体地说，将第一个函数想象为 PyTorch 模型（可能有多个输入和多个输出），第二个函数作为损失函数（模型的输出作为输入，损失值作为标量输出）。

如果我们将第一个函数的雅可比行列式乘以第二个函数的梯度，并应用链式法则，我们得到：

注意：您还可以使用等效操作 $$v^T * J$$，并返回行向量。

得到的列向量是第二个函数相对于第一个函数的输入的梯度，或者在我们的模型和损失函数的情况下，是损失相对于模型输入的梯度。

“torch.autograd”是计算这些产品的引擎。这就是我们在向后传递过程中累积学习权重梯度的方式。

因此， `backward()` 调用还可以采用可选的向量输入。该向量表示张量上的一组梯度，将其乘以其之前的自动梯度追踪张量的雅可比行列式。让我们尝试一个带有小向量的具体示例：

```python
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
```

```
tensor([  299.4868,   425.4009, -1082.9885], grad_fn=<MulBackward0>)
```

如果我们现在尝试调用 `y.backward()` ，我们会收到运行时错误和一条消息，**表明只能为标量输出隐式计算梯度**。对于多维输出，autograd 希望我们为这三个输出提供梯度，并将其乘以雅可比行列式：

```python
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float) # stand-in for gradients
y.backward(v)

print(x.grad)
```

```
tensor([1.0240e+02, 1.0240e+03, 1.0240e-01])
```



（请注意，输出梯度都与 2 的幂相关 - 这是我们从重复的倍增操作中所期望的。）

### The High-Level API 

autograd 上有一个 API，可让您直接访问重要的微分矩阵和向量运算。特别是，它允许您计算特定输入的特定函数的雅可比矩阵和海塞矩阵。 （Hessian 矩阵类似于雅可比矩阵，但表示所有偏二阶导数。）它还提供了使用这些矩阵求向量积的方法。



让我们采用一个简单函数的雅可比行列式，针对 2 个单元素输入进行计算：

```python
def exp_adder(x, y):
    return 2 * x.exp() + 3 * y

inputs = (torch.rand(1), torch.rand(1)) # arguments for the function
print(inputs)
torch.autograd.functional.jacobian(exp_adder, inputs)
```

```
(tensor([0.7212]), tensor([0.2079]))

(tensor([[4.1137]]), tensor([[3.]]))
```

如果仔细观察，第一个输出应等于 $$2e^x$$ （因为 $$e^x$$的导数是 $$e^x$$ ），第二个值应为 3。

当然，您可以使用高阶张量来做到这一点：

```python
inputs = (torch.rand(3), torch.rand(3)) # arguments for the function
print(inputs)
torch.autograd.functional.jacobian(exp_adder, inputs)
```

```
(tensor([0.2080, 0.2604, 0.4415]), tensor([0.5220, 0.9867, 0.4288]))

(tensor([[2.4623, 0.0000, 0.0000],
        [0.0000, 2.5950, 0.0000],
        [0.0000, 0.0000, 3.1102]]), tensor([[3., 0., 0.],
        [0., 3., 0.],
        [0., 0., 3.]]))
```

`torch.autograd.functional.hessian()` 方法的工作原理相同（假设您的函数是两次可微的），但返回所有二阶导数的矩阵。

如果您提供向量，还有一个函数可以直接计算向量雅可比积：

```python
def do_some_doubling(x):
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    return y

inputs = torch.randn(3)
my_gradients = torch.tensor([0.1, 1.0, 0.0001])
torch.autograd.functional.vjp(do_some_doubling, inputs, v=my_gradients)
```

`torch.autograd.functional.jvp()` 方法执行与 `vjp()` 相同的矩阵乘法，但操作数相反。 `vhp()` 和 `hvp()` 方法对向量 Hessian 乘积执行相同的操作。

有关更多信息，包括函数式 API 文档中的性能说明
