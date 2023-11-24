---
title: 05_pytorch_AUTOMATIC_DIFFERENTIATION
date: 2023-11-10 16:54:29
tags:
- 深度学习
- 自动求导
cover: https://w.wallhaven.cc/full/d6/wallhaven-d6vj3m.png
---

# AUTOMATIC DIFFERENTIATION WITH `TORCH.AUTOGRAD`

在训练神经网络时，最常用的算法是反向传播。在该算法中，根据损失函数相对于给定参数的梯度来调整参数（模型权重）。

为了计算这些梯度，PyTorch 有一个名为 `torch.autograd` 的内置微分引擎。它支持任何计算图的梯度自动计算。

考虑最简单的单层神经网络，具有输入 `x` 、参数 `w` 和 `b` 以及一些损失函数。它可以通过以下方式在 PyTorch 中定义：

```python
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

## Tensors, Functions and Computational graph 

该代码定义了以下计算图：

![img](https://s2.loli.net/2023/11/10/sPfwv1i3QHr7kte.png)

在这个网络中， `w` 和 `b` 是我们需要优化的参数。因此，我们需要能够计算损失函数相对于这些变量的梯度。为此，我们设置这些张量的 `requires_grad` 属性。

> 您可以在创建张量时设置 `requires_grad` 的值，或者稍后使用 `x.requires_grad_(True)` 方法设置。

我们应用于张量来构造计算图的函数实际上是类 `Function` 的对象。该对象知道如何向前计算函数，以及如何在向后传播步骤中计算其导数。对反向传播函数的引用存储在张量的 `grad_fn` 属性中。您可以在[文档](https://pytorch.org/docs/stable/autograd.html#function)中找到 `Function` 的更多信息。



```python
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
```

## Computing Gradients 

为了优化神经网络中参数的权重，我们需要计算损失函数相对于参数的导数，即，我们需要在<的一些固定值下的 $$ \frac{loss}{w} 和 \frac{loss}{b} $$  b2> 和 `y` 。为了计算这些导数，我们调用 `loss.backward()` ，然后从 `w.grad` 和 `b.grad` 检索值：

```python
loss.backward()
print(w.grad)
print(b.grad)
```

```python
tensor([[0.3313, 0.0626, 0.2530],
        [0.3313, 0.0626, 0.2530],
        [0.3313, 0.0626, 0.2530],
        [0.3313, 0.0626, 0.2530],
        [0.3313, 0.0626, 0.2530]])
tensor([0.3313, 0.0626, 0.2530])
```

> - 我们只能获取计算图的叶节点的 `grad` 属性，其中 `requires_grad` 属性设置为 `True` 。对于我们图中的所有其他节点，梯度将不可用。
> - 出于性能原因，我们只能在给定图上使用 `backward` 执行一次梯度计算。如果我们需要在同一个图表上执行多个 `backward` 调用，则需要将 `retain_graph=True` 传递给 `backward` 调用。

## Disabling Gradient Tracking 

默认情况下，所有具有 `requires_grad=True` 的张量都会跟踪其计算历史并支持梯度计算。然而，在某些情况下，我们不需要这样做，例如，当我们训练了模型并且只想将其应用于某些输入数据时，即我们只想通过网络进行前向计算。我们可以通过用 `torch.no_grad()` 块包围我们的计算代码来停止跟踪计算：

```python
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
```

```python
True
False
```

获得相同结果的另一种方法是在张量上使用 `detach()` 方法：

```python
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
```

```
False
```

您可能想要禁用梯度跟踪的原因有：

- 将神经网络中的某些参数标记为冻结参数。
- 当您仅进行前向传递时加快计算速度，因为对不跟踪梯度的张量进行计算会更有效。

## More on Computational Graphs 

从概念上讲，autograd 在由 Function 对象组成的有向无环图 (DAG) 中保存数据（张量）和所有执行的操作（以及生成的新张量）的记录。在这个 DAG 中，叶子是输入张量，根是输出张量。通过从根到叶追踪该图，您可以使用链式法则自动计算梯度。

在前向传递中，autograd 同时执行两件事：

- 运行请求的操作来计算结果张量
- 在 DAG 中维护操作的梯度函数.

当在 DAG 根上调用 `.backward()` 时，向后传递开始。 `autograd` 然后：

- 计算每个 `.grad_fn` 的梯度，
- 将它们累积到相应张量的 `.grad` 属性中
- 使用链式法则，一直传播到叶张量。

DAG 在 PyTorch 中是动态的需要注意的重要一点是图是从头开始重新创建的；每次 `.backward()` 调用后，autograd 开始填充新图表。这正是允许您在模型中使用控制流语句的原因；如果需要，您可以在每次迭代时更改形状、大小和操作。
