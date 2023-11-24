---
title: 04_pytorch_build_the_neural_network
date: 2023-11-10 16:20:41
tags:
- 深度学习
- neural network
cover: https://w.wallhaven.cc/full/jx/wallhaven-jxlpem.png
---



# BUILD THE NEURAL NETWORK 

**神经网络的构成是什么？**

神经网络由对数据执行操作的层/模块组成。 

**层和模块在哪里？**

torch.nn 命名空间提供了构建您自己的神经网络所需的所有构建块。

 PyTorch 中的每个模块都是 nn.Module 的子类。

**神经网络的的嵌套构成**

神经网络本身就是一个模块，由其他模块（层）组成。

**嵌套结构的好处**

这种嵌套结构允许轻松构建和管理复杂的架构。



在以下部分中，我们将构建一个神经网络来对 FashionMNIST 数据集中的图像进行分类。

```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

## Get Device for Training 

如果我们在具有GPU的设备上，可以使用GPU加速。

**如何使用GPU？**

我们希望能够在 GPU 或 MPS 等硬件加速器（如果可用）上训练我们的模型。让我们检查一下  [torch.cuda](https://pytorch.org/docs/stable/notes/cuda.html) 或者 [torch.backends.mps](https://pytorch.org/docs/stable/notes/mps.html)是否可用，否则我们使用 CPU。

```python
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```

## Define the Class 

依照我们之前的说法，神经网络本身就是一个module，所以我们需要继承nn.Module。

我们通过子类化 `nn.Module` 来定义神经网络，并初始化 `__init__` 中的神经网络层。每个 `nn.Module` 子类都实现 `forward` 方法中对输入数据的操作。

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

我们创建 `NeuralNetwork` 的实例，并将其移动到 `device` ，并打印其结构。

```python
model = NeuralNetwork().to(device)
print(model)
```

为了使用该模型，我们将输入数据传递给它。这将执行模型的 `forward` 以及一些[background operations](https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866)。不要直接调用 `model.forward()` ！

在输入上调用模型会返回一个二维张量，其中 dim=0 对应于每个类的 10 个原始预测值的每个输出，dim=1 对应于每个输出的各个值。我们通过将预测概率传递给 `nn.Softmax` 模块的实例来获取预测概率。

```python 
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

## Model Layers

让我们分解 FashionMNIST 模型中的各个层。为了说明这一点，我们将采用 3 张大小为 28x28 的图像的小批量样本，看看当我们将其传递到网络时会发生什么。

```python
input_image = torch.rand(3,28,28)
print(input_image.size())
```

### nn.Flatten

我们初始化 [nn.Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html)  层，将每个 2D 28x28 图像转换为 784 个像素值的连续数组（维持小批量维度（在 dim=0 时））。

```python
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
```

```
torch.Size([3, 784])
```

### nn.Linear

 [linear layer](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)是一个使用其存储的权重和偏差对输入变量线性变换的模块。

```python
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```

### nn.ReLU 

非线性激活在模型的输入和输出之间创建复杂的映射。它们在线性变换后应用以引入非线性，帮助神经网络学习各种现象。

在此模型中，我们在线性层之间使用 nn.ReLU，但还有其他激活可以在模型中引入非线性。

```python
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```

```
Before ReLU: tensor([[ 0.4158, -0.0130, -0.1144,  0.3960,  0.1476, -0.0690, -0.0269,  0.2690,
          0.1353,  0.1975,  0.4484,  0.0753,  0.4455,  0.5321, -0.1692,  0.4504,
          0.2476, -0.1787, -0.2754,  0.2462],
        [ 0.2326,  0.0623, -0.2984,  0.2878,  0.2767, -0.5434, -0.5051,  0.4339,
          0.0302,  0.1634,  0.5649, -0.0055,  0.2025,  0.4473, -0.2333,  0.6611,
          0.1883, -0.1250,  0.0820,  0.2778],
        [ 0.3325,  0.2654,  0.1091,  0.0651,  0.3425, -0.3880, -0.0152,  0.2298,
          0.3872,  0.0342,  0.8503,  0.0937,  0.1796,  0.5007, -0.1897,  0.4030,
          0.1189, -0.3237,  0.2048,  0.4343]], grad_fn=<AddmmBackward0>)


After ReLU: tensor([[0.4158, 0.0000, 0.0000, 0.3960, 0.1476, 0.0000, 0.0000, 0.2690, 0.1353,
         0.1975, 0.4484, 0.0753, 0.4455, 0.5321, 0.0000, 0.4504, 0.2476, 0.0000,
         0.0000, 0.2462],
        [0.2326, 0.0623, 0.0000, 0.2878, 0.2767, 0.0000, 0.0000, 0.4339, 0.0302,
         0.1634, 0.5649, 0.0000, 0.2025, 0.4473, 0.0000, 0.6611, 0.1883, 0.0000,
         0.0820, 0.2778],
        [0.3325, 0.2654, 0.1091, 0.0651, 0.3425, 0.0000, 0.0000, 0.2298, 0.3872,
         0.0342, 0.8503, 0.0937, 0.1796, 0.5007, 0.0000, 0.4030, 0.1189, 0.0000,
         0.2048, 0.4343]], grad_fn=<ReluBackward0>)
```

### nn.Sequential

[nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)  是模块的有序容器。数据按照定义的相同顺序传递通过所有模块。您可以使用顺序容器来组合一个快速网络，例如 `seq_modules` .

```python
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

```

### nn.Softmax

神经网络的最后一个线性层返回 logits - [-infty, infty] 中的原始值 - 被传递到 nn.Softmax 模块。 Logits 缩放为值 [0, 1]，表示模型对每个类别的预测概率。 `dim` 参数指示维度，沿该维度值的总和必须为 1。

```python
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

## Model Parameters 

神经网络内的许多层都是参数化的，即具有在训练期间优化的相关权重和偏差。子类化 `nn.Module` 自动跟踪模型对象内定义的所有字段，并使所有参数可使用模型的 `parameters()` 或 `named_parameters()` 方法访问。

在此示例中，我们迭代每个参数，并打印其大小及其值的预览。

```python
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```

