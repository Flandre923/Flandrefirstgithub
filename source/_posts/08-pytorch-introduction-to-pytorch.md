---
title: 08-pytorch-introduction to pytorch
date: 2023-11-13 10:33:31
tags:
- pytorch
- 深度学习
cover: https://view.moezx.cc/images/2020/05/07/77616185_p0.jpg
---

## PyTorch Tensors 

首先，我们将导入 pytorch。

```python
import torch
```

让我们看看一些基本的张量操作。首先，介绍几种创建张量的方法：

```python
z = torch.zeros(5, 3)
print(z)
print(z.dtype)
```

```
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
torch.float32
```

上面，我们创建了一个用零填充的 5x3 矩阵，并查询其数据类型，发现零是 32 位浮点数，这是默认的。

如果您想要整数怎么办？您始终可以覆盖默认值：

```python
i = torch.ones((5, 3), dtype=torch.int16)
print(i)
```

您可以看到，当我们更改默认值时，张量会在打印时有用地报告这一点。

随机初始化学习权重是很常见的，通常使用 PRNG 的特定种子来实现结果的可重复性：

```python
torch.manual_seed(1729)
r1 = torch.rand(2, 2)
print('A random tensor:')
print(r1)

r2 = torch.rand(2, 2)
print('\nA different random tensor:')
print(r2) # new values

torch.manual_seed(1729)
r3 = torch.rand(2, 2)
print('\nShould match r1:')
print(r3) # repeats values of r1 because of re-seed
```

```python
A random tensor:
tensor([[0.3126, 0.3791],
        [0.3087, 0.0736]])

A different random tensor:
tensor([[0.4216, 0.0691],
        [0.2332, 0.4047]])

Should match r1:
tensor([[0.3126, 0.3791],
        [0.3087, 0.0736]])
```



PyTorch 张量直观地执行算术运算。相似形状的张量可以相加、相乘等。标量的运算分布在张量上：

```python
ones = torch.ones(2, 3)
print(ones)

twos = torch.ones(2, 3) * 2 # every element is multiplied by 2
print(twos)

threes = ones + twos       # addition allowed because shapes are similar
print(threes)              # tensors are added element-wise
print(threes.shape)        # this has the same dimensions as input tensors

r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)
# uncomment this line to get a runtime error
# r3 = r1 + r2
```

```
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[2., 2., 2.],
        [2., 2., 2.]])
tensor([[3., 3., 3.],
        [3., 3., 3.]])
torch.Size([2, 3])
```

以下是可用数学运算的一小部分示例：

```python
r = (torch.rand(2, 2) - 0.5) * 2 # values between -1 and 1
print('A random matrix, r:')
print(r)

# Common mathematical operations are supported:
print('\nAbsolute value of r:')# 绝对值
print(torch.abs(r))

# ...as are trigonometric functions:
print('\nInverse sine of r:')# sin计算
print(torch.asin(r))

# ...and linear algebra operations like determinant and singular value decomposition
print('\nDeterminant of r:')# 行列式
print(torch.det(r))
print('\nSingular value decomposition of r:')
print(torch.svd(r)) # 矩阵的奇异值分解

# ...and statistical and aggregate operations:
print('\nAverage and standard deviation of r:')
print(torch.std_mean(r)) # 平均值和标准差
print('\nMaximum value of r:')
print(torch.max(r)) # 最大值
```

```
A random matrix, r:
tensor([[ 0.9956, -0.2232],
        [ 0.3858, -0.6593]])

Absolute value of r:
tensor([[0.9956, 0.2232],
        [0.3858, 0.6593]])

Inverse sine of r:
tensor([[ 1.4775, -0.2251],
        [ 0.3961, -0.7199]])

Determinant of r:
tensor(-0.5703)

Singular value decomposition of r:
torch.return_types.svd(
U=tensor([[-0.8353, -0.5497],
        [-0.5497,  0.8353]]),
S=tensor([1.1793, 0.4836]),
V=tensor([[-0.8851, -0.4654],
        [ 0.4654, -0.8851]]))

Average and standard deviation of r:
(tensor(0.7217), tensor(0.1247))

Maximum value of r:
tensor(0.9956)
```

## PyTorch Models

我们来谈谈如何在 PyTorch 中表达模型

```python
import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function
```

![image-20231113104802766](https://s2.loli.net/2023/11/13/yZ8xSDdnarG9upw.png)



上图是 LeNet-5 的图，它是最早的卷积神经网络之一，也是深度学习爆炸式增长的驱动因素之一。它旨在读取手写数字的小图像（MNIST 数据集），并正确分类图像中表示的数字。

以下是其工作原理的精简版：

- C1 层是一个卷积层，这意味着它会扫描输入图像以查找在训练期**间学到的特征。**它输出一张地图，显示它在图像中看到的每个学习特征的位置。该“激活图”在 S2 层中进行下采样。
- C3 层是另一个卷积层，这次扫描 C1 的激活图**以查找特征组合**。它还提供了一个描述这些特征组合的空间位置的激活图，该激活图在 S4 层中进行下采样。
- 最后，最后的全连接层 F5、F6 和 OUTPUT 是一个分类器，它采用最终的激活图，并将其分类为代表 10 个数字的 10 个容器之一。

我们如何用代码表达这个简单的神经网络？

```python
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```



这演示了典型 PyTorch 模型的结构：

- 它继承自 `torch.nn.Module` - 模块可以嵌套 - 事实上，甚至 `Conv2d` 和 `Linear` 层类也继承自 `torch.nn.Module` 。
- 模型将具有 `__init__()` 函数，在其中实例化其层，并加载它可能需要的任何数据工件（例如，NLP 模型可能加载词汇表）。
- 模型将具有 `forward()` 函数。这是实际计算发生的地方：输入通过网络层和各种函数传递以生成输出。
- 除此之外，您可以像任何其他 Python 类一样构建模型类，添加支持模型计算所需的任何属性和方法。

让我们实例化该对象并通过它运行示例输入。

```python
net = LeNet()
print(net)                         # what does the object tell us about itself?

input = torch.rand(1, 1, 32, 32)   # stand-in for a 32x32 black & white image
print('\nImage batch shape:')
print(input.shape)

output = net(input)                # we don't call forward() directly
print('\nRaw output:')
print(output)
print(output.shape)
```

```
LeNet(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)

Image batch shape:
torch.Size([1, 1, 32, 32])

Raw output:
tensor([[ 0.0898,  0.0318,  0.1485,  0.0301, -0.0085, -0.1135, -0.0296,  0.0164,
          0.0039,  0.0616]], grad_fn=<AddmmBackward0>)
torch.Size([1, 10])
```

上面发生了一些重要的事情：

首先，我们实例化 `LeNet` 类，然后打印 `net` 对象。 `torch.nn.Module` 的子类将报告它创建的图层及其形状和参数。如果您想了解模型处理的要点，这可以提供模型的便捷概述。

下面，我们创建一个虚拟输入，表示具有 1 个颜色通道的 32x32 图像。通常，您会加载图像图块并将其转换为这种形状的张量。

您可能已经注意到我们的张量有一个额外的维度 - 批量维度。 PyTorch 模型假设它们正在处理批量数据 - 例如，一批 16 个图像图块的形状为 `(16, 1, 32, 32)` 。由于我们只使用一张图像，因此我们创建了一批形状为 `(1, 1, 32, 32)` 的 1 图像。

我们通过像函数一样调用模型来请求模型进行推理： `net(input)` 。此调用的输出代表模型对输入代表特定数字的置信度。 （由于模型的这个实例还没有学到任何东西，所以我们不应该期望在输出中看到任何信号。）查看 `output` 的形状，我们可以看到它还有一个批次维度，其大小应始终与输入批次维度匹配。如果我们传入 16 个实例的输入批次， `output` 将具有 `(16, 10)` 的形状。

## Datasets and Dataloaders 

下面，我们将演示如何使用 TorchVision 中可供下载的开放访问数据集之一、如何转换图像以供模型使用，以及如何使用 DataLoader 将批量数据提供给模型。

我们需要做的第一件事是将传入的图像转换为 PyTorch 张量。

```python
#%matplotlib inline

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
```

在这里，我们为输入指定两种转换：

- `transforms.ToTensor()` 将 Pillow 加载的图像转换为 PyTorch 张量。
- `transforms.Normalize()` 调整张量的值，使其平均值为零，标准差为 1.0。大多数激活函数在 x = 0 附近有最强的梯度，因此将数据集中在那里可以加快学习速度。传递给变换的值是数据集中图像的 rgb 值的平均值（第一个元组）和标准差（第二个元组）。您可以通过运行以下几行代码自行计算这些值：

还有更多可用的变换，包括裁剪、居中、旋转和反射。

接下来，我们将创建 CIFAR10 数据集的实例。这是一组 32x32 彩色图像图块，代表 10 类物体：6 种动物（鸟、猫、鹿、狗、青蛙、马）和 4 种车辆（飞机、汽车、轮船、卡车）：

```python
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
```

> When you run the cell above, it may take a little time for the dataset to download.

这是在 PyTorch 中创建数据集对象的示例。可下载的数据集（如上面的 CIFAR-10）是 `torch.utils.data.Dataset` 的子类。 PyTorch 中的 `Dataset` 类包括 TorchVision、Torchtext 和 TorchAudio 中的可下载数据集，以及实用数据集类，例如 `torchvision.datasets.ImageFolder` ，它将读取标记图像的文件夹。您还可以创建自己的 `Dataset` 子类。

当我们实例化数据集时，我们需要告诉它一些事情：

- 我们想要数据存放的文件系统路径。
- 我们是否使用这套数据集进行训练；大多数数据集将分为训练和测试子集。
- 如果我们还没有下载数据集，我们是否愿意下载。
- 我们想要应用于数据的转换。

数据集准备好后，您可以将其提供给 `DataLoader` ：

```python
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
```

`Dataset` 子类包装对数据的访问，并专门针对其所服务的数据类型。 `DataLoader` 对数据一无所知，但会使用您指定的参数将 `Dataset` 提供的输入张量组织成批次。

在上面的示例中，我们要求 `DataLoader` 为我们提供来自 `trainset` 的 4 个图像批次，随机化它们的顺序 ( `shuffle=True` )，然后我们告诉它启动两个工作进程以从磁盘加载数据。

最好的做法是可视化 `DataLoader` 服务的批次：

```python
import matplotlib.pyplot as plt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

```
ship   car horse  ship
```

运行上面的单元格应该会向您显示一条由四个图像组成的条带，以及每个图像的正确标签。

## Training Your PyTorch Model 

让我们将所有部分放在一起，并训练一个模型：

```python
#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

首先，我们需要训练和测试数据集。如果尚未下载，请运行下面的单元格以确保数据集已下载。 （可能需要一分钟。）

```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

我们将对 `DataLoader` 的输出进行检查：

```python
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

这是我们要训练的模型。如果它看起来很熟悉，那是因为它是 LeNet 的一个变体（在本视频前面讨论过），适用于 3 色图像。

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
```

我们需要的最后一个成分是损失函数和优化器：

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

正如本视频前面所讨论的，损失函数是衡量模型预测与理想输出的距离的指标。交叉熵损失是像我们这样的分类模型的典型损失函数。

优化器是驱动学习的动力。在这里，我们创建了一个实现随机梯度下降的优化器，这是更简单的优化算法之一。除了算法的参数，如学习率 ( `lr` ) 和动量，我们还传入 `net.parameters()` ，它是模型中所有学习权重的集合 - 这就是优化器进行调整。

最后，所有这些都被组装到训练循环中。继续运行此单元，因为执行可能需要几分钟：

```python
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

在这里，我们只进行 2 个训练周期（第 1 行）——即对训练数据集进行两次遍历。每个通道都有一个内部循环，用于迭代训练数据（第 4 行），提供批量转换后的输入图像及其正确标签。

将梯度归零（第 9 行）是重要的一步。梯度在一批中累积；如果我们不为每个批次重置它们，它们将不断累积，这将提供不正确的梯度值，使学习变得不可能。

在第 12 行中，我们询问模型对此批次的预测。在下面的第 (13) 行中，我们计算损失 - `outputs` （模型预测）和 `labels` （正确输出）之间的差异。

在第 14 行中，我们执行 `backward()` 遍，并计算指导学习的梯度。

在第 15 行中，优化器执行一个学习步骤 - 它使用 `backward()` 调用中的梯度将学习权重推向它认为会减少损失的方向。

循环的其余部分对纪元数、已完成的训练实例数以及训练循环中收集的损失进行一些简单的报告。

当您运行上面的单元格时，您应该看到如下内容：

```python
[1,  2000] loss: 2.235
[1,  4000] loss: 1.940
[1,  6000] loss: 1.713
[1,  8000] loss: 1.573
[1, 10000] loss: 1.507
[1, 12000] loss: 1.442
[2,  2000] loss: 1.378
[2,  4000] loss: 1.364
[2,  6000] loss: 1.349
[2,  8000] loss: 1.319
[2, 10000] loss: 1.284
[2, 12000] loss: 1.267
Finished Training
```

请注意，损失是单调下降的，表明我们的模型正在继续提高其在训练数据集上的性能。

作为最后一步，我们应该检查模型是否确实在进行一般学习，而不是简单地“记住”数据集。这称为过度拟合，通常表明数据集太小（没有足够的示例用于一般学习），或者模型的学习参数多于正确建模数据集所需的参数。

这就是数据集被分为训练和测试子集的原因 - 为了测试模型的通用性，我们要求它对尚未训练的数据进行预测：

```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```

如果您继续跟进，您应该会发现该模型此时的准确率大约为 50%。这并不完全是最先进的，但它比我们期望的随机输出 10% 的准确度要好得多。这表明模型中确实发生了一些一般性学习。



## 总结

- tensor的创建，类型的修改，通过随机数种子固定初始随机数，tensor计算
- LeNet-5的图卷积层1，卷积层3的作用？如何搭建这个网络？
  - C1 层是一个卷积层，显示它在图像中看到的**每个学习特征的位置**
  - C3 层是另一个卷积层，个描述这些**特征组合**的空间位置的激活图
- 如何输入网络，如何获得网络输出。什么叫做推力。
- 如何加载模型，如何转化数据类型，使其可以供网络使用。
- 如何变化图像
- 如何可视化显示数据集，dataset，dataloader的配置和使用
- 如何训练网络，损失函数和优化器的作用，如何判断训练效果，如何判断是否是一般学习（区别过拟合情况）
- 过拟合可能出现的情况是？
  - 这称为过度拟合，通常表明数据集太小（没有足够的示例用于一般学习），或者模型的学习参数多于正确建模数据集所需的参数。
