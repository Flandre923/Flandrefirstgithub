---
title: 13-pytorch-training with pytorch
date: 2023-11-15 11:52:17
tags:
- 深度学习
- pytorch
cover: https://view.moezx.cc/images/2017/11/25/_64600536_1.jpg
---

## Introduction

在过去的视频中，我们讨论并演示了：

- 使用 torch.nn 模块的神经网络层和函数构建模型
- 自动梯度计算的机制，这是基于梯度的模型训练的核心
- 使用 TensorBoard 可视化训练进度和其他活动

在本视频中，我们将向您的库存添加一些新工具：

- 我们将熟悉数据集和数据加载器抽象，以及它们如何简化在训练循环期间向模型提供数据的过程
- 我们将讨论特定的损失函数以及何时使用它们
- 我们将了解 PyTorch 优化器，它实现根据损失函数的结果调整模型权重的算法

最后，我们将把所有这些放在一起，并看到完整的 PyTorch 训练循环的实际运行。

## Dataset and DataLoader 

`Dataset` 和 `DataLoader` 类封装了从存储中提取数据并将其批量暴露给训练循环的过程。

`Dataset` 负责访问和处理单个数据实例。

`DataLoader` 从 `Dataset` 中提取数据实例（自动或使用您定义的采样器），批量收集它们，然后返回它们以供训练循环使用。 `DataLoader` 适用于所有类型的数据集，无论它们包含什么类型的数据。

在本教程中，我们将使用 TorchVision 提供的 Fashion-MNIST 数据集。我们使用 `torchvision.transforms.Normalize()` 对图像图块内容的分布进行零中心和标准化，并下载训练和验证数据分割。

```python
import torch
import torchvision
import torchvision.transforms as transforms

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# Class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))
```

与往常一样，让我们将数据可视化作为健全性检查：

```python
import matplotlib.pyplot as plt
import numpy as np

# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(training_loader)
images, labels = next(dataiter)

# Create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
print('  '.join(classes[labels[j]] for j in range(4)))
```

## The Model 

我们将在本示例中使用的模型是 LeNet-5 的变体 - 如果您看过本系列之前的视频，它应该很熟悉。

```python
import torch.nn as nn
import torch.nn.functional as F

# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = GarmentClassifier()
```

## Loss Function

对于这个例子，我们将使用交叉熵损失。出于演示目的，我们将创建一批虚拟输出和标签值，通过损失函数运行它们，并检查结果。

```python
loss_fn = torch.nn.CrossEntropyLoss()

# NB: Loss functions expect data in batches, so we're creating batches of 4
# Represents the model's confidence in each of the 10 classes for a given input
dummy_outputs = torch.rand(4, 10)
# Represents the correct class among the 10 being tested
dummy_labels = torch.tensor([1, 5, 3, 7])

print(dummy_outputs)
print(dummy_labels)

loss = loss_fn(dummy_outputs, dummy_labels)
print('Total loss for this batch: {}'.format(loss.item()))
```

```python
tensor([[0.7026, 0.1489, 0.0065, 0.6841, 0.4166, 0.3980, 0.9849, 0.6701, 0.4601,
         0.8599],
        [0.7461, 0.3920, 0.9978, 0.0354, 0.9843, 0.0312, 0.5989, 0.2888, 0.8170,
         0.4150],
        [0.8408, 0.5368, 0.0059, 0.8931, 0.3942, 0.7349, 0.5500, 0.0074, 0.0554,
         0.1537],
        [0.7282, 0.8755, 0.3649, 0.4566, 0.8796, 0.2390, 0.9865, 0.7549, 0.9105,
         0.5427]])
tensor([1, 5, 3, 7])
Total loss for this batch: 2.428950071334839
```

## Optimizer 

在这个例子中，我们将使用简单的动量随机梯度下降。

尝试此优化方案的一些变化可能会很有启发：

- 学习率决定优化器采取的步骤的大小。就准确性和收敛时间而言，不同的学习率对训练结果有何影响？
- 动量将优化器推向多个步骤中梯度最强的方向。改变这个值对你的结果有什么影响？
- 尝试一些不同的优化算法，例如平均 SGD、Adagrad 或 Adam。您的结果有何不同？

```python
# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

## The Training Loop 

下面，我们有一个执行一个训练周期的函数。它枚举来自 DataLoader 的数据，并在循环的每次传递中执行以下操作：

- 从DataLoader获取一批训练数据
- 将优化器的梯度归零
- 执行推理 - 即从模型中获取输入批次的预测
- 计算该组预测与数据集标签的损失
- 计算学习权重的后向梯度
- 告诉优化器执行一个学习步骤 - 即根据我们选择的优化算法，根据观察到的该批次的梯度来调整模型的学习权重
- 它报告每 1000 批次的损失。
- 最后，它报告最后 1000 个批次的平均每批次损失，以便与验证运行进行比较

```python
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
```

### Per-Epoch Activity 

每个时期我们需要做一次以下几件事：

- 通过检查一组未用于训练的数据的相对损失来执行验证，并报告这一点
- 保存模型的副本

在这里，我们将在 TensorBoard 中进行报告。这将需要进入命令行来启动 TensorBoard，并在另一个浏览器选项卡中将其打开。

```python
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
```

```python
EPOCH 1:
  batch 1000 loss: 1.6334228584356607
  batch 2000 loss: 0.8325267538074403
  batch 3000 loss: 0.7359380583595484
  batch 4000 loss: 0.6198329215242994
  batch 5000 loss: 0.6000315657821484
  batch 6000 loss: 0.555109024874866
  batch 7000 loss: 0.5260250487388112
  batch 8000 loss: 0.4973462742221891
  batch 9000 loss: 0.4781935699362075
  batch 10000 loss: 0.47880298678041433
  batch 11000 loss: 0.45598648857555235
  batch 12000 loss: 0.4327470133750467
  batch 13000 loss: 0.41800182418141046
  batch 14000 loss: 0.4115047634313814
  batch 15000 loss: 0.4211296908891527
LOSS train 0.4211296908891527 valid 0.414460688829422
EPOCH 2:
  batch 1000 loss: 0.3879808729066281
  batch 2000 loss: 0.35912817339546743
  batch 3000 loss: 0.38074520684120944
  batch 4000 loss: 0.3614532373107213
  batch 5000 loss: 0.36850082185724753
  batch 6000 loss: 0.3703581801643886
  batch 7000 loss: 0.38547042514081115
  batch 8000 loss: 0.37846584360170527
  batch 9000 loss: 0.3341486988377292
  batch 10000 loss: 0.3433013284947956
  batch 11000 loss: 0.35607743899174965
  batch 12000 loss: 0.3499939931873523
  batch 13000 loss: 0.33874178926000603
  batch 14000 loss: 0.35130289171106416
  batch 15000 loss: 0.3394507191307202
LOSS train 0.3394507191307202 valid 0.3581162691116333
EPOCH 3:
  batch 1000 loss: 0.3319729989422485
  batch 2000 loss: 0.29558994361863006
  batch 3000 loss: 0.3107374766407593
  batch 4000 loss: 0.3298987646112146
  batch 5000 loss: 0.30858693152241906
  batch 6000 loss: 0.33916381367447684
  batch 7000 loss: 0.3105102765217889
  batch 8000 loss: 0.3011080777524912
  batch 9000 loss: 0.3142058177240979
  batch 10000 loss: 0.31458891937109
  batch 11000 loss: 0.31527258940579483
  batch 12000 loss: 0.31501667268342864
  batch 13000 loss: 0.3011875962628328
  batch 14000 loss: 0.30012811454350596
  batch 15000 loss: 0.31833117976446373
LOSS train 0.31833117976446373 valid 0.3307691514492035
EPOCH 4:
  batch 1000 loss: 0.2786161053752294
  batch 2000 loss: 0.27965198021690596
  batch 3000 loss: 0.28595415444140965
  batch 4000 loss: 0.292985666413857
  batch 5000 loss: 0.3069892351147719
  batch 6000 loss: 0.29902250939945224
  batch 7000 loss: 0.2863366014406201
  batch 8000 loss: 0.2655441066541243
  batch 9000 loss: 0.3045048695363293
  batch 10000 loss: 0.27626545656517554
  batch 11000 loss: 0.2808379335970967
  batch 12000 loss: 0.29241049340573955
  batch 13000 loss: 0.28030834131941446
  batch 14000 loss: 0.2983542350126445
  batch 15000 loss: 0.3009556676162611
LOSS train 0.3009556676162611 valid 0.41686952114105225
EPOCH 5:
  batch 1000 loss: 0.2614263167564495
  batch 2000 loss: 0.2587047562422049
  batch 3000 loss: 0.2642477260621345
  batch 4000 loss: 0.2825975873669813
  batch 5000 loss: 0.26987933717705165
  batch 6000 loss: 0.2759250026817317
  batch 7000 loss: 0.26055969463163275
  batch 8000 loss: 0.29164007206353565
  batch 9000 loss: 0.2893096504513578
  batch 10000 loss: 0.2486029507305684
  batch 11000 loss: 0.2732803234480907
  batch 12000 loss: 0.27927226484491985
  batch 13000 loss: 0.2686819267635074
  batch 14000 loss: 0.24746483912148323
  batch 15000 loss: 0.27903492261294194
LOSS train 0.27903492261294194 valid 0.31206756830215454
```

要加载模型的保存版本：

```python
saved_model = GarmentClassifier()
saved_model.load_state_dict(torch.load(PATH))
```

加载模型后，它就可以满足您的任何需要 - 更多训练、推理或分析。

请注意，如果您的模型具有影响模型结构的构造函数参数，则您需要提供它们并将模型配置为与保存模型时的状态相同。

## Other Resources 

- 有关数据实用程序的文档，包括 Dataset 和 DataLoader，位于 pytorch.org
- 关于使用固定内存进行 GPU 训练的说明
- 有关 TorchVision、TorchText 和 TorchAudio 中可用数据集的文档
- 有关 PyTorch 中可用损失函数的文档
- torch.optim 包的文档，其中包括优化器和相关工具，例如学习率调度
- 有关保存和加载模型的详细教程
- pytorch.org 的教程部分包含有关各种训练任务的教程，包括不同领域的分类、生成对抗网络、强化学习等
