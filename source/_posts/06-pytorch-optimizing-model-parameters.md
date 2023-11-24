---
title: 06-pytorch-optimizing model parameters
date: 2023-11-10 17:26:08
tags:
- 深度学习
- optimizing model parameters
cover: https://w.wallhaven.cc/full/o5/wallhaven-o57qy9.png
---

# OPTIMIZING MODEL PARAMETERS 

现在我们有了模型和数据，是时候通过优化数据上的参数来训练、验证和测试我们的模型了。训练模型是一个迭代过程；在每次迭代中，模型都会对输出进行猜测，计算其猜测的误差（损失），收集误差相对于其参数的导数（如我们在上一节中看到的），并使用梯度下降优化这些参数。有关此过程的更详细演练，请观看 3Blue1Brown 的有关反向传播的
[视频](https://www.youtube.com/watch?v=tIeHLnjs5U8)

## Prerequisite Code 

我们加载前面有关数据集和数据加载器以及构建模型部分的代码。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

model = NeuralNetwork()
```

## Hyperparameters 

超参数是可调整的参数，可让您控制模型优化过程。不同的超参数值会影响模型训练和收敛速度（阅读有关超参数调整的更多信息）

我们定义以下训练超参数：

- Number of Epochs - 迭代数据集的次数
- Batch Size - 参数更新之前通过网络传播的数据样本数量
- 学习率 - 每个批次/时期更新模型参数的量。较小的值会导致学习速度较慢，而较大的值可能会导致训练期间出现不可预测的行为。

```python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

## Optimization Loop 

一旦我们设置了超参数，我们就可以使用优化循环来训练和优化我们的模型。优化循环的每次迭代称为一个时期epoch。

每个时期epoch由两个主要部分组成：

- The Train Loop:迭代训练数据集并尝试收敛到最佳参数。
- The Validation/Test Loop:迭代测试数据集以检查模型性能是否有所改善。

让我们简单熟悉一下训练循环中使用的一些概念。向前跳转查看优化循环的完整实现。

### Loss Function 

当提供一些训练数据时，我们未经训练的网络可能不会给出正确的答案。损失函数衡量的是得到的结果与目标值的不相似程度，它是我们在训练时想要最小化的损失函数。为了计算损失，我们使用给定数据样本的输入进行预测，并将其与真实数据标签值进行比较。

常见的损失函数包括用于回归任务的 nn.MSELoss（均方误差）和用于分类的 nn.NLLLoss（负对数似然）。 nn.CrossEntropyLoss 结合了 `nn.LogSoftmax` 和 `nn.NLLLoss` 。

我们将模型的输出 logits 传递给 `nn.CrossEntropyLoss` ，这将标准化 logits 并计算预测误差。

```python
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
```

### Optimizer

优化是调整模型参数以减少每个训练步骤中模型误差的过程。优化算法定义了如何执行此过程（在本例中我们使用随机梯度下降）。所有优化逻辑都封装在 `optimizer` 对象中。这里，我们使用SGD优化器；此外，PyTorch 中还有许多不同的优化器，例如 ADAM 和 RMSProp，它们可以更好地处理不同类型的模型和数据。

我们通过注册需要训练的模型参数并传入学习率超参数来初始化优化器。

在训练循环中，优化分三个步骤进行：

- 调用 `optimizer.zero_grad()` 重置模型参数的梯度。默认情况下渐变相加；为了防止重复计算，我们在每次迭代时明确地将它们归零。
- 通过调用 `loss.backward()` 反向传播预测损失。 PyTorch 存储损失的梯度。每个参数。
- 一旦我们有了梯度，我们就调用 optimizer.step() 通过向后传递中收集的梯度来调整参数。

## Full Implementation

我们定义了循环优化代码的 `train_loop` 和根据测试数据评估模型性能的 `test_loop` 。

```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

我们初始化损失函数和优化器，并将其传递给 `train_loop` 和 `test_loop` 。您可以随意增加纪元数来跟踪模型性能的改进。

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

