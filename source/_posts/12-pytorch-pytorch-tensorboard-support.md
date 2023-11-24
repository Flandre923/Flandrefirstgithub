---
title: 12-pytorch-pytorch tensorboard support
date: 2023-11-15 11:37:50
tags:
- 深度学习
- pytorch
cover: https://view.moezx.cc/images/2017/11/25/miku_52113740.jpg
---

# PYTORCH TENSORBOARD SUPPORT

## Before You Start 

要运行本教程，您需要安装 PyTorch、TorchVision、Matplotlib 和 TensorBoard。

With `conda`:

```python
conda install pytorch torchvision -c pytorch
conda install matplotlib tensorboard
```

With pip:

```python
pip install torch torchvision matplotlib tensorboard
```

安装依赖项后，在安装它们的 Python 环境中重新启动此笔记本。

## Introduction 

在本笔记本中，我们将针对 Fashion-MNIST 数据集训练 LeNet-5 的变体。 Fashion-MNIST 是一组描绘各种服装的图像图块，其中有十个类别标签指示所描绘服装的类型。

```python
# PyTorch model and training necessities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Image datasets and image manipulation
import torchvision
import torchvision.transforms as transforms

# Image display
import matplotlib.pyplot as plt
import numpy as np

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

# In case you are using an environment that has TensorFlow installed,
# such as Google Colab, uncomment the following code to avoid
# a bug with saving embeddings to your TensorBoard directory

# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
```

## Showing Images in TensorBoard 

首先，我们将数据集中的示例图像添加到 TensorBoard：

```python
# Gather datasets and prepare them for consumption
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# Store separate training and validations splits in ./data
training_set = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
validation_set = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)

training_loader = torch.utils.data.DataLoader(training_set,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=2)


validation_loader = torch.utils.data.DataLoader(validation_set,
                                                batch_size=4,
                                                shuffle=False,
                                                num_workers=2)

# Class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

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

# Extract a batch of 4 images
dataiter = iter(training_loader)
images, labels = next(dataiter)

# Create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
```

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ./data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 65536/26421880 [00:00<01:12, 364190.06it/s]
  1%|          | 229376/26421880 [00:00<00:38, 681499.72it/s]
  2%|2         | 655360/26421880 [00:00<00:14, 1797638.14it/s]
  7%|6         | 1769472/26421880 [00:00<00:06, 3863535.64it/s]
 14%|#3        | 3670016/26421880 [00:00<00:02, 7836390.00it/s]
 21%|##1       | 5668864/26421880 [00:00<00:02, 9387876.09it/s]
 32%|###1      | 8454144/26421880 [00:01<00:01, 13858138.54it/s]
 40%|###9      | 10551296/26421880 [00:01<00:01, 13275947.57it/s]
 50%|#####     | 13303808/26421880 [00:01<00:00, 16592868.65it/s]
 58%|#####8    | 15433728/26421880 [00:01<00:00, 15098119.07it/s]
 69%|######8   | 18120704/26421880 [00:01<00:00, 17772010.13it/s]
 77%|#######7  | 20381696/26421880 [00:01<00:00, 16117299.56it/s]
 87%|########6 | 22970368/26421880 [00:01<00:00, 18330844.79it/s]
 96%|#########5| 25362432/26421880 [00:01<00:00, 16821085.78it/s]
100%|##########| 26421880/26421880 [00:02<00:00, 13143960.49it/s]
Extracting ./data/FashionMNIST/raw/train-images-idx3-ubyte.gz to ./data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ./data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 324555.51it/s]
Extracting ./data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to ./data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ./data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|1         | 65536/4422102 [00:00<00:12, 361458.87it/s]
  5%|5         | 229376/4422102 [00:00<00:06, 680363.18it/s]
 20%|##        | 884736/4422102 [00:00<00:01, 2498877.41it/s]
 43%|####2     | 1900544/4422102 [00:00<00:00, 4401526.26it/s]
 79%|#######9  | 3506176/4422102 [00:00<00:00, 7212994.99it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 6063383.04it/s]
Extracting ./data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to ./data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ./data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 39187435.56it/s]
Extracting ./data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to ./data/FashionMNIST/raw
```

上面，我们使用 TorchVision 和 Matplotlib 创建了小批量输入数据的可视化网格。下面，我们使用 `SummaryWriter` 上的 `add_image()` 调用来记录图像以供 TensorBoard 使用，并且我们还调用 `flush()` 以确保它立即写入磁盘。

```python 
# Default log_dir argument is "runs" - but it's good to be specific
# torch.utils.tensorboard.SummaryWriter is imported above
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# Write image data to TensorBoard log dir
writer.add_image('Four Fashion-MNIST Images', img_grid)
writer.flush()

# To view, start TensorBoard on the command line with:
#   tensorboard --logdir=runs
# ...and open a browser tab to http://localhost:6006/
```

如果您在命令行启动 TensorBoard 并在新的浏览器选项卡（通常在 localhost:6006）中打开它，您应该在 IMAGES 选项卡下看到图像网格。

## Graphing Scalars to Visualize Training 

TensorBoard 对于跟踪训练进度和效果非常有用。下面，我们将运行一个训练循环，跟踪一些指标，并保存数据以供 TensorBoard 的使用。

让我们定义一个模型来对图像图块进行分类，以及用于训练的优化器和损失函数：

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

现在让我们训练一个 epoch，并评估每 1000 个批次的训练集损失与验证集损失：

```python
print(len(validation_loader))
for epoch in range(1):  # loop over the dataset multiple times
    running_loss = 0.0

    for i, data in enumerate(training_loader, 0):
        # basic training loop
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # Every 1000 mini-batches...
            print('Batch {}'.format(i + 1))
            # Check against the validation set
            running_vloss = 0.0

            # In evaluation mode some model specific operations can be omitted eg. dropout layer
            net.train(False) # Switching to evaluation mode, eg. turning off regularisation
            for j, vdata in enumerate(validation_loader, 0):
                vinputs, vlabels = vdata
                voutputs = net(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()
            net.train(True) # Switching back to training mode, eg. turning on regularisation

            avg_loss = running_loss / 1000
            avg_vloss = running_vloss / len(validation_loader)

            # Log the running loss averaged per batch
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch * len(training_loader) + i)

            running_loss = 0.0
print('Finished Training')

writer.flush()
```

切换到打开的 TensorBoard 并查看 SCALARS 选项卡。

## Visualizing Your Model

TensorBoard 还可用于检查模型中的数据流。为此，请使用模型和示例输入调用 `add_graph()` 方法。当你打开时

```python
# Again, grab a single mini-batch of images
dataiter = iter(training_loader)
images, labels = next(dataiter)

# add_graph() will trace the sample input through your model,
# and render it as a graph.
writer.add_graph(net, images)
writer.flush()
```

当您切换到 TensorBoard 时，您应该会看到一个 GRAPHS 选项卡。双击“NET”节点以查看模型中的层和数据流。

## Visualizing Your Dataset with Embeddings 

我们使用的 28×28 图像块可以建模为 784 维向量 (28 * 28 = 784)。将其投影到较低维的表示形式可能会很有启发。 `add_embedding()` 方法会将一组数据投影到方差最大的三个维度上，并将它们显示为交互式 3D 图表。 `add_embedding()` 方法通过投影到方差最高的三个维度来自动执行此操作。

下面，我们将采集数据样本，并生成这样的嵌入：

```python
# Select a random subset of data and corresponding labels
def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# Extract a random subset of data
images, labels = select_n_random(training_set.data, training_set.targets)

# get the class labels for each image
class_labels = [classes[label] for label in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.flush()
writer.close()
```



现在，如果您切换到 TensorBoard 并选择“投影仪”选项卡，您应该会看到投影的 3D 表示。您可以旋转和缩放模型。在大尺度和小尺度上检查它，看看是否可以发现投影数据和标签聚类中的模式。

为了获得更好的可见性，建议：

- 从左侧的“颜色依据”下拉列表中选择“标签”。
- 切换顶部的夜间模式图标，将浅色图像放置在深色背景上。

## Other Resources

欲了解更多信息，请查看：

- torch.utils.tensorboard.SummaryWriter 上的 PyTorch 文档
- PyTorch.org 教程中的 Tensorboard 教程内容
- 有关 TensorBoard 的更多信息，请参阅 TensorBoard 文档
