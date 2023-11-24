---
title: 02_pytorch_datasets_DataLoaders
date: 2023-11-10 15:27:31
tags:
- 深度学习
- pytorch
cover: https://w.wallhaven.cc/full/2y/wallhaven-2y6v16.jpg
---



# DATASETS & DATALOADERS 

**DATASETS是什么？**

torch.utils.data.Dataset，

`Dataset`允许您使用预加载的数据集以及您自己的数据

`Dataset` 存储样本及其相应的标签

**DATALOADERS是什么？**

torch.utils.data.DataLoader

`DataLoader` 在 `Dataset` 周围包装一个迭代，以便轻松访问样本。

**解决什么问题？**

用于处理数据样本的代码可能会变得混乱且难以维护

理想情况下，我们希望数据集代码与模型训练代码分离，以获得更好的可读性和模块化性

**PyTorch 提供了什么数据集，有什么用**

提供了许多预加载的数据集（例如 FashionMNIST）。

它们对 `torch.utils.data.Dataset` 进行子类化并实现特定于特定数据的函数

> 给我们提示，如果要构建自己的数据集同样应该继承Dataset

可以做什么：对您的模型进行原型设计和基准测试。

在哪里找到：[Image Datasets](https://pytorch.org/vision/stable/datasets.html), [Text Datasets](https://pytorch.org/text/stable/datasets.html), and [Audio Datasets](https://pytorch.org/audio/stable/datasets.html)

## Loading a Dataset 加载数据集

**Fashion-MNIST是什么数据集？**

 [Fashion-MNIST](https://research.zalando.com/project/fashion_mnist/fashion_mnist/) 是 Zalando 文章图像的数据集，由 60,000 个训练示例和 10,000 个测试示例组成。每个示例包含一个 28×28 灰度图像和来自 10 个类别之一的关联标签。

 **如何加载[FashionMNIST Dataset](https://pytorch.org/vision/stable/datasets.html#fashion-mnist) 数据集？**

- `root` 是存储训练/测试数据的路径，
- `train` 指定训练或测试数据集，
- 如果 `root` 上没有数据，则 `download=True` 会从 Internet 下载数据。
- transform 和 target_transform 指定特征和标签转换

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


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
```

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 65536/26421880 [00:00<01:12, 365336.26it/s]
  1%|          | 229376/26421880 [00:00<00:38, 685596.52it/s]
  3%|3         | 851968/26421880 [00:00<00:10, 2415122.44it/s]
  7%|7         | 1900544/26421880 [00:00<00:06, 4075765.30it/s]
 18%|#8        | 4882432/26421880 [00:00<00:01, 10855436.54it/s]
 25%|##4       | 6586368/26421880 [00:00<00:01, 11527877.37it/s]
 31%|###1      | 8257536/26421880 [00:00<00:01, 12111293.67it/s]
 43%|####2     | 11337728/26421880 [00:01<00:00, 16825176.39it/s]
 50%|####9     | 13205504/26421880 [00:01<00:00, 14645303.60it/s]
 62%|######1   | 16252928/26421880 [00:01<00:00, 18391128.94it/s]
 69%|######9   | 18284544/26421880 [00:01<00:00, 16056294.06it/s]
 80%|########  | 21266432/26421880 [00:01<00:00, 19223679.44it/s]
 89%|########8 | 23396352/26421880 [00:01<00:00, 16815978.59it/s]
 99%|#########9| 26279936/26421880 [00:01<00:00, 19554902.51it/s]
100%|##########| 26421880/26421880 [00:01<00:00, 13595820.68it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 325785.44it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|1         | 65536/4422102 [00:00<00:12, 362766.41it/s]
  5%|5         | 229376/4422102 [00:00<00:06, 682006.25it/s]
 20%|##        | 884736/4422102 [00:00<00:01, 2511806.30it/s]
 44%|####3     | 1933312/4422102 [00:00<00:00, 4114396.78it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 6090778.97it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 65830112.78it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

## Iterating and Visualizing the Dataset 

如何可视化查看数据集？

虽然我们可以切片的方式查看数据集，但是这样有些麻烦和不直观。

```
training_data[index]
```

我们使用 `matplotlib` 来可视化训练数据中的一些样本。

> 给我们的一种可视化的方式

```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

![Ankle Boot, Shirt, Bag, Ankle Boot, Trouser, Sandal, Coat, Sandal, Pullover](https://s2.loli.net/2023/11/10/4yYK3lGWcBizCPD.png)

## Creating a Custom Dataset for your files 

**怎么自定义自己的数据集?**

自定义 Dataset 类必须实现三个函数,\_\_init\_\_、\_\_len\_\_ 和\_\_getitem\_\_.

可以参考FashionMNIST的实现，下面是例子：

图像存储在目录 `img_dir` 中，它们的标签单独存储在 CSV 文件 `annotations_file` 中。

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

### \_\_init\_\_ 

**在\_\_init_\_方法中我需要做什么？**

__init__ 函数在实例化 Dataset 对象时运行一次。我们初始化包含图像、注释文件和两种转换的目录（下一节将更详细地介绍）。

labels.csv 文件如下所示：

```
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

```python
def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
```

### \__len__ 

**\__len__方法中我需要完成什么工作?**

\__len__ 函数返回数据集中的样本数。

```
def __len__(self):
    return len(self.img_labels)
```

### \__getitem__ 

\__getitem__ 函数加载并返回给定索引 `idx` 处的数据集的样本。

1. 基于索引，它识别图像在磁盘上的位置，
2. 使用 `read_image` 将其转换为张量，
3. 从 `self.img_labels` 中的 csv 数据中检索相应的标签，
4. 对其调用转换函数（如果适用），
5. 并返回元组中的张量图像和相应的标签。

## Preparing your data for training with DataLoaders 

**为什么使用DataLoaders ？DataSet不够用吗？**

`Dataset` 检索数据集的特征并一次标记一个样本。在训练模型时，我们通常希望以“小批量”的方式传递样本，在每个时期重新整理数据以减少模型过度拟合，并使用 Python 的 `multiprocessing` 来加速数据检索。

**基于上述我们的需求，希望有一种可以通用的API帮我我们获得数据。**

`DataLoader` 是一个可迭代对象，它通过一个简单的 API 为我们抽象了这种复杂性。

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

## Iterate through the DataLoader 

**我们应该如何通过DataLoader获得数据和标签？**

我们已将该数据集加载到 `DataLoader` 中，并且可以根据需要迭代数据集。下面的每次迭代都会返回一批 `train_features` 和 `train_labels` （分别包含 `batch_size=64` 特征和标签）。因为我们指定了 `shuffle=True` ，所以在迭代所有批次后，数据会被打乱（为了更细粒度地控制数据加载顺序，请查看 Samplers）。

```python
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 5
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

```
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 5
```
