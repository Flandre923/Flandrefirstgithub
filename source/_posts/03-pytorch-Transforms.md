---
title: 03_pytorch_Transforms
date: 2023-11-10 16:09:19
tags:
- 深度学习
- transforms
cover: https://w.wallhaven.cc/full/yx/wallhaven-yxk6qd.png
---

# TRANSFORMS

**Transforms是什么？**

数据并不总是以训练机器学习算法所需的最终处理形式出现。我们使用转换来对数据执行一些操作并使其适合训练。

**应该如何转化？**

所有 TorchVision 数据集都有两个参数 - 用于修改功能的 `transform` 和用于修改标签的 `target_transform` - 接受包含转换逻辑的可调用对象。 [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)  模块提供了几种开箱即用的常用转换。

例如:

FashionMNIST特征采用PIL图像格式，标签为整数。对于训练，我们需要将特征作为归一化张量，将标签作为单热编码张量。为了进行这些转换，我们使用 `ToTensor` 和 `Lambda` 。

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 65536/26421880 [00:00<01:12, 361134.05it/s]
  1%|          | 229376/26421880 [00:00<00:38, 678664.07it/s]
  3%|2         | 753664/26421880 [00:00<00:12, 2053982.29it/s]
  5%|5         | 1409024/26421880 [00:00<00:08, 2875069.78it/s]
 12%|#2        | 3244032/26421880 [00:00<00:03, 6812170.26it/s]
 21%|##1       | 5570560/26421880 [00:00<00:02, 9643810.01it/s]
 30%|##9       | 7897088/26421880 [00:01<00:01, 12668660.86it/s]
 39%|###9      | 10354688/26421880 [00:01<00:01, 13567884.70it/s]
 47%|####7     | 12517376/26421880 [00:01<00:00, 15107192.72it/s]
 57%|#####7    | 15138816/26421880 [00:01<00:00, 15464642.87it/s]
 65%|######4   | 17170432/26421880 [00:01<00:00, 16205991.36it/s]
 75%|#######5  | 19922944/26421880 [00:01<00:00, 16485428.61it/s]
 83%|########2 | 21823488/26421880 [00:01<00:00, 16620077.17it/s]
 94%|#########3| 24739840/26421880 [00:02<00:00, 17125086.87it/s]
100%|##########| 26421880/26421880 [00:02<00:00, 12608239.78it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 327666.12it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|1         | 65536/4422102 [00:00<00:12, 360810.29it/s]
  5%|5         | 229376/4422102 [00:00<00:06, 676632.69it/s]
 19%|#8        | 819200/4422102 [00:00<00:01, 2275103.44it/s]
 33%|###2      | 1441792/4422102 [00:00<00:01, 2899610.49it/s]
 74%|#######4  | 3276800/4422102 [00:00<00:00, 6877973.06it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 5392190.04it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 36597079.65it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

## ToTensor() 

ToTensor 将 PIL 图像或 NumPy `ndarray` 转换为 `FloatTensor` 。并在 [0., 1.] 范围内缩放图像的像素强度值

## Lambda Transforms 

Lambda 转换应用任何用户定义的 lambda 函数。在这里，我们定义一个函数将整数转换为 one-hot 编码张量。它首先创建一个大小为 10 的零张量（数据集中的标签数量）并调用 scatter_ ，它在标签 `y` 给出的索引上分配 `value=1` 。

```python
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```

