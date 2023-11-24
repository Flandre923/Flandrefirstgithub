---
title: 11-pytorch-building models with pytorch
date: 2023-11-15 10:49:32
tags:
- 深度学习
- pytorch
cover: https://view.moezx.cc/images/2018/09/20/id-65179698.jpg
---

# BUILDING MODELS WITH PYTORCH

## `torch.nn.Module` and `torch.nn.Parameter`

除了 `Parameter` 之外，我们在本视频中讨论的类都是 `torch.nn.Module` 的子类。这是 PyTorch 基类，旨在封装特定于 PyTorch 模型及其组件的行为。

`torch.nn.Module` 的一项重要行为是注册参数。如果特定的 `Module` 子类具有学习权重，这些权重将表示为 `torch.nn.Parameter` 的实例。 `Parameter` 类是 `torch.Tensor` 的子类，具有特殊行为，当它们被指定为 `Module` 的属性时，它们将被添加到该列表中模块参数。这些参数可以通过 `Module` 类上的 `parameters()` 方法访问。

作为一个简单的例子，这是一个非常简单的模型，具有两个线性层和一个激活函数。我们将创建它的一个实例并要求它报告其参数：

```python
import torch

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = TinyModel()

print('The model:')
print(tinymodel)

print('\n\nJust one layer:')
print(tinymodel.linear2)

print('\n\nModel params:')
for param in tinymodel.parameters():
    print(param)

print('\n\nLayer params:')
for param in tinymodel.linear2.parameters():
    print(param)
```

```
The model:
TinyModel(
  (linear1): Linear(in_features=100, out_features=200, bias=True)
  (activation): ReLU()
  (linear2): Linear(in_features=200, out_features=10, bias=True)
  (softmax): Softmax(dim=None)
)


Just one layer:
Linear(in_features=200, out_features=10, bias=True)


Model params:
Parameter containing:
tensor([[ 0.0765,  0.0830, -0.0234,  ..., -0.0337, -0.0355, -0.0968],
        [-0.0573,  0.0250, -0.0132,  ..., -0.0060,  0.0240,  0.0280],
        [-0.0908, -0.0369,  0.0842,  ..., -0.0078, -0.0333, -0.0324],
        ...,
        [-0.0273, -0.0162, -0.0878,  ...,  0.0451,  0.0297, -0.0722],
        [ 0.0833, -0.0874, -0.0020,  ..., -0.0215,  0.0356,  0.0405],
        [-0.0637,  0.0190, -0.0571,  ..., -0.0874,  0.0176,  0.0712]],
       requires_grad=True)
Parameter containing:
tensor([ 0.0304, -0.0758, -0.0549, -0.0893, -0.0809, -0.0804, -0.0079, -0.0413,
        -0.0968,  0.0888,  0.0239, -0.0659, -0.0560, -0.0060,  0.0660, -0.0319,
        -0.0370,  0.0633, -0.0143, -0.0360,  0.0670, -0.0804,  0.0265, -0.0870,
         0.0039, -0.0174, -0.0680, -0.0531,  0.0643,  0.0794,  0.0209,  0.0419,
         0.0562, -0.0173, -0.0055,  0.0813,  0.0613, -0.0379,  0.0228,  0.0304,
        -0.0354,  0.0609, -0.0398,  0.0410,  0.0564, -0.0101, -0.0790, -0.0824,
        -0.0126,  0.0557,  0.0900,  0.0597,  0.0062, -0.0108,  0.0112, -0.0358,
        -0.0203,  0.0566, -0.0816, -0.0633, -0.0266, -0.0624, -0.0746,  0.0492,
         0.0450,  0.0530, -0.0706,  0.0308,  0.0533,  0.0202, -0.0469, -0.0448,
         0.0548,  0.0331,  0.0257, -0.0764, -0.0892,  0.0783,  0.0062,  0.0844,
        -0.0959, -0.0468, -0.0926,  0.0925,  0.0147,  0.0391,  0.0765,  0.0059,
         0.0216, -0.0724,  0.0108,  0.0701, -0.0147, -0.0693, -0.0517,  0.0029,
         0.0661,  0.0086, -0.0574,  0.0084, -0.0324,  0.0056,  0.0626, -0.0833,
        -0.0271, -0.0526,  0.0842, -0.0840, -0.0234, -0.0898, -0.0710, -0.0399,
         0.0183, -0.0883, -0.0102, -0.0545,  0.0706, -0.0646, -0.0841, -0.0095,
        -0.0823, -0.0385,  0.0327, -0.0810, -0.0404,  0.0570,  0.0740,  0.0829,
         0.0845,  0.0817, -0.0239, -0.0444, -0.0221,  0.0216,  0.0103, -0.0631,
         0.0831, -0.0273,  0.0756,  0.0022,  0.0407,  0.0072,  0.0374, -0.0608,
         0.0424, -0.0585,  0.0505, -0.0455,  0.0268, -0.0950, -0.0642,  0.0843,
         0.0760, -0.0889, -0.0617, -0.0916,  0.0102, -0.0269, -0.0011,  0.0318,
         0.0278, -0.0160,  0.0159, -0.0817,  0.0768, -0.0876, -0.0524, -0.0332,
        -0.0583,  0.0053,  0.0503, -0.0342, -0.0319, -0.0562,  0.0376, -0.0696,
         0.0735,  0.0222, -0.0775, -0.0072,  0.0294,  0.0994, -0.0355, -0.0809,
        -0.0539,  0.0245,  0.0670,  0.0032,  0.0891, -0.0694, -0.0994,  0.0126,
         0.0629,  0.0936,  0.0058, -0.0073,  0.0498,  0.0616, -0.0912, -0.0490],
       requires_grad=True)
Parameter containing:
tensor([[ 0.0504, -0.0203, -0.0573,  ...,  0.0253,  0.0642, -0.0088],
        [-0.0078, -0.0608, -0.0626,  ..., -0.0350, -0.0028, -0.0634],
        [-0.0317, -0.0202, -0.0593,  ..., -0.0280,  0.0571, -0.0114],
        ...,
        [ 0.0582, -0.0471, -0.0236,  ...,  0.0273,  0.0673,  0.0555],
        [ 0.0258, -0.0706,  0.0315,  ..., -0.0663, -0.0133,  0.0078],
        [-0.0062,  0.0544, -0.0280,  ..., -0.0303, -0.0326, -0.0462]],
       requires_grad=True)
Parameter containing:
tensor([ 0.0385, -0.0116,  0.0703,  0.0407, -0.0346, -0.0178,  0.0308, -0.0502,
         0.0616,  0.0114], requires_grad=True)


Layer params:
Parameter containing:
tensor([[ 0.0504, -0.0203, -0.0573,  ...,  0.0253,  0.0642, -0.0088],
        [-0.0078, -0.0608, -0.0626,  ..., -0.0350, -0.0028, -0.0634],
        [-0.0317, -0.0202, -0.0593,  ..., -0.0280,  0.0571, -0.0114],
        ...,
        [ 0.0582, -0.0471, -0.0236,  ...,  0.0273,  0.0673,  0.0555],
        [ 0.0258, -0.0706,  0.0315,  ..., -0.0663, -0.0133,  0.0078],
        [-0.0062,  0.0544, -0.0280,  ..., -0.0303, -0.0326, -0.0462]],
       requires_grad=True)
Parameter containing:
tensor([ 0.0385, -0.0116,  0.0703,  0.0407, -0.0346, -0.0178,  0.0308, -0.0502,
         0.0616,  0.0114], requires_grad=True)
```

这显示了 PyTorch 模型的基本结构：有一个 `__init__()` 方法定义模型的层和其他组件，还有一个 `forward()` 方法完成计算。请注意，我们可以打印模型或其任何子模块，以了解其结构。

## Common Layer Types 

### Linear Layers 

神经网络层最基本的类型是线性或全连接层。在该层中，每个输入都会影响该层的每个输出，其程度由该层的权重指定。如果模型有 m 个输入和 n 个输出，则权重将是 m x n 矩阵。例如：

```python
lin = torch.nn.Linear(3, 2)
x = torch.rand(1, 3)
print('Input:')
print(x)

print('\n\nWeight and Bias parameters:')
for param in lin.parameters():
    print(param)

y = lin(x)
print('\n\nOutput:')
print(y)
```

```
Input:
tensor([[0.8790, 0.9774, 0.2547]])


Weight and Bias parameters:
Parameter containing:
tensor([[ 0.1656,  0.4969, -0.4972],
        [-0.2035, -0.2579, -0.3780]], requires_grad=True)
Parameter containing:
tensor([0.3768, 0.3781], requires_grad=True)


Output:
tensor([[ 0.8814, -0.1492]], grad_fn=<AddmmBackward0>)
```

如果将 `x` 与线性层的权重进行矩阵乘法，并添加偏差，您会发现得到输出向量 `y` 。

需要注意的另一个重要功能：当我们使用 `lin.weight` 检查图层的权重时，它会将自己报告为 `Parameter` （它是 `Tensor` 的子类） ，并让我们知道它正在使用 autograd 跟踪梯度。这是 `Parameter` 的默认行为，与 `Tensor` 不同。

线性层广泛应用于深度学习模型中。最常见的地方之一是在分类器模型中，该模型通常在末尾有一个或多个线性层，其中最后一层将有 n 个输出，其中 n 是分类器处理的类的数量。

### Convolutional Layers 

构建卷积层是为了处理具有**高度空间相关性的数据**。它们在计算机视觉中非常常用，它们检测紧密的特征分组，然后将其组合成更高级别的特征。它们也会出现在其他上下文中 - 例如，在 NLP 应用程序中，单词的直接上下文（即序列中附近的其他单词）可以影响句子的含义。

我们在之前的视频中看到了 LeNet5 中卷积层的作用：

```python
import torch.functional as F


class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

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

让我们分解一下该模型的卷积层中发生的情况。以 `conv1` 开头：

- LeNet5 旨在接收 1x32x32 黑白图像。卷积层构造函数的第一个参数是输入通道的数量。在这里，它是 1。如果我们构建这个模型来查看 3 色通道，那么它就是 3。
- 卷积层就像一个扫描图像的窗口，寻找它可以识别的模式。这些模式称为特征，卷积层的参数之一是我们希望它学习的特征数量。这是构造函数的第二个参数，是输出特征的数量。在这里，我们要求我们的层学习 6 个特征。
- 就在上面，我将卷积层比作一个窗口——但是窗口有多大呢？第三个参数是窗口或内核大小。这里，“5”意味着我们选择了 5x5 内核。 （如果您想要高度与宽度不同的内核，您可以为此参数指定一个元组 - 例如， `(3, 5)` 以获得 3x5 卷积内核。）

卷积层的输出是激活图 - 输入张量中特征存在的空间表示。 `conv1` 将为我们提供 6x28x28 的输出张量； 6 是要素的数量，28 是地图的高度和宽度。 （28 是因为当在 32 像素行上扫描 5 像素窗口时，只有 28 个有效位置。）

然后，我们将卷积的输出传递给 ReLU 激活函数（稍后将详细介绍激活函数），然后传递给最大池化层。最大池化层采用激活图中彼此靠近的特征并将它们分组在一起。它通过减少张量、将输出中的每个 2x2 单元格合并为一个单元格，并为该单元格分配进入其中的 4 个单元格的最大值来实现此目的。这为我们提供了激活图的较低分辨率版本，尺寸为 6x14x14。

我们的下一个卷积层 `conv2` 需要 6 个输入通道（对应于第一层寻求的 6 个特征）、16 个输出通道和一个 3x3 内核。它输出一个 16x12x12 的激活图，再次通过最大池化层减少到 16x6x6。在将此输出传递到线性层之前，它会被重塑为 16 * 6 * 6 = 576 元素的向量，以供下一层使用。

有用于处理 1D、2D 和 3D 张量的卷积层。卷积层构造函数还有更多可选参数，包括输入中的步幅长度（例如，仅扫描每秒或每三个位置）、填充（以便您可以扫描到输入的边缘）等等。请参阅文档以获取更多信息。

### Recurrent Layers 

循环神经网络（或 RNN）用于处理顺序数据——从科学仪器的时间序列测量到自然语言句子再到 DNA 核苷酸。 RNN 通过维护一个隐藏状态来实现这一点，该隐藏状态充当迄今为止在序列中所看到内容的一种记忆。

RNN 层或其变体 LSTM（长短期记忆）和 GRU（门控循环单元）的内部结构相当复杂，超出了本视频的范围，但我们将向您展示其中的结构使用基于 LSTM 的词性标注器（一种分类器，可以告诉你一个单词是否是名词、动词等）：

```python
class LSTMTagger(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
```

构造函数有四个参数：

- `vocab_size` 是输入词汇中的单词数。每个单词都是 `vocab_size` 维空间中的一个单热向量（或单位向量）。
- `tagset_size` 是输出集中的标签数量。
- `embedding_dim` 是词汇嵌入空间的大小。嵌入将词汇映射到低维空间，其中具有相似含义的单词在空间中靠近在一起。
- `hidden_dim` 是 LSTM 内存的大小。

输入将是一个句子，其中单词表示为 one-hot 向量的索引。然后嵌入层会将它们映射到 `embedding_dim` 维空间。 LSTM 获取此嵌入序列并对其进行迭代，生成长度为 `hidden_dim` 的输出向量。最后的线性层充当分类器；将 `log_softmax()` 应用于最后一层的输出会将输出转换为给定单词映射到给定标签的归一化估计概率集。

如果您想了解该网络的实际应用，请查看 pytorch.org 上的序列模型和 LSTM 网络教程。

### Transformers 

Transformer 是一种多用途网络，它已经通过 BERT 等模型取代了 NLP 领域的最先进技术。对 Transformer 架构的讨论超出了本视频的范围，但 PyTorch 有一个 `Transformer` 类，允许您定义 Transformer 模型的整体参数 - 注意力头的数量、编码器和解码器的数量层、dropout 和激活函数等。（您甚至可以使用正确的参数从这个单个类构建 BERT 模型！） `torch.nn.Transformer` 类还具有用于封装各个组件的类（ `TransformerEncoder` ）和子组件（ `TransformerEncoderLayer` 、 `TransformerDecoderLayer` ）。有关详细信息，请查看变压器类的文档以及 pytorch.org 上的相关教程。

### Data Manipulation Layers 

还有其他层类型在模型中执行重要功能，但本身不参与学习过程。

最大池化（及其孪生最小池化）通过组合单元格并将输入单元格的最大值分配给输出单元格来减少张量（我们看到了这一点）。例如：

```python
my_tensor = torch.rand(1, 6, 6)
print(my_tensor)

maxpool_layer = torch.nn.MaxPool2d(3)
print(maxpool_layer(my_tensor))
```

```
tensor([[[0.5036, 0.6285, 0.3460, 0.7817, 0.9876, 0.0074],
         [0.3969, 0.7950, 0.1449, 0.4110, 0.8216, 0.6235],
         [0.2347, 0.3741, 0.4997, 0.9737, 0.1741, 0.4616],
         [0.3962, 0.9970, 0.8778, 0.4292, 0.2772, 0.9926],
         [0.4406, 0.3624, 0.8960, 0.6484, 0.5544, 0.9501],
         [0.2489, 0.8971, 0.7499, 0.1803, 0.9571, 0.6733]]])
tensor([[[0.7950, 0.9876],
         [0.9970, 0.9926]]])
```

如果仔细观察上面的值，您会发现 maxpooled 输出中的每个值都是 6x6 输入的每个象限的最大值。

归一化层将一层的输出重新居中并归一化，然后再将其输入到另一层。居中和缩放中间张量有许多有益的效果，例如让您使用更高的学习率而不会爆炸/消失梯度。

```python
my_tensor = torch.rand(1, 4, 4) * 20 + 5
print(my_tensor)

print(my_tensor.mean())

norm_layer = torch.nn.BatchNorm1d(4)
normed_tensor = norm_layer(my_tensor)
print(normed_tensor)

print(normed_tensor.mean())
```

```python
tensor([[[ 7.7375, 23.5649,  6.8452, 16.3517],
         [19.5792, 20.3254,  6.1930, 23.7576],
         [23.7554, 20.8565, 18.4241,  8.5742],
         [22.5100, 15.6154, 13.5698, 11.8411]]])
tensor(16.2188)
tensor([[[-0.8614,  1.4543, -0.9919,  0.3990],
         [ 0.3160,  0.4274, -1.6834,  0.9400],
         [ 1.0256,  0.5176,  0.0914, -1.6346],
         [ 1.6352, -0.0663, -0.5711, -0.9978]]],
       grad_fn=<NativeBatchNormBackward0>)
tensor(3.3528e-08, grad_fn=<MeanBackward0>)
```

运行上面的单元，我们向输入张量添加了一个大的缩放因子和偏移量；你应该看到输入张量的 `mean()` 位于 15 附近的某个地方。在通过归一化层运行它之后，你可以看到值更小，并且分组在零附近 - 事实上，平均值应该非常大小（> 1e-8）。

这是有益的，因为许多激活函数（如下所述）的最强梯度接近 0，但有时会遭受输入梯度消失或爆炸的影响，从而使它们远离零。将数据保持在最陡梯度区域的中心往往意味着更快、更好的学习和更高的可行学习率。

Dropout 层是一种鼓励模型中稀疏表示的工具，也就是说，推动模型用更少的数据进行推理。

Dropout 层通过在训练期间随机设置部分输入张量来工作 - Dropout 层始终关闭以进行推理。这迫使模型针对这个屏蔽或缩减的数据集进行学习。例如：

```python
my_tensor = torch.rand(1, 4, 4)

dropout = torch.nn.Dropout(p=0.4)
print(dropout(my_tensor))
print(dropout(my_tensor))
```

在上面，您可以看到 dropout 对样本张量的影响。您可以使用可选的 `p` 参数来设置单个权重退出的概率；如果不这样做，则默认为 0.5。

### Activation Functions

激活函数使深度学习成为可能。神经网络实际上是一个具有许多参数的程序，用于模拟数学函数。如果我们所做的只是重复使用层权重的多个张量，我们只能模拟线性函数；此外，拥有许多层是没有意义的，因为整个网络将减少到单个矩阵乘法。在层之间插入非线性激活函数使得深度学习模型能够模拟任何函数，而不仅仅是线性函数。

`torch.nn.Module` 具有封装所有主要激活函数的对象，包括 ReLU 及其许多变体、Tanh、Hardtanh、sigmoid 等。它还包括其他函数，例如 Softmax，它们在模型的输出阶段最有用。

### Loss Functions 

损失函数告诉我们模型的预测与正确答案的差距有多大。 PyTorch 包含多种损失函数，包括常见的 MSE（均方误差 = L2 范数）、交叉熵损失和负似然损失（对分类器有用）等。
