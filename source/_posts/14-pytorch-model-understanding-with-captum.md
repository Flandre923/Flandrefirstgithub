---
title: 14-pytorch-model understanding with captum
date: 2023-11-15 15:28:20
tags:
- 深度学习
- pytorch
cover: https://view.moezx.cc/images/2017/11/25/_45337261_0.jpg
---

# MODEL UNDERSTANDING WITH CAPTUM 

Captum（拉丁语中的“理解”）是一个开源、可扩展的库，用于构建在 PyTorch 上的模型可解释性。

随着模型复杂性的增加以及由此导致的透明度的缺乏，模型可解释性方法变得越来越重要。模型理解既是一个活跃的研究领域，也是使用机器学习跨行业实际应用的重点领域。 Captum 提供最先进的算法，包括集成梯度，为研究人员和开发人员提供一种简单的方法来了解哪些特征有助于模型的输出。

captum.ai 网站上提供了完整的文档、API 参考以及针对特定主题的一套教程。

## Introduction 

Captum 的模型可解释性方法是根据归因。 Captum 中提供三种归因：

- 特征归因**Feature Attribution** 试图根据生成特定输出的输入的特征来解释该输出。根据评论中的某些单词来解释电影评论是正面还是负面，就是特征归因的一个例子。
- 层归因**Layer Attribution** 检查特定输入后模型隐藏层的活动。在层属性示例中检查卷积层响应于输入图像的空间映射输出。
- 神经元**Neuron Attribution**归因与层归因类似，但侧重于单个神经元的活动。

在这个交互式笔记本中，我们将了解特征归因和图层归因。

三种归因类型中的每一种都有多种与其关联的归因算法。许多归因算法分为两大类：

- 基于梯度的算法计算模型输出、层输出或神经元激活相对于输入的后向梯度。积分梯度（针对特征）、层梯度*激活和神经元电导都是基于梯度的算法。
- 基于扰动的算法检查模型、层或神经元的输出随输入变化的变化。输入扰动可以是定向的或随机的。遮挡、特征消融和特征排列都是基于扰动的算法。

我们将在下面研究这两种类型的算法。

特别是在涉及大型模型的情况下，以易于将归因数据与正在检查的输入特征相关联的方式可视化归因数据可能很有价值。虽然当然可以使用 Matplotlib、Plotly 或类似工具创建您自己的可视化，但 Captum 提供了特定于其属性的增强工具：

- `captum.attr.visualization` 模块（下面导入为 `viz` ）提供了有用的功能来可视化与图像相关的属性。
- Captum Insights 是 Captum 之上的一个易于使用的 API，它提供了一个可视化小部件，其中包含针对图像、文本和任意模型类型的现成可视化效果。

这两个可视化工具集都将在本笔记本中演示。前几个示例将重点关注计算机视觉用例，但最后的 Captum Insights 部分将演示多模型、视觉问答模型中归因的可视化。

## Installation 

在开始之前，您需要有一个 Python 环境：

- Python 版本 3.6 或更高版本
- 对于 Captum Insights 示例，Flask 1.1 或更高版本以及 Flask-Compress（建议使用最新版本）
- PyTorch 1.2 或更高版本（推荐最新版本）
- TorchVision 0.6 或更高版本（推荐最新版本）
- Captum（推荐最新版本）
- Matplotlib 版本 3.3.4，因为 Captum 当前使用 Matplotlib 函数，其参数已在更高版本中重命名

要在 Anaconda 或 pip 虚拟环境中安装 Captum，请使用以下适合您环境的命令：

With `conda`:

```python
conda install pytorch torchvision captum flask-compress matplotlib=3.3.4 -c pytorch

```

With `pip`: 

```python
pip install torch torchvision captum matplotlib==3.3.4 Flask-Compress
```

## A First Example 

首先，让我们举一个简单、直观的例子。我们将从在 ImageNet 数据集上预训练的 ResNet 模型开始。我们将获得测试输入，并使用不同的特征归因算法来检查输入图像如何影响输出，并查看一些测试图像的输入归因图的有用可视化。

首先，一些导入库：

```python
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

import os, sys
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
```

现在我们将使用 TorchVision 模型库下载预训练的 ResNet。由于我们没有进行训练，因此我们现在将其置于评估模式。

```python
model = models.resnet18(weights='IMAGENET1K_V1')
model = model.eval()
```

您获得此交互式笔记本的位置还应该有一个 `img` 文件夹，其中包含文件 `cat.jpg` 。

```python
test_img = Image.open('img/cat.jpg')
test_img_data = np.asarray(test_img)
plt.imshow(test_img_data)
plt.show()
```

我们的 ResNet 模型是在 ImageNet 数据集上进行训练的，并期望图像具有一定的大小，通道数据标准化为特定的值范围。我们还将提取我们的模型识别的类别的人类可读标签列表 - 该列表也应该位于 `img` 文件夹中。

```python
# model expects 224x224 3-color image
transform = transforms.Compose([
 transforms.Resize(224),
 transforms.CenterCrop(224),
 transforms.ToTensor()
])

# standard ImageNet normalization
transform_normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )

transformed_img = transform(test_img)
input_img = transform_normalize(transformed_img)
input_img = input_img.unsqueeze(0) # the model requires a dummy batch dimension

labels_path = 'img/imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)
```

现在，我们可以问一个问题：我们的模型认为这张图像代表什么？

```python
output = model(input_img)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)
pred_label_idx.squeeze_()
predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')
```

我们已经确认 ResNet 认为我们的猫图像实际上就是一只猫。但为什么模型认为这是猫的图像呢？

为了找到这个问题的答案，我们求助于 Captum。

## Feature Attribution with Integrated Gradients 

特征归因将特定输出归因于输入的特征。它使用特定的输入（这里是我们的测试图像）来生成每个输入特征与特定输出特征的相对重要性的地图。

集成梯度是 Captum 中可用的特征归因算法之一。积分梯度通过近似模型输出相对于输入的梯度积分，为每个输入特征分配重要性分数。

在我们的例子中，我们将采用输出向量的特定元素 - 即指示模型对其所选类别的置信度的元素 - 并使用积分梯度来了解输入图像的哪些部分对该输出做出了贡献。

一旦我们从积分梯度中获得重要性图，我们将使用 Captum 中的可视化工具来提供重要性图的有用表示。 Captum 的 `visualize_image_attr()` 功能提供了多种用于自定义归因数据显示的选项。在这里，我们传入一个自定义的 Matplotlib 颜色图。

使用 `integrated_gradients.attribute()` 调用运行单元通常需要一两分钟。

```python
# Initialize the attribution algorithm with the model
integrated_gradients = IntegratedGradients(model)

# Ask the algorithm to attribute our output target to
attributions_ig = integrated_gradients.attribute(input_img, target=pred_label_idx, n_steps=200)

# Show the original image for comparison
_ = viz.visualize_image_attr(None, np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                      method="original_image", title="Original Image")

default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#0000ff'),
                                                  (1, '#0000ff')], N=256)

_ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive',
                             title='Integrated Gradients')
```

在上图中，您应该看到积分梯度为我们提供了图像中猫位置周围最强的信号。

## Feature Attribution with Occlusion 

基于梯度的归因方法有助于通过直接计算出相对于输入的输出变化来理解模型。基于扰动的归因方法通过引入输入变化来衡量对输出的影响，更直接地解决这个问题。闭塞就是这样一种方法。它涉及替换输入图像的部分，并检查对输出信号的影响。

下面，我们设置遮挡归因。与配置卷积神经网络类似，您可以指定目标区域的大小和步幅长度来确定各个测量的间距。我们将使用 `visualize_image_attr_multiple()` 可视化遮挡归因的输出，按区域显示正面和负面归因的热图，并使用正面归因区域掩盖原始图像。遮罩提供了一个非常有启发性的视图，让我们了解模型发现猫照片中的哪些区域最“像猫”。

```python
occlusion = Occlusion(model)

attributions_occ = occlusion.attribute(input_img,
                                       target=pred_label_idx,
                                       strides=(3, 8, 8),
                                       sliding_window_shapes=(3,15, 15),
                                       baselines=0)


_ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map", "heat_map", "masked_image"],
                                      ["all", "positive", "negative", "positive"],
                                      show_colorbar=True,
                                      titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                      fig_size=(18, 6)
                                     )
```

我们再次看到包含猫的图像区域具有更大的重要性。

## Layer Attribution with Layer GradCAM 

层归因允许您将模型中隐藏层的活动归因于输入的特征。下面，我们将使用层归因算法来检查模型中卷积层之一的活动。

GradCAM 计算目标输出相对于给定层的梯度、每个输出通道（输出的维度 2）的平均值，并将每个通道的平均梯度乘以层激活。将所有通道的结果相加。 GradCAM 是为卷积网络设计的；由于卷积层的活动通常在空间上映射到输入，因此 GradCAM 属性通常会被上采样并用于屏蔽输入。

层归因的设置与输入归因类似，不同之处在于除了模型之外，您还必须在模型中指定要检查的隐藏层。如上所述，当我们调用 `attribute()` 时，我们指定感兴趣的目标类。

```python
layer_gradcam = LayerGradCam(model, model.layer3[1].conv2)
attributions_lgc = layer_gradcam.attribute(input_img, target=pred_label_idx)

_ = viz.visualize_image_attr(attributions_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                             sign="all",
                             title="Layer 3 Block 1 Conv 2")
```

我们将使用 LayerAttribution 基类中的便捷方法 `interpolate()` 对该属性数据进行上采样，以便与输入图像进行比较。

```python
upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, input_img.shape[2:])

print(attributions_lgc.shape)
print(upsamp_attr_lgc.shape)
print(input_img.shape)

_ = viz.visualize_image_attr_multiple(upsamp_attr_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                                      transformed_img.permute(1,2,0).numpy(),
                                      ["original_image","blended_heat_map","masked_image"],
                                      ["all","positive","positive"],
                                      show_colorbar=True,
                                      titles=["Original", "Positive Attribution", "Masked"],
                                      fig_size=(18, 6))
```



诸如此类的可视化可以让您对隐藏层如何响应输入有新的见解。

## Visualization with Captum Insights 

Captum Insights 是一个构建在 Captum 之上的可解释性可视化小部件，旨在促进模型理解。 Captum Insights 跨图像、文本和其他特征工作，帮助用户了解特征归因。它允许您可视化多个输入/输出对的归因，并提供图像、文本和任意数据的可视化工具。

在笔记本的这一部分中，我们将使用 Captum Insights 可视化多个图像分类推理。

首先，让我们收集一些图像，看看模型对它们的看法。为了增加多样性，我们将带上我们的猫、茶壶和三叶虫化石：

```python
imgs = ['img/cat.jpg', 'img/teapot.jpg', 'img/trilobite.jpg']

for img in imgs:
    img = Image.open(img)
    transformed_img = transform(img)
    input_img = transform_normalize(transformed_img)
    input_img = input_img.unsqueeze(0) # the model requires a dummy batch dimension

    output = model(input_img)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()
    predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    print('Predicted:', predicted_label, '/', pred_label_idx.item(), ' (', prediction_score.squeeze().item(), ')')
```

......看起来我们的模型正确地识别了它们 - 但当然，我们想更深入地挖掘。为此，我们将使用 Captum Insights 小部件，我们使用下面导入的 `AttributionVisualizer` 对象对其进行配置。 `AttributionVisualizer` 需要批量数据，因此我们将引入 Captum 的 `Batch` 帮助器类。我们将专门查看图像，因此我们还将导入 `ImageFeature` 。

我们使用以下参数配置 `AttributionVisualizer` ：

- 一系列要检查的模型（在我们的例子中，只有一个）
- 评分函数，允许 Captum Insights 从模型中提取前 k 个预测
- 我们的模型所训练的类的有序的、人类可读的列表
- 要查找的功能列表 - 在我们的例子中是 `ImageFeature`
- 数据集，它是一个可迭代对象，返回批量输入和标签 - 就像您用于训练一样

```python
from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

# Baseline is all-zeros input - this may differ depending on your data
def baseline_func(input):
    return input * 0

# merging our image transforms from above
def full_img_transform(input):
    i = Image.open(input)
    i = transform(i)
    i = transform_normalize(i)
    i = i.unsqueeze(0)
    return i


input_imgs = torch.cat(list(map(lambda i: full_img_transform(i), imgs)), 0)

visualizer = AttributionVisualizer(
    models=[model],
    score_func=lambda o: torch.nn.functional.softmax(o, 1),
    classes=list(map(lambda k: idx_to_labels[k][1], idx_to_labels.keys())),
    features=[
        ImageFeature(
            "Photo",
            baseline_transforms=[baseline_func],
            input_transforms=[],
        )
    ],
    dataset=[Batch(input_imgs, labels=[282,849,69])]
)
```

请注意，与我们上面的归因不同，运行上面的单元格根本不需要太多时间。这是因为 Captum Insights 允许您在可视化小部件中配置不同的归因算法，然后它将计算并显示归因。该过程将需要几分钟。

运行下面的单元格将呈现 Captum Insights 小部件。然后，您可以选择归因方法及其参数，根据预测类别或预测正确性过滤模型响应，查看模型的预测以及相关概率，并查看与原始图像相比的归因热图。

```python
visualizer.render()
```

