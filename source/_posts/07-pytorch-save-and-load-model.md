---
title: 07-pytorch-save and load model
date: 2023-11-12 09:44:40
tags:
- 深度学习
- save load model
cover: https://w.wallhaven.cc/full/p9/wallhaven-p9m1j9.jpg
---

# SAVE AND LOAD THE MODEL 

在本节中，我们将了解如何通过保存、加载和运行模型预测来持久保存模型状态。

```python
import torch
import torchvision.models as models
```

## Saving and Loading Model Weights 

PyTorch 模型将学习到的参数存储在内部状态字典中，称为 `state_dict` 。这些可以通过 `torch.save` 方法保存：

```python
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```

要加载模型权重，您需要先创建同一模型的实例，然后使用 `load_state_dict()` 方法加载参数。

```python
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

> 请务必在推理之前调用 `model.eval()` 方法，将 dropout 和批量归一化层设置为评估模式。如果不这样做将会产生不一致的推理结果。

## Saving and Loading Models with Shapes 

加载模型权重时，我们需要首先实例化模型类，因为该类定义了网络的结构。我们可能希望将此类的结构与模型一起保存，在这种情况下，我们可以将 `model` （而不是 `model.state_dict()` ）传递给保存函数：

```python
torch.save(model, 'model.pth')
```

然后我们可以像这样加载模型：

```python
model = torch.load('model.pth')
```

> 此方法在序列化模型时使用 Python pickle 模块，因此它依赖于加载模型时可用的实际类定义。
