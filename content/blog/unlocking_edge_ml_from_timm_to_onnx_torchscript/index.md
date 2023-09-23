---
title: "Unlocking Edge ML: From PyTorch Image Models (TIMM) to ONNX/Torchscript"
date: 2023-09-16T11:00:15+08:00
featureImage: images/blog/unlocking_edge_ml_from_timm_to_onnx_torchscript/thumbnail.jpg
postImage: images/blog/unlocking_edge_ml_from_timm_to_onnx_torchscript/post_image.jpg
tags: ["TIMM", "PyTorch", "Hugging Face", "ONNX", "Torchscript", "Netron"]
categories: ["optimization", "edge-deployment", "computer-vision", "real-time"]
toc: true
socialshare: true
description: "Learn how to convert PyTorch Image Models (TIMM) to ONNX format for efficient edge deployment. This step-by-step guide covers model optimization techniques, code examples, and troubleshooting tips, making it a go-to resource for machine learning practitioners looking to deploy models on edge devices."
images : 
- images/blog/unlocking_edge_ml_from_timm_to_onnx_torchscript/post_image.jpg
---

### 🚀 Motivation: Why Edge Deployment?
It's late 2023, everyone seems to be talking about complex and larger models.

Sophisticated models perform well at specific tasks. But they come with the cost of massive computational power. And typically that's available only in cloud-based environments.
Cloud-based environments come with limitations, such as latency, bandwidth constraints, and sometimes even privacy concerns.

This is when edge deployment comes into play.

{{% blockquote %}}
In simple terms, edge deployment means running a model close to the source of the data. For example running a face recognition model on an IPhone. 
{{% /blockquote %}}

Some advantage of edge deployment:

1. **Low Latency**: Edge devices process data locally. This reduces the time it takes to receive a response compared to sending a query to the cloud.
  
2. **Reduced Bandwidth**: By processing data on the device, you minimize the amount of data that needs to be sent back and forth between the cloud and the edge device. Useful in scenarios where you have limited network connectivity.
  
3. **Privacy and Security**: Your data stays on the device. This reduces the risk of data breaches and better compliance with data privacy regulations.

But, edge devices often have limited computational resources. 
 
This is why model optimization techniques, such as converting models to ONNX format, are crucial. 

They allow us to compress the model size without a significant loss in performance, making it feasible to deploy advanced machine learning models on edge devices.

In this guide, you'll learn how to convert PyTorch Image Models (TIMM) into ONNX format, a crucial step in preparing your models for efficient edge deployment.


### 🖼️ Torch Image Models (TIMM)

```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'vit_small_patch14_dinov2.lvd142m',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

# or equivalently (without needing to set num_classes=0)

output = model.forward_features(transforms(img).unsqueeze(0))
# output is unpooled, a (1, 1370, 384) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor

```

Output shape is `torch.Size([1, 384])`.



### 🔁 Convert to ONNX

Using TIMM export

```bash
python onnx_export.py vit_base_patch14_dinov2.lvd142m.onnx --model timm/vit_base_patch14_dinov2.lvd142m --opset 10 --num-classes 0 --reparam --verbose
```

Using torch export

```python
from timm.utils.model import reparameterize_model
model = reparameterize_model(model)
```

```python
import torch.onnx
torch.onnx.export(model,
                 torch.rand(1, 3, 518, 518, requires_grad=True),
                 "vit_small_patch14_dinov2.lvd142m.onnx",
                 export_params=True,
                 opset_version=16,
                 do_constant_folding=True,
                 input_names=['input'],
                 output_names=['output'], 
                 dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                               'output' : {0 : 'batch_size'}}
)

```

```bash
pip install onnxsim -Uq
onnxsim efficientvit_m5.r224_in1k.onnx efficientvit_m5.r224_in1k_simplified.onnx
```

### 👁️ Visualize Model with Netron
Netron.app

### 📜 Convert to Torchscript

```python
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("torchscript_edgenext_xx_small.pt")
```


### 🏁 Wrap Up
