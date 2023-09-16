---
title: "Convert Any Model from PyTorch Image Models (TIMM) into ONNX"
date: 2023-09-16T11:00:15+08:00
featureImage: images/blog/convert_timm_onnx/thumbnail.jpg
postImage: images/blog/convert_timm_onnx/post_image.jpg
tags: ["TIMM", "PyTorch", "Hugging Face"]
categories: ["optimization", "edge-ML", "computer-vision"]
toc: true
socialshare: true
description: "Deploy large models into small devices."
images : 
- images/blog/convert_timm_onnx/post_image.jpg
---

### ✅ Motivation
Making models smaller and more efficient for edge deployment.

### Torch Image Model (TIMM)

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



### Convert to ONNX

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

### Visualize Model with Netron
Netron.app

### Convert to Torchscript

```python
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("torchscript_edgenext_xx_small.pt")
```