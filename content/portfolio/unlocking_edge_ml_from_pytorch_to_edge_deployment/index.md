---
title: "Unlocking Edge ML: From PyTorch to Edge Deployment"
date: 2024-08-30T11:00:15+08:00
featureImage: images/portfolio/unlocking_edge_ml_from_pytorch_to_edge_deployment/thumbnail.png
postImage: images/portfolio/unlocking_edge_ml_from_pytorch_to_edge_deployment/post_image.png
tags: ["TIMM", "PyTorch", "Hugging Face", "ONNX", "Torchscript", "Netron"]
categories: ["optimization", "edge-deployment", "computer-vision", "real-time"]
toc: true
socialshare: true
description: "Learn how to convert PyTorch Image Models (TIMM) to ONNX format for efficient edge deployment. This step-by-step guide covers model optimization techniques, code examples, and troubleshooting tips, making it a go-to resource for machine learning practitioners looking to deploy models on edge devices."
images : 
- images/portfolio/unlocking_edge_ml_from_pytorch_to_edge_deployment/post_image.jpg
---
{{< notice info >}}
This blog post is still a work in progress. If you require further clarifications before the contents are finalized, please get in touch with me [here](https://dicksonneoh.com/contact/), on [LinkedIn](https://www.linkedin.com/in/dickson-neoh/), or [Twitter](https://twitter.com/dicksonneoh7).
{{< /notice >}}

### 🚀 Motivation - Edge Deployment
It's 2024, everyone seems to be talking about complex and larger models. 

Sophisticated models perform well at specific tasks. But they come with the cost of massive computational power. 

Typically that's available in cloud-based environments.
Cloud-based environments has limitations, such as latency, bandwidth constraints, and privacy concerns.

This is when edge deployment comes into play.

{{% blockquote %}}
In simple terms, edge deployment means running a model close to the source of the data. For example running a face recognition model on an IPhone. 
{{% /blockquote %}}

Why edge deployment:

1. **Low Latency**: Edge devices process data locally. This reduces the time it takes for a model to produce an output.
  
2. **Privacy**: Your data stays on the device. This reduces the risk of data breaches and better compliance with data privacy regulations.

3. **Robustness**: Edge devices can function with or without an internet connection. This provides reliability and robustness.

{{< notice note >}}
But, there's a caveat - Edge devices often have limited computational resources. 
{{< /notice >}}

This is why large models typically go through optimizations before it is deployed on edge devices. In this blog post, we'll look into ONNX, OpenVINO and TFlite - some of the most popular form of deployement format.


In this post, you'll learn how to convert PyTorch Image Models (TIMM) into ONNX format, a crucial step in preparing your models for efficient edge deployment.

{{< notice tip >}}
By the end of this post you'll learn how to
+ Load any model from TIMM.
+ Convert the model into ONNX, OpenVINO and TFlite format.
+ Optmize the model to improve inference latency.

The codes for this post are on my [GitHub repo](https://github.com/dnth/from-pytorch-to-edge-deployment-blogpost).
{{< /notice >}}

But first, let's load a PyTorch computer vision model from TIMM.

### 🖼️ Torch Image Models (TIMM)

TIMM, or [Torch Image Models](https://huggingface.co/docs/timm/quickstart), is a Python library that provides a collection of pre-trained machine learning models specifically designed for computer vision tasks.

To date, TIMM provides more than 1000 state-of-the-art computer vision models trained on various datasets.
Many state-of-the-art models are also build using TIMM. 

Install TIMM by running:

```shell
pip install timm
```

I'm using version `timm==0.9.7` in the post.

Presently there are 1212 models on timm as listed on [Hugging Face](https://huggingface.co/timm).

{{< figure_autoresize src="timm_hf.png" caption="Over a thousand pre-trained models on TIMM." >}}


Once installed load any model with 2 lines of code:

```python
import timm
model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', 
                          pretrained=True)

```

Now, put model in evaluation mode for inference.

```python
model = model.eval()
```

Next let's load an image from the web.

```python
from urllib.request import urlopen
from PIL import Image

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

img
```
 
{{< figure_autoresize src="image_from_web.png" width="400" align="center" >}}


Next let's get the model's specific transforms

```python
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
```

With the right transforms we can run an inference on the downloaded image.

```python
output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1
```

And view the results
```python
top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

top5_probabilities
>>> tensor([[12.4517,  8.8304,  5.8010,  3.0997,  3.0730]], grad_fn=<TopkBackward0>)

top5_class_indices
>>> tensor([[968, 967, 969, 960, 504]])

output.shape
>>> torch.Size([1, 1000])
```

To view the class names we load the ImageNet classe names with the corresponding index from the inference results.

```python
from imagenet_classes import IMAGENET2012_CLASSES

# Retrieving class names
im_classes = list(IMAGENET2012_CLASSES.values())
class_names = [im_classes[i] for i in top5_class_indices[0]]

class_names
>>> ['cup', 'espresso', 'eggnog', 'chocolate sauce, chocolate syrup', 'coffee mug']
```

Now let's measure the inference time on CPU.

```python
import time
num_images = 100

with torch.inference_mode():
    start = time.perf_counter()
    for _ in range(num_images):
        model(transforms(img).unsqueeze(0)) 
    end = time.perf_counter()
    time_taken = end - start
```

```python
print(
    f"PyTorch model on CPU: {time_taken/num_images*1000:.3f} ms per image,\n"
    f"FPS: {num_images/time_taken:.2f}")
```

```bash
>>> PyTorch model on CPU: 109.419 ms per image,
>>> FPS: 9.14

```

There we have a baseline of **9.14** FPS on a pure PyTorch model.

{{< notice warning >}}
Although the inference time is not bad, deploying PyTorch models directly into production environments is often not ideal:

1. **Large Dependency:** PyTorch requires numerous dependencies, challenging for resource-constrained environments.

2. **Deployment Complexity:** Packaging PyTorch models with dependencies is complex in containerized/serverless environments.

3. **Version Compatibility:** Ensuring PyTorch version compatibility between development and production can be difficult.

4. **Resource Usage:** PyTorch models often consume more memory and computational resources than optimized formats.

5. **Platform Limitations:** Some platforms or edge devices may not support PyTorch natively.

{{< /notice >}}
These factors often lead practitioners to convert PyTorch models to more deployment-friendly formats like ONNX.

### 🏆 ONNX (Open Neural Network Exchange)

[ONNX](https://github.com/onnx/onnx) is an open format built to represent machine learning models. 
It defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.

With ONNX, you can train a machine learning model in one framework (e.g. PyTorch) use the trained model in another (e.g. Tensorflow)

{{< notice note >}}
💫 In short, ONNX offers two benefits that helps edge deployment:

+ **Interoperability** - Develop in your preferred framework and not worry about deployment contranints.
+ **Hardware access** - ONNX compatible runtimes can maximize performance across hardware.
{{< /notice >}}

So let's now convert our PyTorch model to ONNX format.

### 🔁 PyTorch to ONNX

Since we loaded our model from TIMM, we can use the `timm` library utils to export the model to ONNX.
Before that, make sure onnx is installed.

```bash
pip install onnx
```

```bash
python onnx_export.py convnextv2_base.fcmae_ft_in22k_in1k.onnx \
    --model timm/convnextv2_base.fcmae_ft_in22k_in1k \
    --opset 16 \
    --reparam \
```

Sometimes, the above command might not work. In that case, we can manually reparameterize the model and export it to ONNX.

```python
from timm.utils.model import reparameterize_model
model = reparameterize_model(model)
```

Reparameterizing the model reduces/combines the number of parameters in the model, which can help improve the inference speed.

Once done, we can export the model using PyTorch's built-in `torch.onnx.export` function.

```python
import torch.onnx
torch.onnx.export(
    model, # PyTorch model
    torch.rand(1, 3, 224, 224, requires_grad=True), # dummy input
    "convnextv2_base.fcmae_ft_in22k_in1k.onnx", # output file name
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'], 
    dynamic_axes={'input' : {0 : 'batch_size'},   
                  'output' : {0 : 'batch_size'}}
)
```

{{< notice tip >}}
Description of the parameters:
- `dynamic_axes`: Allows the ONNX model to accept inputs of different sizes. Usually, the first dimension of the input is the batch size and we want the model to be able to handle different batch sizes at inference time. With this, we can run inference on 1 image, 10 images, or 100 images without changing the model at inference time.

- `do_constant_folding=True` - This optimizes the model by folding constants, which can improve inference speed and reduce the size of the model.

More on `torch.onnx.export` [here](https://pytorch.org/docs/stable/onnx.html).

{{< /notice >}}


Sometimes the resulting ONNX file becomes unnecessarily complicated. We can simplify the converted ONNX model using a tool like [`onnx-simplifier`](https://github.com/daquexian/onnx-simplifier). 

{{< notice note >}}
This is not strictly necessary, but it may help reduce the size of the model and improve inference speed. 
{{< /notice >}}

Let's start by installing `onnx-simplifier`.

```bash
pip install onnxsim
```

Run the following CLI command to simplify the ONNX model by specifying the input and output file names.

```bash
onnxsim convnextv2_base.fcmae_ft_in22k_in1k.onnx \
        convnextv2_base.fcmae_ft_in22k_in1k_simplified.onnx
```


The output will show the difference between the original and simplified model. 

{{< figure_autoresize src="onnxsim.png" width="500" align="center" caption="The difference between the original and simplified model." >}}

Looks like the simplified model has fewer `Constant` and `Mul` operations but the model size remains the same. The result is a new file `convnextv2_base.fcmae_ft_in22k_in1k_simplified.onnx`.

To run an inference in ONNX, install `onnxruntime`:

```shell
pip install onnxruntime
```


Now let's load the simplified ONNX model and run an inference using `onnxruntime`.

`onnxruntime` can run on CPU, GPU, or other hardware accelerators. For the sake of simplicity, we'll run the inference on CPU. After all, majority of edge devices are CPU-based.

```python
import numpy as np
import onnxruntime as ort
from PIL import Image
from urllib.request import urlopen

# Load an image
img = img.convert('RGB')
img = img.resize((224, 224))
img_np = np.array(img).astype(np.float32)

# Load ONNX model
session = ort.InferenceSession(
                "convnextv2_base.fcmae_ft_in22k_in1k_simplified.onnx", 
                providers=['CPUExecutionProvider'])

# Convert data to the shape the ONNX model expects
input_data = np.transpose(img_np, (2, 0, 1))  # Convert to (C, H, W)
input_data = np.expand_dims(input_data, axis=0)  # Add a batch dimension

# Get input name from the model
input_name = session.get_inputs()[0].name
```

Let's run an inference and measure the time taken.

```python
import time
num_images = 100

start = time.perf_counter()
for _ in range(num_images):
    session.run(None, {input_name: input_data})
end = time.perf_counter()
time_taken = end - start

print(
    f"ONNX model on CPU: {time_taken/num_images*1000:.3f} ms per image,\n"
    f"FPS: {num_images/time_taken:.2f}")
```

```bash
>>> ONNX model on CPU: 71.991 ms per image,
>>> FPS: 13.89
```
Not bad! We went from **9.14** FPS to **13.89** FPS!

Plus we don't need to worry about installing PyTorch anymore on the inference device. All we need is the ONNX file and `onnxruntime`. This is way more portable!

### 📜 PyTorch to Torchscript

Other than ONNX, Torchscript is another format to prepare models for deployment. Both share a common goal - to make the model more efficient for inference.

Let's convert our PyTorch model to Torchscript and run an inference.

```python
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("torchscript_edgenext_xx_small.pt")
```

### 🪜 ONNX to OpenVINO

```python
import openvino as ov
ov_model = ov.convert_model('dv2s_redo_simplified.onnx')

###### Option 1: Save to OpenVINO IR:

# save model to OpenVINO IR for later use
ov.save_model(ov_model, 'dv2s_redo_simplified.xml')

###### Option 2: Compile and infer with OpenVINO:

# compile model
compiled_model = ov.compile_model(ov_model)

# prepare input_data
import numpy as np
input_data = np.random.rand(1, 3, 224, 224)

# run inference
result = compiled_model(input_data)
```

### 💫 ONNX to TFlite with onnx2tf
[onnx2tf](https://github.com/PINTO0309/onnx2tf) is a tool to convert ONNX files (NCHW) to TensorFlow/TFLite/Keras format (NHWC).

### 💥 PyTorch to OpenVINO

```python
import openvino.torch
model = torch.compile(model, backend='openvino')
# OR
model = torch.compile(model, backend='openvino_ts')
```
{{< notice note >}}

+ `openvino` -
With this backend, Torch FX subgraphs are directly converted to OpenVINO representation without any additional PyTorch based tracing/scripting.
+ `openvino_ts` -
With this backend, Torch FX subgraphs are first traced/scripted with PyTorch Torchscript, and then converted to OpenVINO representation.
{{< /notice >}}


```python
import openvino as ov

# Create OpenVINO Core object instance
core = ov.Core()

# Convert model to openvino.runtime.Model object
ov_model = ov.convert_model(model)

MODEL_NAME = "DINOV2S"
# Save openvino.runtime.Model object on disk
ov.save_model(ov_model, f"{MODEL_NAME}_dynamic.xml")

# Load OpenVINO model on device
compiled_model = core.compile_model(ov_model, 'AUTO')

input_tensor=transforms(img).unsqueeze(0)
result = compiled_model(input_tensor)[0]
```
### 🏁 Wrap Up
