---
title: "Supercharge Your PyTorch Image Models: Bag of Tricks to 8x Faster Inference with ONNX Runtime & Optimizations"
date: 2024-09-09T09:00:00+08:00
featureImage: images/portfolio/supercharge_your_pytorch_image_models/thumbnail.gif
postImage: images/portfolio/supercharge_your_pytorch_image_models/post_image.png
tags: ["TIMM", "ONNX", "TensorRT", "ImageNet"]
categories: ["inference", "deployment", "Image-Classification"]
toc: true
socialshare: true
description: "Learn how to accelerate TIMM model inference up to 84x faster using ONNX Runtime and TensorRT optimization techniques!"
images: 
- images/portfolio/supercharge_your_pytorch_image_models/post_image.jpg
---

### 🚀 Motivation
Real time inference speed is crucial for many applications in production. Some could mean life or death. 💀

Imagine you're behind the wheels of a self-driving car and the car takes 1 second to detect an oncoming truck.

Just one second too late, and you could end up in the clouds, talking to celestial beings... 👼👼👼

Or if you're lucky, on the ground.



{{< figure src="banana_peel_robot.gif" width="480" align="center" >}}

I hope that shows you how crucial this problem is.

Today (2024), ML models are being deployed in all kinds of high-stakes industries like healthcare, finance, and self-driving cars.

{{% blockquote %}}
It's not just about being right - it's about being right, right now.
{{% /blockquote %}}

This post shows how you can bring any models from [TIMM](https://huggingface.co/docs/timm/index) and supercharge its inference speed with optimized [ONNX Runtime](https://onnxruntime.ai/) and [TensorRT](https://developer.nvidia.com/tensorrt).

{{< notice tip >}}
By the end of this post you'll learn how to:
- 📥 Load any pre-trained model from [TIMM](https://huggingface.co/docs/timm/index)
- 🔄 Convert the model to ONNX format
- 🖥️ Run inference with ONNX Runtime (CPU & GPU)
- 🎮 Run inference with TensorRT (GPU)
- 🛠️ Tweak the TensorRT parameters for better performance
- 🧠 Bake the pre-processing into the ONNX model

You can find the code for this post on my GitHub repository [here](https://github.com/dnth/timm_onnx_tensort).

{{< /notice >}}

If that sounds exciting let's dive in! 🏊‍♂️

### 💻 Installation
I will be using the conda environment to install the packages. Feel free to use any other environment of your choice.

```bash
conda create -n supercharge_timm_tensorrt python=3.11
conda activate supercharge_timm_tensorrt
```

In this post I will be using the [`timm`](https://github.com/huggingface/pytorch-image-models) library to load a pre-trained model and run inference. So let's install `timm`.

```bash
pip install timm
```
If you're not familiar, `timm` is a library that provides thousands of pre-trained models that's being used in research and production.
If you've used a PyTorch vision model, chances are it's using a model from `timm`.

### 🔧 Load and Infer
Let's load one of the top performing models from the timm [leaderboard](https://huggingface.co/spaces/timm/leaderboard) - the `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` model. 

This model boasts impressive ImageNet accuracy scores of **90.05%** for top-1 and **99.06%** for top-5 classifications.

<!-- <iframe
	src="https://timm-leaderboard.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe> -->


```python
import timm

model_name = 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'
model = timm.create_model(model_name, pretrained=True).eval()
```

Next, we need to get the data config and transformations for the model.


```python
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
```
With the model and transformations ready, let's run inference to get the top 5 predictions.

```python
import torch
from PIL import Image
from urllib.request import urlopen

img = Image.open(urlopen('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'))

with torch.inference_mode():
    output = model(transforms(img).unsqueeze(0))

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
```

Get the top 5 predictions and print them.

```python
from imagenet_classes import IMAGENET2012_CLASSES

im_classes = list(IMAGENET2012_CLASSES.values())
class_names = [im_classes[i] for i in top5_class_indices[0]]

for name, prob in zip(class_names, top5_probabilities[0]):
    print(f"{name}: {prob:.2f}%")
```

{{< figure_autoresize src="beignets-task-guide.png" width="400" align="center" >}}

Top 5 predictions:
```
>>> espresso: 26.78%
>>> eggnog: 2.88%
>>> cup: 2.60%
>>> chocolate sauce, chocolate syrup: 2.39%
>>> bakery, bakeshop, bakehouse: 1.48%
```
The predictions looks good! Now let's benchmark the model inference latency.

{{< notice note >}}
You can find the code for this section on my GitHub repository [here](https://github.com/dnth/timm_onnx_tensorrt/blob/main/00_benchmark_timm.py).
{{< /notice >}}

### ⏱️ PyTorch Latency Benchmark

We will run the inference 10 times and record the average time on both CPU and GPU.

```python
import time

def run_benchmark(model, device, num_images=10):
    model = model.to(device)
    
    with torch.inference_mode():
        start = time.perf_counter()
        for _ in range(num_images):
            input_tensor = transforms(img).unsqueeze(0).to(device)
            model(input_tensor)
        end = time.perf_counter()
    
    ms_per_image = (end - start) / num_images * 1000
    fps = num_images / (end - start)
    
    print(f"PyTorch model on {device}: {ms_per_image:.3f} ms per image, FPS: {fps:.2f}")
```


```python
# CPU Benchmark
run_benchmark(model, torch.device("cpu"))

# GPU Benchmark 
if torch.cuda.is_available():
    run_benchmark(model, torch.device("cuda"))
```
Alright the benchmarks are in
```
>>> PyTorch model on cpu: 1584.379 ms per image, FPS: 0.63
>>> PyTorch model on cuda: 77.226 ms per image, FPS: 12.95
```

{{< notice note >}}
I'm using the following hardware for the benchmarks:
- GPU: NVIDIA RTX 3090
- CPU: 11th Gen Intel® Core™ i9-11900 @ 2.50GHz × 16

You can find the code for the PyTorch benchmarks on my GitHub repository [here](https://github.com/dnth/timm_onnx_tensort/blob/main/01_pytorch_latency_benchmark.py).
{{< /notice >}}

Although the performance on the GPU is not bad, 12 FPS is still not fast enough for real-time inference.
Let's forget about using the model on a CPU for inference. Remember a one second too late could mean a lot!

But we can do better.

### 🔄 Convert to ONNX
ONNX is an open and interoperable format for deep learning models. It lets us deploy models across different frameworks and devices. 

As a bonus, ONNX Runtime can optimize the model for faster inference. Before we can use ONNX Runtime to run inference, we need to convert the model to ONNX format.

So let's first install `onnx` and run the conversion.

```bash
pip install onnx
```


```python
import timm
import torch

model = timm.create_model(
    "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", pretrained=True
).eval()

onnx_filename = "eva02_large_patch14_448.onnx"
torch.onnx.export(
    model,
    torch.randn(1, 3, 448, 448),
    onnx_filename,
    export_params=True,
    opset_version=20,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, 
                  "output": {0: "batch_size"}},
)

```

You will end up with a file called `eva02_large_patch14_448.onnx` in your working directory.


{{< notice note >}}
Parameters:
- `model`: The pre-trained model to be exported.
- `torch.randn(1, 3, 448, 448)`: A dummy input tensor with the appropriate shape.
- `"eva02_large_patch14_448.onnx"`: The name of the output ONNX file.
- `export_params=True`: Whether to export the model parameters.
- `opset_version=18`: The ONNX operator set version to use.
- `do_constant_folding=True`: Whether to do constant folding for optimization.
- `input_names=['input']`: The name of the input tensor.
- `output_names=['output']`: The name of the output tensor.
- `dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}`: Dynamic axes for the input and output tensors.

You can find the code for the ONNX conversion on my GitHub repository [here](https://github.com/dnth/timm_onnx_tensort/blob/main/03_convert_to_onnx.py).
{{< /notice >}}

### 🖥️ ONNX Runtime - CPU
Now that we have the ONNX model, let's run inference with ONNX Runtime on the CPU.

If you haven't installed `onnxruntime`, do so now.

```bash
pip install onnxruntime onnxruntime-gpu
```

Now that we have the ONNX model and the ONNX Runtime installed, let's run inference with ONNX Runtime on the CPU.



First, let's replicate the transforms from the PyTorch model using numpy.
If you print the transforms used in the PyTorch model, you can see that it's a sequence of transformations that converts the image to the appropriate shape and normalization for the model.

```python
print(transforms)
```

```
>>> Compose(
>>>     Resize(size=(448, 448), interpolation=bicubic, max_size=None, antialias=True)
>>>     CenterCrop(size=(448, 448))
>>>     MaybeToTensor()
>>>     Normalize(mean=tensor([0.4815, 0.4578, 0.4082]), std=tensor([0.2686, 0.2613, 0.2758]))
>>> )
```

The equivalent transforms in numpy is as follows:

```python
def transforms_numpy(image: Image.Image):
    image = image.convert('RGB')
    image = image.resize((448, 448), Image.BICUBIC)
    img_numpy = np.array(image).astype(np.float32) / 255.0
    img_numpy = img_numpy.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    img_numpy = (img_numpy - mean) / std
    img_numpy = np.expand_dims(img_numpy, axis=0)
    img_numpy = img_numpy.astype(np.float32)
    return img_numpy
```

Now let's run inference with ONNX Runtime.

```python
import onnxruntime as ort

# Create ONNX Runtime session with CPU provider
onnx_filename = "eva02_large_patch14_448.onnx"
session = ort.InferenceSession(
    onnx_filename, 
    providers=["CPUExecutionProvider"]
)

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
output = session.run(
    [output_name], 
    {input_name: transforms_numpy(img)}
)[0]

```
If we inspect the output shape, we can see that it's the same as the number of classes in the ImageNet dataset.

```
output.shape
>>> (1, 1000)
```

And the results are

```
>>> espresso: 28.65%
>>> cup: 2.77%
>>> eggnog: 2.28%
>>> chocolate sauce, chocolate syrup: 2.13%
>>> bakery, bakeshop, bakehouse: 1.42%
```

While the results aren't an exact match to the PyTorch model, they're sufficiently similar. This slight variation can be attributed to differences in how normalization is implemented, leading to minor discrepancies in the precise values.

{{< notice tip >}}
One of the benefits of using ONNX Runtime is we can get rid of the PyTorch dependency - which is a pain to install on some systems. Plus it's a huge dependency to have in your project.
{{< /notice >}}

{{< notice note >}}
You can find the code for the ONNX Runtime CPU inference on my GitHub repository [here](https://github.com/dnth/timm_onnx_tensort/blob/main/04_onnx_cpu_inference.py).
{{< /notice >}}


Now let's repeat the CPU inference benchmark but using ONNX Runtime.

```python
import time

num_images = 10
start = time.perf_counter()
for i in range(num_images):
    output = session.run([output_name], {input_name: transforms_numpy(img)})[0]
end = time.perf_counter()
time_taken = end - start

ms_per_image = time_taken / num_images * 1000
fps = num_images / time_taken

print(f"Onnxruntime CPU: {ms_per_image:.3f} ms per image, FPS: {fps:.2f}")
```

```
>>> Onnxruntime CPU: 2002.446 ms per image, FPS: 0.50
```

Ouch! That's slower than the PyTorch model. What a bummer!
We have to do better on the GPU. Let's try ONNX Runtime on the GPU.

### 🖼️ ONNX Runtime - GPU
ONNX Runtime offers other backends for inference. We can easily swap to a different backend by changing the provider.

```python
providers = ['CUDAExecutionProvider']
onnx_filename = "eva02_large_patch14_448.onnx"
session = ort.InferenceSession(onnx_filename, providers=["CUDAExecutionProvider"])
```
The rest of the code is the same as the CPU inference. 

Just with that change, the benchmarks are as follows:

```
>>> Onnxruntime CUDA numpy transforms: 56.430 ms per image, FPS: 17.72
```

Theres is one more trick we can use to squeeze out more performance - using [CuPy](https://cupy.dev/) for the transforms instead of numpy.

{{< notice tip >}}
CuPy is a library that lets us run NumPy code on the GPU. It's a drop-in replacement for NumPy, so you can just replace `numpy` with `cupy` in your code and it will run on the GPU.
{{< /notice >}}

To use CuPy, we need to install it first.

```bash
pip install cupy-cuda12x
```

And we can use it to run the transforms.
```python
def transforms_cupy(image: Image.Image):
    # Convert image to RGB and resize
    image = image.convert("RGB")
    image = image.resize((448, 448), Image.BICUBIC)

    # Convert to CuPy array and normalize
    img_cupy = cp.array(image, dtype=cp.float32) / 255.0
    img_cupy = img_cupy.transpose(2, 0, 1)

    # Apply mean and std normalization
    mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32).reshape(-1, 1, 1)
    std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32).reshape(-1, 1, 1)
    img_cupy = (img_cupy - mean) / std

    # Add batch dimension
    img_cupy = cp.expand_dims(img_cupy, axis=0)

    return img_cupy
```




With CuPy, we got a tiny bit of performance improvement:

```
>>> Onnxruntime CUDA cupy transforms: 54.267 ms per image, FPS: 18.43
```

{{< notice note >}}
You can find the code for the ONNX Runtime CUDA cupy inference on my GitHub repository [here](https://github.com/dnth/timm_onnx_tensort/blob/main/05_onnx_cuda_inference.py).
{{< /notice >}}

Using Onnx Runtime with CUDA is a little better than the PyTorch model on the GPU, but still not fast enough for real-time inference.

We have one more trick up our sleeve.

### 📊 ONNX Runtime - TensorRT
The TensorRT EP is a specialized provider for TensorRT. It lets us run the model with TensorRT optimizations.

Add in TensorRT parameters for final performance gains. 

Building on from the previous example, we can add in TensorRT parameters for final performance gains.

```python

providers = [
    (
        "TensorrtExecutionProvider",
        {
            "device_id": 0,
            "trt_max_workspace_size": 8589934592,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./trt_cache",
            "trt_force_sequential_engine_build": False,
            "trt_max_partition_iterations": 10000,
            "trt_min_subgraph_size": 1,
            "trt_builder_optimization_level": 5,
            "trt_timing_cache_enable": True,
        },
    ),
]

onnx_filename = "eva02_large_patch14_448.onnx"
session = ort.InferenceSession(onnx_filename, providers=providers)
```


And running the benchmark:
```
>>> Onnxruntime CUDA numpy transforms: 19.898 ms per image, FPS: 50.26
>>> Onnxruntime CUDA cupy transforms: 16.836 ms per image, FPS: 59.40
```

That's a 4x speedup over the PyTorch model on the GPU and 84x speedup over the PyTorch model on the CPU!

{{< notice note >}}
You can find the code for the ONNX Runtime TensorRT inference on my GitHub repository [here](https://github.com/dnth/timm_onnx_tensort/blob/main/06_onnx_trt_inference.py).
{{< /notice >}}

That's the end of this post. Or is it?

Not yet. You could stop here and be happy with the results. After all we already got a 84x speedup over the PyTorch model.

But.. if you're like me and you want to squeeze out every last bit of performance, there's one final trick up our sleeve.

### 🧠 Supercharge - Bake pre-processing into ONNX
If you recall, we did our pre-processing transforms outside of the ONNX model in CuPy or Numpy. 

This incurs some overhead because we need to transfer the data to and from the GPU for the transforms.

We can avoid this overhead by baking the transforms operations into the ONNX model. This lets us run the inference faster because we don't need to do the transforms separately.

To do this we need to write some custom code to convert the transforms to an ONNX model.
If you recall, the numpy transforms we used earlier uses the resize and normalization operations. These operations are supported in ONNX and we can add them to the model.

To do this we need to write the preprocessing code as a PyTorch model and export it to ONNX.

```python
import torch.nn as nn

class Preprocess(nn.Module):
    def __init__(self, input_shape: List[int]):
        super(Preprocess, self).__init__()
        self.input_shape = tuple(input_shape)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x: torch.Tensor):
        # Resize the image to the input shape
        x = torch.nn.functional.interpolate(input=x, 
                                           size=self.input_shape[2:], 
                                           mode='bicubic', 
                                           align_corners=False)
        # Normalize the image
        x = x / 255.0
        x = (x - self.mean) / self.std

        return x
```

And now export the `Preprocess` module to ONNX.

```python
input_shape = [1, 3, 448, 448]
output_onnx_file = "preprocessing.onnx"
model = Preprocess(input_shape=input_shape)

torch.onnx.export(
        model,
        torch.randn(input_shape),
        output_onnx_file,
        opset_version=20,
        input_names=["input_rgb"],
        output_names=["output_prep"],
        dynamic_axes={
            "input_rgb": {
                0: "batch_size",
                2: "height",
                3: "width",
            },
        },
    )
```

Let's visualize the preprocess model on Netron.

{{< figure_autoresize src="preprocess_model.png" width="auto" align="center" >}}
Looks like the input and output shapes are correct.

{{< notice note >}}
You can find the code for the export of the preprocessing model on my GitHub repository [here](https://github.com/dnth/timm_onnx_tensort/blob/main/07_export_preprocessing_onnx.py).
{{< /notice >}}

And let's visualize the original model on Netron.

{{< figure_autoresize src="original_model.png" width="auto" align="center" >}}

Now we need to merge the preprocess model with the original model.
Note the name of the input and output nodes from Netron. We will need this for the merge.

To merge the models, we use the `compose` function from the `onnx` library.

```python
import onnx

# Load the models
model1 = onnx.load("preprocessing.onnx")
model2 = onnx.load("eva02_large_patch14_448.onnx")

# Merge the models
merged_model = onnx.compose.merge_models(
    model1,
    model2,
    io_map=[("output_preprocessing", "input")],
    prefix1="preprocessing_",
    prefix2="model_",
    doc_string="Merged preprocessing and eva02_large_patch14_448 model",
    producer_name="dickson.neoh@gmail.com using onnx compose",
)

# Save the merged model
onnx.save(merged_model, "merged_model_compose.onnx")
```
Note the `io_map` parameter. This lets us map the output of the preprocessing model to the input of the original model.

Let's visualize the merged model on Netron.

{{< figure_autoresize src="merged_model.png" width="auto" align="center" >}}

{{< notice note >}}
Note the input to the merged model is `[batch_size, 3, height, width]`. This model can be given any input of size height x width and the batch size can vary. As we've seen in the Preprocess module earlier, the height and width are resized to 448x448 internally.
{{< /notice >}}



And the results are in!

```
TensorRT with pre-processing: 12.875 ms per image, FPS: 77.67
```

{{< notice tip >}}
That's a 6x improvement over the original PyTorch model on the GPU and a whopping 123x improvement over the PyTorch model on the CPU! 🚀
{{< /notice >}}

Let's do a final sanity check on the predictions.

```
>>> espresso: 34.25%
>>> cup: 2.06%
>>> chocolate sauce, chocolate syrup: 1.31%
>>> bakery, bakeshop, bakehouse: 0.97%
>>> coffee mug: 0.85%
```

Looks like the predictions are close to the original model. We can sign off and say that the model is working as expected.

### 🚧 Conclusion

In this post we have seen how we can supercharge our TIMM models for faster inference using ONNX Runtime and TensorRT.


{{< notice tip >}}
In this post you've learned how to:
- 📥 Load any pre-trained model from [TIMM](https://huggingface.co/docs/timm/index)
- 🔄 Convert the model to ONNX format
- 🖥️ Run inference with ONNX Runtime (CPU & GPU)
- 🎮 Run inference with TensorRT (GPU)
- 🛠️ Tweak the TensorRT parameters for better performance
- 🧠 Bake the pre-processing into the ONNX model

You can find the code for this post on my GitHub repository [here](https://github.com/dnth/timm_onnx_tensort).

{{< /notice >}}
