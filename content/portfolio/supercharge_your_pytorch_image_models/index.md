---
title: "Supercharge Your PyTorch Image Models: Bag of Tricks to 8x Faster Inference with ONNX Runtime & Optimizations"
date: 2024-09-09T09:00:00+08:00
featureImage: images/portfolio/supercharge_your_pytorch_image_models/thumbnail.gif
postImage: images/portfolio/supercharge_your_pytorch_image_models/post_image.png
tags: ["TIMM", "ONNX", "TensorRT", "ImageNet", "Hugging Face"]
categories: ["inference", "deployment", "image-classification", "optimization"]
toc: true
socialshare: true
description: Learn how to optimize PyTorch image models using ONNX Runtime and TensorRT, achieving up to 8x faster inference speeds for real-time applications.
images: 
- images/portfolio/supercharge_your_pytorch_image_models/post_image.jpg
---

### 🚀 Motivation
Having real time inference is crucial for many computer vision applications.
In some domain, a 1-second delay in inference could mean life or death.

Imagine you're sitting in a self-driving car and the car takes one full second to detect an oncoming truck.

Just one second too late, and you could end up in the clouds 👼👼👼

Or if you’re really lucky, you get a very up-close view of the pavement.

{{< figure src="banana_peel_robot.gif" width="480" align="center" >}}

I hope that shows you how crucial this problem is.

{{% blockquote %}}
In many high-stake applications, it's not just about being right - it's about being right, right now.
{{% /blockquote %}}

Thus, having real-time inference capability is crucial and will determine whether a model gets deployed or not. 
In many cases, you can pick one or the other:
- A fast model with low accuracy
- A slow model with high accuracy

But can we have the best of both worlds? I.e. a **fast and accurate** model?

That's what this post is about.

{{< notice tip >}}
By the end of the post you'll learn how to supercharge the inference speed of any image models from [TIMM](https://huggingface.co/docs/timm/index) with optimized [ONNX Runtime](https://onnxruntime.ai/) and [TensorRT](https://developer.nvidia.com/tensorrt).

In short:
- 📥 Load any pre-trained model from [TIMM](https://huggingface.co/docs/timm/index).
- 🔄 Convert the model to ONNX format.
- 🖥️ Run inference with [ONNX Runtime](https://onnxruntime.ai/) (CPU & CUDA Provider).
- 🎮 Run inference with [TensorRT](https://developer.nvidia.com/tensorrt) provider and optimized runtime parameters.
- 🧠 Bake the pre-processing into the ONNX model for faster inference.

You can find the code for this post on my GitHub repository [here](https://github.com/dnth/supercharge-your-pytorch-image-models-blogpost).

{{< /notice >}}

If that sounds exciting let's dive in! 🏊‍♂️

### 💻 Installation
Let's begin with the installation.
I will be using a `conda` environment to install the packages required for this post. Feel free to the environment of your choice.

```bash
conda create -n supercharge_timm_tensorrt python=3.11
conda activate supercharge_timm_tensorrt
```

We'll be using the [`timm`](https://github.com/huggingface/pytorch-image-models) library to load a pre-trained model and run inference. So let's install `timm`.

```bash
pip install timm
```
At the time of writing, there are over [1370 models](https://huggingface.co/timm) available in timm. Any of which can be used in this post.

### 🔧 Load and Infer
Let's load a top performing model from the timm [leaderboard](https://huggingface.co/spaces/timm/leaderboard) - the `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` model.

{{< figure_autoresize src="eva_timm.png" width="auto" align="center" >}}

The plot above shows the accuracy vs inference speed for the EVA02 model. 

Look closely, the EVA02 model achieves top ImageNet accuracy (90.05% top-1, 99.06% top-5) but is lags in speed.
Check out the model on the `timm` leaderboard [here](https://huggingface.co/spaces/timm/leaderboard).


So let's get the EVA02 model on our local machine

```python
import timm

model_name = 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'
model = timm.create_model(model_name, pretrained=True).eval()
```

Get the data config and transformations for the model


```python
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
```
And run an inference to get the top 5 predictions

```python
import torch
from PIL import Image
from urllib.request import urlopen

img = Image.open(urlopen('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'))

with torch.inference_mode():
    output = model(transforms(img).unsqueeze(0))

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
```

Next, decode the predictions into class names as a sanity check

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
Looks like the model is doing it's job. 

Now let's benchmark the inference latency.

### ⏱️ Baseline Latency

We will run the inference 10 times and average the inference time.

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

Let's benchmark on CPU and GPU.
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

Although the performance on the GPU is not bad, 12+ FPS is still not fast enough for real-time inference.
On my reasonably modern CPU, it took over 1.5 seconds to run an inference. 

Definitely not self-driving car material 🤷


{{< notice note >}}
I'm using the following hardware for the benchmarks:
- GPU - NVIDIA RTX 3090
- CPU - 11th Gen Intel® Core™ i9-11900 @ 2.50GHz × 16
```
{{< /notice >}}

Now let's start to improve the inference time.

### 🔄 Convert to ONNX
[ONNX](https://onnx.ai/) is an open and interoperable format for deep learning models. It lets us deploy models across different frameworks and devices. 

The key advantage of using ONNX is that it lets us deploy models across different frameworks and devices, and offers some performance gains.

To convert the model to ONNX format, let's first install `onnx`.

```bash
pip install onnx
```

And export the model to ONNX format

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

{{< notice note >}}
Here are the descriptions for the arguments you can pass to the `torch.onnx.export` function:
- `torch.randn(1, 3, 448, 448)`: A dummy input tensor with the appropriate shape.
- `export_params=True`: Whether to export the model parameters.
- `do_constant_folding=True`: Whether to do constant folding for optimization.
- `input_names=['input']`: The name of the input tensor.
- `output_names=['output']`: The name of the output tensor.
- `dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}`: Dynamic axes for the input and output tensors.
```
{{< /notice >}}

If there are no errors, you will end up with a file called `eva02_large_patch14_448.onnx` in your working directory.

{{< notice tip >}}
Inspect and visualize the ONNX model using the [Netron](https://netron.app/) webapp.
{{< /notice >}}

### 🖥️ ONNX Runtime on CPU
To run the and inference on the ONNX model, we need to install `onnxruntime`. This is the 'engine' that will run the ONNX model.

```bash
pip install onnxruntime
```

{{< notice tip >}}
One (major) benefit of using ONNX Runtime is the ability to run the model without PyTorch as a dependency. 

This is great for deployment and for running inference in environments where PyTorch is not available.
{{< /notice >}}

The ONNX model we exported earlier only includes the model weights and the graph structure. It does not include the pre-processing transforms.
To run the inference using `onnxruntime`, we need to replicate the PyTorch transforms.

To find out the transforms that was used, you can print out the `transforms`. 

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

Now let's replicate the transforms using `numpy`.

```python
def transforms_numpy(image: PIL.Image.Image):
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

Using the `numpy`, transforms let's run inference with ONNX Runtime.

{{< highlight python "hl_lines=7" >}}
import onnxruntime as ort

# Create ONNX Runtime session with CPU provider
onnx_filename = "eva02_large_patch14_448.onnx"
session = ort.InferenceSession(
    onnx_filename, 
    providers=["CPUExecutionProvider"] # Run on CPU
)

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
output = session.run([output_name], 
                    {input_name: transforms_numpy(img)})[0]
{{< / highlight >}}


If we inspect the output shape, we can see that it's the same as the number of classes in the ImageNet dataset.

```
output.shape
>>> (1, 1000)
```

And printing the top 5 predictions:

```
>>> espresso: 28.65%
>>> cup: 2.77%
>>> eggnog: 2.28%
>>> chocolate sauce, chocolate syrup: 2.13%
>>> bakery, bakeshop, bakehouse: 1.42%
```

While the results aren't an exact match to the PyTorch model, they're sufficiently similar. This slight variation can be attributed to differences in how normalization is implemented, leading to minor discrepancies in the precise values.


Now let's benchmark the inference latency on ONNX Runtime with a CPU provider (backend).

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
It may seem like a step back, but we are only getting started.

Read on.




### 🖼️ ONNX Runtime on CUDA
Other than the CPU, ONNX Runtime offers other backends for inference. We can easily swap to a different backend by changing the provider. In this case we will use the CUDA backend.

To use the CUDA backend, we need to install the `onnxruntime-gpu` package.
{{< notice warning >}}
You **must** uninstall the `onnxruntime` package before installing the `onnxruntime-gpu` package.

Run the following to uninstall the `onnxruntime` package.
```bash
pip uninstall onnxruntime
```
Then install the `onnxruntime-gpu` package.

```bash
pip install onnxruntime-gpu==1.19.2
```

The `onnxruntime-gpu` package requires a compatible CUDA and cuDNN version. I'm running on `onnxruntime-gpu==1.19.2` at the time of writing this post. This version is compatible with CUDA `12.x` and cuDNN `9.x`.

See the compatibility matrix [here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
{{< /notice >}}


You can install all the CUDA dependencies using conda with the following command.

```bash
conda install -c nvidia cuda=12.2.2 \
                 cuda-tools=12.2.2 \
                 cuda-toolkit=12.2.2 \
                 cuda-version=12.2 \
                 cuda-command-line-tools=12.2.2 \
                 cuda-compiler=12.2.2 \
                 cuda-runtime=12.2.2
```

Once done, replace the CPU provider with the CUDA provider.

{{< highlight python "hl_lines=5 15-17" >}}
onnx_filename = "eva02_large_patch14_448.onnx"

session = ort.InferenceSession(
    onnx_filename, 
    providers=["CUDAExecutionProvider"] # change the provider 
)
{{< / highlight >}}


The rest of the code is the same as the CPU inference. 

Just with one line of code change, the benchmarks are as follows:

```
>>> Onnxruntime CUDA numpy transforms: 56.430 ms per image, FPS: 17.72
```

But that's kinda expected. Running on the GPU, we should expect a speedup.

{{< notice tip >}}
If you encounter the following error:

```bash
Failed to load library libonnxruntime_providers_cuda.so 
with error: libcublasLt.so.12: cannot open shared object 
file: No such file or directory
```
It means that the CUDA library is not in the library path.
You need to export the library path to include the CUDA library.
```bash
export LD_LIBRARY_PATH="/home/dnth/mambaforge-pypy3/envs/supercharge_timm_tensorrt/lib:$LD_LIBRARY_PATH"
```

Replace the `/home/dnth/mambaforge-pypy3/envs/supercharge_timm_tensorrt/lib` with the path to your CUDA library.
{{< /notice >}}

Theres is one more trick we can use to squeeze out more performance - using [CuPy](https://cupy.dev/) for the transforms instead of NumPy.

CuPy is a library that lets us run NumPy code on the GPU. It's a drop-in replacement for NumPy, so you can just replace `numpy` with `cupy` in your code and it will run on the GPU.


Let's install CuPy compatible with our CUDA version.

```bash
pip install cupy-cuda12x
```

And we can use it to run the transforms.
```python
def transforms_cupy(image: PIL.Image.Image):
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



Using ONNX Runtime with CUDA is a little better than the PyTorch model on the GPU, but still not fast enough for real-time inference.

We have one more trick up our sleeve.

### 📊 ONNX Runtime on TensorRT
Similar to the CUDA provider, we have the TensorRT provider on ONNX Runtime. This lets us run the model using the TensorRT high performance inference engine by NVIDIA.

From the [compatibility matrix](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements), we can see that `onnxruntime-gpu==1.19.2` is compatible with TensorRT 10.1.0.

To use the TensorRT provider, you need to have TensorRT installed on your system.

```bash
pip install tensorrt==10.1.0 \
            tensorrt-cu12==10.1.0 \
            tensorrt-cu12-bindings==10.1.0 \
            tensorrt-cu12-libs==10.1.0
```

Next you need to export library path to include the TensorRT library.

```bash
export LD_LIBRARY_PATH="/home/dnth/mambaforge-pypy3/envs/supercharge_timm_tensorrt/python3.11/site-packages/tensorrt_libs:$LD_LIBRARY_PATH"
```
Replace the `/home/dnth/mambaforge-pypy3/envs/supercharge_timm_tensorrt/python3.11/site-packages/tensorrt_libs` with the path to your TensorRT library.

Otherwise you'll encounter the following error:

```bash
Failed to load library libonnxruntime_providers_tensorrt.so 
with error: libnvinfer.so.10: cannot open shared object file: 
No such file or directory
```

Next we need so set the TensorRT provider options in ONNX Runtime inference code.

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
The rest of the code is the same as the CUDA inference.

{{< notice note >}}
Here are the descriptions for the arguments you can pass to the `TensorrtExecutionProvider`:
- `device_id`: 0 - This specifies the GPU device ID to use. In this case, it's set to 0, which typically refers to the first GPU in the system.
- `trt_max_workspace_size`: 8589934592 - This sets the maximum workspace size for TensorRT in bytes. Here, it's set to 8GB, which allows TensorRT to use up to 8GB of GPU memory for its operations.
- `trt_fp16_enable`: True - This enables FP16 (half-precision) mode, which can significantly speed up inference on supported GPUs while reducing memory usage.
- `trt_engine_cache_enable`: True - This enables caching of TensorRT engines, which can speed up subsequent runs by avoiding engine rebuilding.
- `trt_engine_cache_path`: `./trt_cache` - This specifies the directory where TensorRT engine cache files will be stored.
- `trt_force_sequential_engine_build`: False - When set to False, it allows parallel building of TensorRT engines for different subgraphs.
- `trt_max_partition_iterations`: 10000 - This sets the maximum number of iterations for TensorRT to attempt partitioning the graph.
- `trt_min_subgraph_size`: 1 - This specifies the minimum number of nodes required for a subgraph to be considered for conversion to TensorRT.
- `trt_builder_optimization_level`: 5 - This sets the optimization level for the TensorRT builder. Level 5 is the highest optimization level, which can result in longer build times but potentially better performance.
- `trt_timing_cache_enable`: True - This enables the timing cache, which can help speed up engine building by reusing layer timing information from previous builds.
{{< /notice >}}


And now let's run the benchmark:
```
>>> TensorRT + numpy: 18.852 ms per image, FPS: 53.04
>>> TensorRT + cupy: 16.892 ms per image, FPS: 59.20
```

Running with TensorRT and cupy give us a 4.5x speedup over the PyTorch model on the GPU and 93x speedup over the PyTorch model on the CPU!



Thank you for reading this far. That's the end of this post. 

Or is it?

You could stop here and be happy with the results. After all we already got a 93x speedup over the PyTorch model.

But.. if you're like me and you wonder how much more performance we can squeeze out of the model, there's one final trick up our sleeve.

### 🎂 Bake pre-processing into ONNX
If you recall, we did our pre-processing transforms outside of the ONNX model in CuPy or NumPy. 

This incurs some data transfer overhead.
We can avoid this overhead by baking the transforms operations into the ONNX model. 

Okay so how do we do this?

First, we need to write the preprocessing code as a PyTorch module.

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

Let's visualize the exported `preprocessing.onnx` model on Netron.

{{< figure_autoresize src="preprocess_model.png" width="auto" align="center" >}}



{{< notice note >}}
Note the name of the output node of the `preprocessing.onnx` model - `output_preprocessing`.
{{< /notice >}}

Next, let's visualize the original `eva02_large_patch14_448` model on Netron.

{{< figure_autoresize src="original_model.png" width="auto" align="center" >}}

Note the name of the input node of the `eva02_large_patch14_448` model. We will need this for the merge.
The name of the input node is `input`.

Now, we merge the `preprocessing.onnx` model with the `eva02_large_patch14_448` model.
To achieve this, we need to merge the output of the `preprocessing.onnx` model with the input of the `eva02_large_patch14_448` model.

To merge the models, we use the `compose` function from the `onnx` library.

{{< highlight python "hl_lines=4-5 11" >}}
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
{{< / highlight >}}

Note the `io_map` parameter. This lets us map the output of the preprocessing model to the input of the original model. You must ensure that the input and output names of the models are correct.

If there are no errors, you will end up with a file called `merged_model_compose.onnx` in your working directory.
Let's visualize the merged model on Netron.

{{< figure_autoresize src="merged_model.png" width="auto" align="center" >}}

{{< notice tip >}}
The merged model expects an input of size `[batch_size, 3, height, width]`.
This means that the model can take arbitrary input of size height, width and batch size.

{{< /notice >}}

Now using this merged model, let's run the inference benchmark again using the TensorRT provider.

We'll need to make a small change to how the input tensor is passed to the model.

```python
def read_image(image: Image.Image):
    image = image.convert("RGB")
    img_numpy = np.array(image).astype(np.float32)
    img_numpy = img_numpy.transpose(2, 0, 1)
    img_numpy = np.expand_dims(img_numpy, axis=0)
    return img_numpy

```
Notice we are no longer doing the resize and normalization inside the function. This is because the merged model already includes these operations.


And the results are in!

```
TensorRT with pre-processing: 12.875 ms per image, FPS: 77.67
```

That's a 8x improvement over the original PyTorch model on the GPU and a whopping 123x improvement over the PyTorch model on the CPU! 🚀




Let's do a final sanity check on the predictions.

```
>>> espresso: 34.25%
>>> cup: 2.06%
>>> chocolate sauce, chocolate syrup: 1.31%
>>> bakery, bakeshop, bakehouse: 0.97%
>>> coffee mug: 0.85%
```

Looks like the predictions are close to the original model. We can sign off and say that the model is working as expected.

### 🎮 Video Inference

Just for fun, let's see how fast the merged model runs on a video.

{{< video src="inference_video.mp4" alt="Video Inference" autoplay="true" loop="true" >}}

The video inference code is also provided in the [repo](https://github.com/dnth/supercharge-your-pytorch-image-models-blogpost).

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

You can find the code for this post on my GitHub repository [here](https://github.com/dnth/supercharge-your-pytorch-image-models-blogpost).

{{< /notice >}}

There are other things that we've not explored in this post that will likely improve the performance even more. For example,
- Quantization - reducing the precision of the model weights from FP32 to FP16 or INT8 or even lower.
- Pruning - removing the redundant model weights to reduce the model size and improve the inference speed.
- Knowledge distillation - training a smaller model to mimic the original model.

I will leave these as an exercise for the reader. 

Thank you for reading!
I hope this has been helpful. If you'd like to find out how to deploy this model on Android check out the following post.

{{<single_portfolio_item "PyTorch at the Edge: Deploying Over 964 TIMM Models on Android with TorchScript and Flutter">}}


